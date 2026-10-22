"""Data latih tambahan dari PINJAMAN LAMA (multi-vintage, notebook 14).

Setiap pinjaman lama yang disetujui (previous_applications.decision == "approved") dan sudah teramati penuh
12 bulan sebelum keputusan aplikasi nasabahnya menjadi satu baris latih tambahan:
- label      : 1 bila ada cicilan pinjaman itu yang DPD-nya mencapai 90 dalam 12 bulan sejak keputusan pinjaman
               (cicilan jatuh tempo + 90 hari ≤ keputusan + 12 bulan dan max_days_past_due ≥ 90) — definisi target
- fitur      : dibangun ulang dengan pipeline features.build_features, tetapi pada CUTOFF LAMA (waktu keputusan
               pinjaman lama), memakai hanya data sebelum cutoff lama:
    * previous_applications : keputusan < cutoff lama
    * repayment_history     : cicilan jatuh tempo < cutoff lama; status bayar/DPD DIREKONSTRUKSI per cutoff lama
                              (kolom asli dicatat per cutoff aplikasi sekarang)
    * account_behavior      : bulan yang sudah selesai sebelum cutoff lama
    * bureau_observations   : available_datetime ≤ cutoff lama (filter bawaan bureau_features); snapshot = laporan
                              yang ditarik untuk aplikasi lama itu
    * ecommerce             : transaksi < cutoff lama (filter bawaan)
    * customer_profiles     : umur dikurangi selisih tahun; kolom lain apa adanya (profil hanya tersedia per
                              keputusan sekarang — keterbatasan yang diketahui)
- loan_purpose tidak ada untuk pinjaman lama → null.

Nasabah yang punya beberapa pinjaman lama diproses per "putaran" (pinjaman ke-1, ke-2, …) agar setiap putaran
memiliki satu cutoff per nasabah, sesuai asumsi fungsi-fungsi di features.py.
"""
import time

import polars as pl

import features as F

DAY = 86400


def hist_loans(pa, rh, apps, min_obs_months=12):
    """Tabel pinjaman lama berlabel. apps: [customer_id, decision_datetime] aplikasi sekarang (train + test)."""
    cur = apps.select("customer_id", pl.col("decision_datetime").alias("current_cutoff"))
    loans = (pa.filter(pl.col("decision") == "approved")
               .join(cur, on="customer_id", how="inner")
               .with_columns(t12=pl.col("decision_datetime").dt.offset_by(f"{min_obs_months}mo"))
               .filter(pl.col("t12") <= pl.col("current_cutoff")))
    lab = (rh.join(loans.select("loan_id", "t12"), on="loan_id", how="inner")
             .group_by("loan_id")
             .agg(target=((pl.col("max_days_past_due") >= 90)
                          & (pl.col("due_date").dt.offset_by("90d") <= pl.col("t12"))).any().cast(pl.Int8),
                  n_inst=pl.len()))
    return (loans.join(lab, on="loan_id", how="inner")
                 .select("application_id", "customer_id", "decision_datetime", "current_cutoff", "loan_id", "term_months",
                         pl.col("approved_amount").alias("loan_amount"), "annual_interest_rate",
                         pl.col("scheduled_monthly_installment").alias("proposed_monthly_installment"),
                         pl.lit(None, dtype=pl.String).alias("loan_purpose"), "target")
                 .sort("customer_id", "decision_datetime")
                 .with_columns(round=pl.col("decision_datetime").rank("ordinal").over("customer_id")))


def repayment_at(rh, cut):
    """Rekonstruksi repayment_history seperti terlihat pada cutoff lama (cut: customer_id, cutoff)."""
    r = rh.join(cut, on="customer_id", how="inner").filter(pl.col("due_date") < pl.col("cutoff"))
    paid_before = pl.col("last_payment_datetime").is_not_null() & (pl.col("last_payment_datetime") < pl.col("cutoff"))
    full = pl.col("amount_paid") >= pl.col("amount_due")
    overdue_days = pl.max_horizontal(((pl.col("cutoff") - pl.col("due_date")).dt.total_seconds() // DAY).cast(pl.Int32), pl.lit(0))
    return (r.with_columns(
                amount_paid=pl.when(paid_before).then(pl.col("amount_paid")).otherwise(0),
                last_payment_datetime=pl.when(paid_before).then(pl.col("last_payment_datetime")).otherwise(None),
                days_past_due=pl.when(paid_before & full).then(0).otherwise(overdue_days).cast(pl.Int32),
                max_days_past_due=pl.when(paid_before & full).then(pl.col("max_days_past_due"))
                                    .otherwise(overdue_days).cast(pl.Int32))
             .drop("cutoff"))


def tables_at(T, cut, years_back):
    """Salinan T (features.load_tables) yang dipotong ke cutoff lama per nasabah (cut: customer_id, cutoff — unik)."""
    pa = (T["previous_applications"].join(cut, on="customer_id", how="inner")
            .filter(pl.col("decision_datetime") < pl.col("cutoff")).drop("cutoff"))
    ab = (T["account_behavior"].join(cut, on="customer_id", how="inner")
            .filter(pl.col("month").cast(pl.Datetime).dt.offset_by("1mo") <= pl.col("cutoff")).drop("cutoff"))
    cp = (T["customer_profiles"].join(years_back, on="customer_id", how="inner")
            .with_columns((pl.col("age") - pl.col("years_back")).alias("age")).drop("years_back"))
    return {**T, "previous_applications": pa, "repayment_history": repayment_at(T["repayment_history"], cut),
            "account_behavior": ab, "customer_profiles": cp}


def build_hist_features(H, T, ecom_batch_size=10_000, verbose=True):
    """Fitur v1 untuk setiap pinjaman lama di H (hasil hist_loans), dihitung pada cutoff lama."""
    parts, t0 = [], time.time()
    for r in sorted(H["round"].unique().to_list()):
        h = H.filter(pl.col("round") == r)
        cut = h.select("customer_id", pl.col("decision_datetime").alias("cutoff"))
        yb = h.select("customer_id", ((pl.col("current_cutoff") - pl.col("decision_datetime")).dt.total_days() // 365)
                      .cast(pl.Int64).alias("years_back"))
        app = h.select("application_id", "customer_id", "decision_datetime", "loan_amount", "annual_interest_rate",
                       "proposed_monthly_installment", "loan_purpose", "target")
        df = F.build_features(app, tables_at(T, cut, yb), ecom_batch_size=ecom_batch_size)
        df = df.join(h.select("application_id", pl.col("term_months").alias("hist_term")), on="application_id", how="left")
        parts.append(df.with_columns(hist_round=pl.lit(r)))
        if verbose:
            print(f"  putaran {r}: {h.height:,} pinjaman | {time.time() - t0:.0f}s", flush=True)
    return pl.concat(parts, how="diagonal_relaxed")
