"""Blok fitur berbasis URUTAN kejadian (notebook 14), terinspirasi solusi top Home Credit Default Risk:
irisan "N terakhir / N pertama" dan lag atribut aplikasi sebelumnya, bukan jendela hari.

Semua fitur dihitung per nasabah dari riwayat SEBELUM decision_datetime aplikasi ini
(tabel previous_applications & repayment_history sudah point-in-time: tidak ada record pada/after cutoff).

Prefix sq_:
- sq_pa_*  aplikasi sebelumnya, diurutkan menurut waktu keputusan
    lag 1..5   : ditolak / dibatalkan (biner)
    lag 1..3   : log jumlah diminta, tenor, suku bunga (bila disetujui), rasio disetujui/diminta, jarak hari ke aplikasi sebelumnya
    irisan     : porsi ditolak di 3 & 5 terakhir dan 2 & 4 pertama; rasio rata-rata jumlah diminta 3 terakhir / 2 pertama;
                 perubahan suku bunga terakhir vs pertama; rata-rata & jarak hari terakhir antar aplikasi
- sq_rp_*  cicilan, diurutkan menurut jatuh tempo (semua pinjaman digabung)
    N terakhir (3, 5, 10): porsi telat, DPD maks, DPD rata-rata, porsi kurang bayar
    tren       : DPD rata-rata 5 terakhir − 5 sebelumnya
    runtun     : jumlah cicilan tepat waktu berturut-turut paling akhir
    awal pinjaman: porsi telat & DPD maks di cicilan ke-1..3 tiap pinjaman (perilaku awal pinjaman)
    pinjaman terakhir: jumlah cicilan, porsi telat, DPD maks, DPD cicilan terakhir, selisih porsi telat vs pinjaman lain
"""
import numpy as np
import polars as pl

DAY = 86400


def _lag(col, k):
    """Nilai ke-k dari belakang (k=1 = terakhir); null kalau riwayat < k."""
    return pl.when(pl.len() >= k).then(pl.col(col).tail(k).first()).otherwise(None)


def prev_app_seq(pa, cut):
    p = (pa.join(cut, on="customer_id", how="inner")
           .sort("customer_id", "decision_datetime")
           .with_columns(rej=(pl.col("decision") == "rejected").cast(pl.Float64),
                         can=(pl.col("decision") == "cancelled").cast(pl.Float64),
                         log_req=pl.col("requested_amount").cast(pl.Float64).log(),
                         ratio=pl.col("approved_amount") / pl.col("requested_amount"),
                         gap=(pl.col("decision_datetime") - pl.col("decision_datetime").shift(1).over("customer_id"))
                             .dt.total_seconds() / DAY))
    aggs = []
    for k in range(1, 6):
        aggs += [_lag("rej", k).alias(f"sq_pa_rej_l{k}"), _lag("can", k).alias(f"sq_pa_can_l{k}")]
    for k in range(1, 4):
        aggs += [_lag("log_req", k).alias(f"sq_pa_logreq_l{k}"), _lag("term_months", k).alias(f"sq_pa_term_l{k}"),
                 _lag("annual_interest_rate", k).alias(f"sq_pa_rate_l{k}"), _lag("ratio", k).alias(f"sq_pa_ratio_l{k}"),
                 _lag("gap", k).alias(f"sq_pa_gap_l{k}")]
    aggs += [
        pl.col("rej").tail(3).mean().alias("sq_pa_rej_last3"),
        pl.col("rej").tail(5).mean().alias("sq_pa_rej_last5"),
        pl.col("rej").head(2).mean().alias("sq_pa_rej_first2"),
        pl.col("rej").head(4).mean().alias("sq_pa_rej_first4"),
        (pl.col("requested_amount").tail(3).mean() / pl.col("requested_amount").head(2).mean()).alias("sq_pa_req_last3_vs_first2"),
        (pl.col("annual_interest_rate").drop_nulls().last() - pl.col("annual_interest_rate").drop_nulls().first())
            .alias("sq_pa_rate_last_minus_first"),
        pl.col("gap").mean().alias("sq_pa_gap_mean"),
    ]
    return p.group_by("customer_id").agg(aggs)


def repayment_seq(rh, cut):
    r = (rh.join(cut, on="customer_id", how="inner")
           .with_columns(late=(pl.col("max_days_past_due") > 0).cast(pl.Float64),
                         under=(pl.col("amount_paid") < pl.col("amount_due")).cast(pl.Float64),
                         dpd=pl.col("max_days_past_due").cast(pl.Float64))
           .with_columns(inst_no=pl.col("due_date").rank("ordinal").over("loan_id"),
                         loan_start=pl.col("due_date").min().over("loan_id"))
           .with_columns(last_loan=pl.col("loan_start") == pl.col("loan_start").max().over("customer_id"))
           .sort("customer_id", "due_date"))
    aggs = []
    for n in (3, 5, 10):
        aggs += [pl.col("late").tail(n).mean().alias(f"sq_rp_late_last{n}"),
                 pl.col("dpd").tail(n).max().alias(f"sq_rp_maxdpd_last{n}"),
                 pl.col("dpd").tail(n).mean().alias(f"sq_rp_meandpd_last{n}"),
                 pl.col("under").tail(n).mean().alias(f"sq_rp_under_last{n}")]
    aggs += [
        pl.when(pl.len() >= 10).then(pl.col("dpd").tail(5).mean() - pl.col("dpd").tail(10).head(5).mean())
          .otherwise(None).alias("sq_rp_dpd_trend5"),
        (pl.col("late").reverse().cum_max() == 0).sum().alias("sq_rp_ontime_streak"),
        pl.col("late").filter(pl.col("inst_no") <= 3).mean().alias("sq_rp_late_first3"),
        pl.col("dpd").filter(pl.col("inst_no") <= 3).max().alias("sq_rp_maxdpd_first3"),
        pl.col("late").filter(pl.col("inst_no") == 1).mean().alias("sq_rp_late_first1"),
        pl.col("last_loan").sum().alias("sq_rp_lastloan_n"),
        pl.col("late").filter(pl.col("last_loan")).mean().alias("sq_rp_lastloan_late"),
        pl.col("dpd").filter(pl.col("last_loan")).max().alias("sq_rp_lastloan_maxdpd"),
        pl.col("dpd").filter(pl.col("last_loan")).last().alias("sq_rp_lastloan_lastdpd"),
        (pl.col("late").filter(pl.col("last_loan")).mean() - pl.col("late").filter(~pl.col("last_loan")).mean())
            .alias("sq_rp_lastloan_late_vs_other"),
    ]
    return r.group_by("customer_id").agg(aggs)


def seq_features(app, pa, rh):
    """1 baris per aplikasi (urutan baris = `app`). app: train/test_applications."""
    cut = app.select("customer_id", pl.col("decision_datetime").alias("cutoff"))
    out = (app.select("application_id", "customer_id", "loan_amount")
              .join(prev_app_seq(pa, cut), on="customer_id", how="left")
              .join(repayment_seq(rh, cut), on="customer_id", how="left")
              .with_columns(sq_pa_loan_vs_req_l1=pl.col("loan_amount").cast(pl.Float64).log() - pl.col("sq_pa_logreq_l1"))
              .drop("loan_amount"))
    assert out["application_id"].equals(app["application_id"])
    num = [c for c, t in out.schema.items() if c.startswith("sq_") and t.is_numeric()]
    return out.with_columns([pl.col(c).cast(pl.Float64) for c in num]).with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in num])


def seq_columns(df):
    return [c for c in df.columns if c.startswith("sq_")]
