"""Fungsi pembangun fitur per nasabah (dipakai EDA Bagian 2+ dan nanti feature engineering).

Semua fungsi menerima `cut` = DataFrame [customer_id, cutoff] dengan cutoff = decision_datetime
aplikasi terpilih, dan menghitung fitur HANYA dari informasi sebelum cutoff, relatif terhadap cutoff.
"""
import polars as pl

DAY = 86400


def days_between(later, earlier):
    return ((later - earlier).dt.total_seconds() // DAY).cast(pl.Int64)


def prev_app_features(pa, cut):
    p = (pa.join(cut, on="customer_id", how="inner")
           .with_columns(age_d=days_between(pl.col("cutoff"), pl.col("decision_datetime")),
                         is_app=pl.col("decision") == "approved",
                         is_rej=pl.col("decision") == "rejected",
                         is_can=pl.col("decision") == "cancelled",
                         cut_ratio=pl.col("approved_amount") / pl.col("requested_amount"))
           .sort("customer_id", "decision_datetime"))
    aggs = [
        pl.len().alias("pa_n"),
        pl.col("is_app").sum().alias("pa_n_approved"),
        pl.col("is_rej").sum().alias("pa_n_rejected"),
        pl.col("is_can").sum().alias("pa_n_cancelled"),
        pl.col("is_rej").mean().alias("pa_reject_share"),
        pl.col("is_can").mean().alias("pa_cancel_share"),
        pl.col("age_d").min().alias("pa_days_since_last"),
        pl.col("age_d").max().alias("pa_days_since_first"),
        pl.col("age_d").filter(pl.col("is_rej")).min().alias("pa_days_since_last_reject"),
        pl.col("decision").last().alias("pa_last_decision"),
        pl.col("requested_amount").max().alias("pa_max_requested"),
        pl.col("requested_amount").mean().alias("pa_mean_requested"),
        pl.col("cut_ratio").mean().alias("pa_mean_cut_ratio"),
        pl.col("annual_interest_rate").drop_nulls().last().alias("pa_last_rate"),
        pl.col("annual_interest_rate").mean().alias("pa_mean_rate"),
        pl.col("annual_interest_rate").max().alias("pa_max_rate"),
        ((pl.col("age_d") < 365) & pl.col("is_rej")).sum().alias("pa_n_reject_365d"),
    ] + [(pl.col("age_d") < w).sum().alias(f"pa_n_{w}d") for w in (30, 90, 180, 365)]
    return p.group_by("customer_id").agg(aggs)


def loan_features(pa, rh, cut):
    loans = (pa.filter(pl.col("decision") == "approved")
               .join(cut, on="customer_id", how="inner")
               .join(rh.group_by("loan_id").agg(n_due=pl.len()), on="loan_id", how="left")
               .with_columns(pl.col("n_due").fill_null(0),
                             end=pl.col("decision_datetime").dt.offset_by(pl.format("{}mo", pl.col("term_months"))))
               .with_columns(active=pl.col("end") > pl.col("cutoff"),
                             remaining=pl.col("term_months") - pl.col("n_due")))
    return loans.group_by("customer_id").agg(
        ln_n=pl.len(),
        ln_n_active=pl.col("active").sum(),
        ln_active_installment=(pl.col("scheduled_monthly_installment") * pl.col("active")).sum(),
        ln_remaining_amount=(pl.col("scheduled_monthly_installment") * pl.col("remaining") * pl.col("active")).sum(),
        ln_total_approved=pl.col("approved_amount").sum(),
        ln_days_since_last_loan=days_between(pl.col("cutoff"), pl.col("decision_datetime")).min(),
    )


def repayment_features(rh, cut):
    r = rh.join(cut, on="customer_id", how="inner").with_columns(
        age_d=days_between(pl.col("cutoff"), pl.col("due_date")),
        unpaid=pl.col("amount_paid") < pl.col("amount_due"),
        partial=(pl.col("amount_paid") > 0) & (pl.col("amount_paid") < pl.col("amount_due")),
        late=pl.col("max_days_past_due") > 0,
    )
    aggs = [
        pl.len().alias("rp_n_inst"),
        pl.col("late").mean().alias("rp_late_share"),
        pl.col("max_days_past_due").mean().alias("rp_mean_dpd"),
        pl.col("max_days_past_due").max().alias("rp_max_dpd_ever"),
        (pl.col("max_days_past_due") >= 30).sum().alias("rp_n_dpd30"),
        (pl.col("max_days_past_due") >= 60).sum().alias("rp_n_dpd60"),
        (pl.col("max_days_past_due") >= 90).sum().alias("rp_n_dpd90"),
        (pl.col("amount_paid").sum() / pl.col("amount_due").sum()).alias("rp_paid_ratio"),
        pl.col("partial").sum().alias("rp_n_partial"),
        pl.col("days_past_due").max().alias("rp_current_dpd"),
        pl.col("unpaid").sum().alias("rp_n_unpaid_now"),
        (pl.col("amount_due") - pl.col("amount_paid")).sum().alias("rp_overdue_amount"),
        pl.col("age_d").filter(pl.col("late")).min().alias("rp_days_since_last_late"),
    ]
    for w in (90, 180, 365):
        m = pl.col("age_d") < w
        aggs += [pl.col("late").filter(m).mean().alias(f"rp_late_share_{w}d"),
                 pl.col("max_days_past_due").filter(m).max().alias(f"rp_max_dpd_{w}d")]
    return r.group_by("customer_id").agg(aggs)


def bureau_features(bo, cut, income=None):
    """Fitur biro kredit.
    - bu_*      : snapshot laporan untuk APLIKASI INI (1–3 provider)
    - bu_hist_* : laporan yang ditarik untuk pengajuan LAMA (snapshot masa lalu)
    - bu_all_*  : rata-rata SEMUA laporan (ini + lama); meredam noise kolom yang berisik
    cut harus berisi application_id, customer_id, cutoff. income (opsional): customer_id, declared_monthly_income."""
    b = (bo.join(cut.select("customer_id", "cutoff"), on="customer_id", how="inner")
           .filter(pl.col("available_datetime") <= pl.col("cutoff"))      # aturan ketersediaan (audit: sudah terpenuhi)
           .with_columns(age_d=days_between(pl.col("cutoff"), pl.col("report_datetime"))))
    sel_ids = cut.select("application_id")
    cur = b.join(sel_ids, on="application_id", how="semi")
    old = b.join(sel_ids, on="application_id", how="anti")
    snap = cur.group_by("customer_id").agg(
        bu_n_reports=pl.len(),
        bu_n_unscored=pl.col("score").is_null().sum(),
        bu_score_mean=pl.col("score").mean(),
        bu_score_min=pl.col("score").min(),
        bu_score_max=pl.col("score").max(),
        bu_score_A=pl.col("score").filter(pl.col("bureau_provider") == "Bureau_A").first(),
        bu_active_max=pl.col("active_accounts").max(),
        bu_outstanding_max=pl.col("outstanding_balance").max(),
        bu_outstanding_mean=pl.col("outstanding_balance").mean(),
        bu_rev_limit_max=pl.col("revolving_credit_limit").max(),
        bu_rev_out_max=pl.col("revolving_outstanding_balance").max(),
        bu_util_mean=pl.col("utilization").mean(),
        bu_has_revolving=(pl.col("revolving_credit_limit") > 0).any(),
        bu_inq_mean=pl.col("inquiries_last_6m").mean(),
        bu_inq_max=pl.col("inquiries_last_6m").max(),
        bu_delinq_mean=pl.col("delinquent_accounts_last_12m").mean(),
        bu_delinq_max=pl.col("delinquent_accounts_last_12m").max(),
        bu_report_age_min=pl.col("age_d").min(),
    )
    hist = (old.sort("customer_id", "report_datetime").group_by("customer_id").agg(
        bu_hist_n=pl.len(),
        bu_hist_score_mean=pl.col("score").mean(),
        bu_hist_score_first=pl.col("score").drop_nulls().first(),
        bu_hist_delinq_mean=pl.col("delinquent_accounts_last_12m").mean(),
        bu_hist_inq_mean=pl.col("inquiries_last_6m").mean(),
        bu_hist_outstanding_mean=pl.col("outstanding_balance").mean(),
    ))
    allr = b.group_by("customer_id").agg(
        bu_all_n=pl.len(),
        bu_all_score_mean=pl.col("score").mean(),
        bu_all_delinq_mean=pl.col("delinquent_accounts_last_12m").mean(),
        bu_all_inq_mean=pl.col("inquiries_last_6m").mean(),
        bu_all_util_mean=pl.col("utilization").mean(),
    )
    f = (cut.select("customer_id")
            .join(snap, on="customer_id", how="left")
            .join(hist, on="customer_id", how="left")
            .join(allr, on="customer_id", how="left")
            .with_columns(bu_score_trend=pl.col("bu_score_mean") - pl.col("bu_hist_score_mean"),
                          bu_score_change_first=pl.col("bu_score_mean") - pl.col("bu_hist_score_first")))
    if income is not None:
        f = (f.join(income, on="customer_id", how="left")
              .with_columns(bu_outstanding_to_income=pl.col("bu_outstanding_max") / (pl.col("declared_monthly_income") * 12),
                            bu_rev_out_to_income=pl.col("bu_rev_out_max") / pl.col("declared_monthly_income"))
              .drop("declared_monthly_income"))
    return f


def account_features(ab, cut):
    """Fitur rekening bulanan. Semua bulan di `ab` sudah lengkap sebelum cutoff (audit).
    k = urutan bulan dari yang terbaru (k=1 = bulan terakhir sebelum bulan keputusan).
    - ac_*_{3m,6m,all} : level saldo / arus kas per jendela waktu
    - ac_*_surge_*     : bulan-bulan terakhir dibanding BASELINE nasabah sendiri (rata-rata bulan k ≥ 4)
    - ac_*_cv          : volatilitas"""
    a = (ab.join(cut, on="customer_id", how="inner")
           .sort("customer_id", "month")
           .with_columns(k=pl.col("month").rank("ordinal", descending=True).over("customer_id"),
                         net=pl.col("inflows") - pl.col("outflows"),
                         days=pl.col("month").dt.month_end().dt.day()))
    k1, base = pl.col("k") == 1, pl.col("k") >= 4
    aggs = [pl.len().alias("ac_n_months")]
    for w, tag in [(3, "3m"), (6, "6m"), (999, "all")]:
        m = pl.col("k") <= w
        aggs += [
            pl.col("average_balance").filter(m).mean().alias(f"ac_avg_bal_{tag}"),
            pl.col("minimum_balance").filter(m).min().alias(f"ac_min_bal_{tag}"),
            pl.col("income_like_credits").filter(m).mean().alias(f"ac_income_{tag}"),
            pl.col("inflows").filter(m).mean().alias(f"ac_inflow_{tag}"),
            pl.col("outflows").filter(m).mean().alias(f"ac_outflow_{tag}"),
            (pl.col("outflows").filter(m).sum() / pl.col("inflows").filter(m).sum()).alias(f"ac_out_in_ratio_{tag}"),
            (pl.col("low_balance_days").filter(m).sum() / pl.col("days").filter(m).sum()).alias(f"ac_low_days_share_{tag}"),
            (pl.col("net") < 0).filter(m).mean().alias(f"ac_neg_month_share_{tag}"),
        ]
    aggs += [
        # lonjakan relatif terhadap baseline nasabah sendiri
        (pl.col("outflows").filter(k1).first() / pl.col("outflows").filter(base).mean()).alias("ac_out_surge_1m"),
        (pl.col("outflows").filter(k1).first() / pl.col("outflows").filter(base).median()).alias("ac_out_surge_1m_med"),
        (pl.col("outflows").filter(pl.col("k") == 2).first() / pl.col("outflows").filter(base).mean()).alias("ac_out_surge_2m"),
        (pl.col("outflows").filter(pl.col("k") <= 3).mean() / pl.col("outflows").filter(base).mean()).alias("ac_out_surge_3m"),
        ((pl.col("outflows").filter(k1).first() - pl.col("outflows").filter(base).mean())
         / pl.col("income_like_credits").filter(base).mean()).alias("ac_excess_out_1m_vs_income"),
        (pl.col("outflows").filter(k1).first() / pl.col("inflows").filter(k1).first()).alias("ac_out_in_ratio_1m"),
        (pl.col("inflows").filter(k1).first() / pl.col("inflows").filter(base).mean()).alias("ac_in_surge_1m"),
        (pl.col("income_like_credits").filter(k1).first() / pl.col("income_like_credits").filter(base).mean()).alias("ac_income_change_1m"),
        (pl.col("average_balance").filter(k1).first() / (pl.col("average_balance").filter(base).mean() + 1)).alias("ac_balance_change_1m"),
        (pl.col("low_balance_days").filter(k1).first() - pl.col("low_balance_days").filter(base).mean()).alias("ac_low_days_change_1m"),
        # volatilitas
        (pl.col("income_like_credits").std() / pl.col("income_like_credits").mean()).alias("ac_income_cv"),
        (pl.col("outflows").filter(base).std() / pl.col("outflows").filter(base).mean()).alias("ac_outflow_cv_base"),
        (pl.col("average_balance").std() / (pl.col("average_balance").mean() + 1)).alias("ac_balance_cv"),
        pl.col("closing_balance").filter(k1).first().alias("ac_last_closing"),
    ]
    return a.group_by("customer_id").agg(aggs)


ECOM_CATEGORIES = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
ECOM_PAYMENTS = ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]


def ecommerce_category_medians(ec, before=None):
    """Median harga per unit tiap kategori dari item completed (tanpa target) — konstanta normalisasi global
    yang sama untuk train & test. `before` (datetime): pakai hanya transaksi sebelum tanggal ini. Dengan
    before = keputusan paling awal di train+test, konstanta ini 'tersedia sebelum' SETIAP decision_datetime."""
    lf = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    if before is not None:
        lf = lf.filter(pl.col("transaction_datetime") < before)
    return (lf.filter(pl.col("order_status") == "completed")
              .with_columns(unit=pl.col("item_total") / pl.col("quantity"))
              .group_by("product_category").agg(cat_median_unit=pl.col("unit").median())
              .collect())


def ecommerce_features(ec, cut, cat_medians=None, batch_size=None):
    """Pembungkus hemat memori: bila `batch_size` diisi, nasabah diproses per kelompok (mis. 10.000) dan hanya
    transaksi milik kelompok itu yang dibaca dari parquet (filter `is_in` didorong ke pembacaan file).
    Hasil identik dengan sekali jalan; puncak RAM jauh lebih kecil untuk file e-commerce penuh."""
    if cat_medians is None:
        cat_medians = ecommerce_category_medians(ec)
    if not batch_size or cut.height <= batch_size:
        return _ecommerce_features_one(ec, cut, cat_medians)
    lf = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    cut_sorted = cut.sort("customer_id")          # file parquet terurut per nasabah → baca blok yang berdekatan
    parts = []
    for i in range(0, cut_sorted.height, batch_size):
        c = cut_sorted.slice(i, batch_size)
        sub = lf.filter(pl.col("customer_id").is_in(c["customer_id"].to_list()))
        parts.append(_ecommerce_features_one(sub, c, cat_medians))
    out = pl.concat(parts, how="vertical_relaxed")
    return cut.select("customer_id").join(out, on="customer_id", how="left")   # urutan & baris sama seperti `cut`


def _ecommerce_features_one(ec, cut, cat_medians=None):
    """Fitur e-commerce. `ec` boleh LazyFrame (disarankan; tabel terbesar) atau DataFrame.
    - order di-dedup dulu (order_total diulang per item); belanja = hanya order COMPLETED
    - jendela waktu dalam HARI sebelum cutoff (7/14/30/60/90/180/365)
    - baseline = hari ke-90 s.d. ke-365 sebelum cutoff → lonjakan 30 hari terakhir relatif kebiasaan sendiri
    Catatan sampling: fitur hitungan/jumlah rupiah bergantung pada versi data (sampling vs penuh);
    pastikan train & test dihitung dari file yang SAMA."""
    lf = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    items = (lf.join(cut.lazy(), on="customer_id", how="inner")
               .filter(pl.col("transaction_datetime") < pl.col("cutoff"))
               .with_columns(age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / DAY))
    orders = (items.group_by("order_id")
                   .agg(pl.col("customer_id", "order_total", "payment_method", "order_status", "age_d").first(),
                        n_items=pl.len(), qty=pl.col("quantity").sum())
                   .with_columns(done=pl.col("order_status") == "completed"))
    done = pl.col("done")
    aggs = [
        pl.len().alias("ec_n_orders_all"),
        pl.col("order_total").filter(done).sum().alias("ec_spend_all"),
        pl.col("age_d").min().alias("ec_days_since_last_order"),
        pl.col("age_d").filter(done).min().alias("ec_days_since_last_completed"),
        pl.col("age_d").max().alias("ec_history_days"),
        pl.col("order_total").filter(done).mean().alias("ec_aov_all"),
        pl.col("order_total").filter(done & (pl.col("age_d") < 30)).mean().alias("ec_aov_30d"),
        pl.col("order_total").filter(done & (pl.col("age_d") >= 90) & (pl.col("age_d") < 365)).mean().alias("ec_aov_base"),
        pl.col("n_items").mean().alias("ec_items_per_order"),
        (~done).mean().alias("ec_cancel_share_all"),
        (~done).filter(pl.col("age_d") < 90).mean().alias("ec_cancel_share_90d"),
        (pl.col("payment_method") == "COD").filter(pl.col("age_d") < 90).mean().alias("ec_cod_share_90d"),
        pl.col("order_total").filter(done & (pl.col("age_d") >= 90) & (pl.col("age_d") < 365)).sum().alias("ec_spend_base"),
        (done & (pl.col("age_d") >= 90) & (pl.col("age_d") < 365)).sum().alias("ec_n_base"),
    ]
    aggs += [(pl.col("payment_method") == p).mean().alias(f"ec_pay_share_{p}") for p in ECOM_PAYMENTS]
    for w in (7, 14, 30, 60, 90, 180, 365):
        m = pl.col("age_d") < w
        aggs += [pl.col("order_total").filter(done & m).sum().alias(f"ec_spend_{w}d"),
                 (done & m).sum().alias(f"ec_n_{w}d")]
    f_orders = orders.group_by("customer_id").agg(aggs)
    f_cat = (items.filter(pl.col("order_status") == "completed").group_by("customer_id")
                  .agg([(pl.col("item_total").filter(pl.col("product_category") == k).sum() / pl.col("item_total").sum())
                        .alias(f"ec_cat_share_{k}") for k in ECOM_CATEGORIES]))
    # price premium: harga per unit dibanding median kategorinya (log), dirata-rata per nasabah
    if cat_medians is None:
        cat_medians = ecommerce_category_medians(ec)
    f_prem = (items.filter(pl.col("order_status") == "completed")
                   .join(cat_medians.lazy(), on="product_category", how="left")
                   .with_columns(lp=(pl.col("item_total") / pl.col("quantity") / pl.col("cat_median_unit")).log())
                   .group_by("customer_id")
                   .agg(ec_price_premium_mean=pl.col("lp").mean(),
                        ec_price_premium_median=pl.col("lp").median(),
                        ec_price_premium_30d=pl.col("lp").filter(pl.col("age_d") < 30).mean(),
                        ec_price_premium_base=pl.col("lp").filter(pl.col("age_d") >= 90).mean()))
    base_days = 365 - 90
    f = (cut.select("customer_id").lazy()
            .join(f_orders, on="customer_id", how="left")
            .join(f_cat, on="customer_id", how="left")
            .join(f_prem, on="customer_id", how="left")
            .with_columns(pl.col("^ec_spend_.*$", "^ec_n_.*$").fill_null(0))
            .with_columns(
                ec_spend_surge_30d=pl.col("ec_spend_30d") / (pl.col("ec_spend_base") / base_days * 30 + 1),
                ec_order_surge_30d=pl.col("ec_n_30d") / (pl.col("ec_n_base") / base_days * 30 + 0.1),
                ec_spend_surge_90d=pl.col("ec_spend_90d") / (pl.col("ec_spend_base") / base_days * 90 + 1),
                ec_spend_30d_share_of_year=pl.col("ec_spend_30d") / (pl.col("ec_spend_365d") + 1),
                ec_aov_change_30d=pl.col("ec_aov_30d") / pl.col("ec_aov_base"),
            ))
    return f.collect()


# =====================================================================================
# Pembangun tabel fitur lengkap (train ATAU test) — satu pintu masuk untuk modeling
# =====================================================================================
APP_CATEGORICAL = ["loan_purpose", "region", "employment_type", "housing_status", "pa_last_decision", "segment"]
ID_COLS = ["application_id", "customer_id"]


def load_tables(load, lazy):
    """Muat semua tabel pendukung sekali. `load`/`lazy` = fungsi baca parquet (mis. eda_utils.load / lazy)."""
    ec = lazy("ecommerce_transactions")
    first_decision = min(load(n, columns=["decision_datetime"])["decision_datetime"].min()
                         for n in ("train_applications", "test_applications"))
    return dict(
        customer_profiles=load("customer_profiles"),
        previous_applications=load("previous_applications"),
        repayment_history=load("repayment_history"),
        bureau_observations=load("bureau_observations"),
        account_behavior=load("account_behavior"),
        ecommerce=ec,
        # konstanta global (sama untuk train & test), hanya dari transaksi sebelum keputusan paling awal
        ecom_cat_medians=ecommerce_category_medians(ec, before=first_decision),
        first_decision=first_decision,
    )


def build_features(app, T, ecom_batch_size=10_000):
    """app: train_applications atau test_applications. T: hasil load_tables().
    Mengembalikan 1 baris per aplikasi: ID (+ target bila ada) + semua fitur.
    Semua fitur riwayat dihitung HANYA dari informasi sebelum decision_datetime aplikasi tsb.
    ecom_batch_size: jumlah nasabah per kelompok saat menghitung fitur e-commerce (perkecil, mis. 5_000,
    bila RAM terbatas; None = sekaligus, paling cepat tetapi paling boros memori)."""
    cut_full = app.select("application_id", "customer_id", pl.col("decision_datetime").alias("cutoff"))
    cut = cut_full.select("customer_id", "cutoff")
    cp = T["customer_profiles"]
    keep = ID_COLS + (["target"] if "target" in app.columns else []) + [
        "loan_amount", "annual_interest_rate", "proposed_monthly_installment", "loan_purpose"]
    df = (app.select(keep)
             .join(cp, on="customer_id", how="left")
             .join(prev_app_features(T["previous_applications"], cut), on="customer_id", how="left")
             .join(loan_features(T["previous_applications"], T["repayment_history"], cut), on="customer_id", how="left")
             .join(repayment_features(T["repayment_history"], cut), on="customer_id", how="left")
             .join(bureau_features(T["bureau_observations"], cut_full,
                                   income=cp.select("customer_id", "declared_monthly_income")), on="customer_id", how="left")
             .join(account_features(T["account_behavior"], cut), on="customer_id", how="left")
             .join(ecommerce_features(T["ecommerce"], cut, cat_medians=T["ecom_cat_medians"], batch_size=ecom_batch_size),
                   on="customer_id", how="left"))
    total_inst = pl.col("proposed_monthly_installment") + pl.col("ln_active_installment").fill_null(0)
    df = df.with_columns(
        # aplikasi
        dti=pl.col("proposed_monthly_installment") / pl.col("declared_monthly_income"),
        lti=pl.col("loan_amount") / (pl.col("declared_monthly_income") * 12),
        income_per_capita=pl.col("declared_monthly_income") / (1 + pl.col("dependents")),
        # riwayat internal
        total_dti=total_inst / pl.col("declared_monthly_income"),
        rate_change_vs_last=(pl.col("annual_interest_rate") - pl.col("pa_last_rate")).round(2),
        amount_vs_max_requested=pl.col("loan_amount") / pl.col("pa_max_requested"),
        rp_is_current=(pl.col("rp_n_unpaid_now") == 0).cast(pl.Int8),
        rp_dpd_ge30_now=(pl.col("rp_current_dpd") >= 30).cast(pl.Int8),
        # rekening
        ac_total_inst_to_income=total_inst / pl.col("ac_income_6m"),
        ac_buffer_months=pl.col("ac_avg_bal_3m") / total_inst,
        ac_net_after_installment=(pl.col("ac_inflow_6m") - pl.col("ac_outflow_6m") - pl.col("proposed_monthly_installment"))
                                 / pl.col("ac_income_6m"),
        # e-commerce lintas sumber
        ec_spend_30d_to_income=pl.col("ec_spend_30d") / pl.col("declared_monthly_income"),
        # segmen
        segment=pl.when(pl.col("pa_n").is_null()).then(pl.lit("A_no_history"))
                  .when(pl.col("rp_n_inst").is_null()).then(pl.lit("B_apps_no_installments"))
                  .otherwise(pl.lit("C_has_installments")),
    )
    # null rp_* untuk nasabah tanpa cicilan → biarkan null (GBDT menanganinya); flag di atas ikut null
    # Rapikan tipe: boolean → Int8; ±inf / NaN (pembagian nol) → null, agar aman untuk semua model
    df = df.with_columns(
        [pl.col(c).cast(pl.Int8) for c, t in df.schema.items() if t == pl.Boolean]
        + [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
           for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)]
    )
    return df


def feature_columns(df):
    """Semua kolom fitur (tanpa ID & target)."""
    return [c for c in df.columns if c not in ID_COLS + ["target"]]


FEATURE_GROUPS = ["aplikasi_profil", "internal", "biro", "rekening", "ecommerce"]
FEATURE_GROUPS_V2 = FEATURE_GROUPS + ["lintas"]          # + rasio lintas sumber (prefix xs_, features_v2.py)
_INTERNAL_CROSS = {"total_dti", "rate_change_vs_last", "amount_vs_max_requested", "segment"}


def feature_group(col):
    """Sumber data sebuah fitur (untuk ringkasan importance / ablation)."""
    if col.startswith(("pa_", "ln_", "rp_")) or col in _INTERNAL_CROSS:
        return "internal"
    if col.startswith("bu_"):
        return "biro"
    if col.startswith("ac_"):
        return "rekening"
    if col.startswith("ec_"):
        return "ecommerce"
    if col.startswith(("xs_", "sn_", "lt_")):
        return "lintas"
    return "aplikasi_profil"