"""Kandidat fitur v2 — diuji satu blok per satu blok terhadap baseline v1 (notebook 05).

Setiap blok = satu hipotesis dari EDA 6 (Q8). Semua fungsi mengikuti aturan yang sama dengan features.py:
fitur dihitung HANYA dari kejadian sebelum cutoff (= decision_datetime aplikasi), relatif terhadap cutoff.
Konstanta global (median harga, prior premium) hanya memakai transaksi sebelum keputusan paling awal.

Blok:
- prem2   : price premium lanjutan (shrinkage, porsi di atas median, kuartil atas, tertimbang nilai, tren, COD)
- rhythm  : ritme belanja (jarak antar order normal vs keterbaruan, hari aktif, order 7/14 hari vs baseline)

Disaring dulu secara univariat (KS ≤ noise floor → dibuang dari blok): SD premium, porsi belanja malam,
porsi belanja akhir pekan.
- rp_loan : riwayat cicilan per pinjaman (pinjaman terburuk/terakhir/pertama, cicilan pertama, siklus terlewat)
- cross   : rasio beban lintas sumber (pendapatan teramati vs dideklarasikan, belanja vs arus masuk, total beban)
- bu_tw   : biro tertimbang waktu (laporan terbaru diberi bobot lebih) + rata-rata per provider + kemiringan skor
"""
import polars as pl

from features import DAY, days_between

BLOCKS = ["prem2", "rhythm", "rp_loan", "cross", "bu_tw"]
BLOCK_DESC = {
    "prem2": "price premium lanjutan",
    "rhythm": "ritme belanja",
    "rp_loan": "cicilan per pinjaman",
    "cross": "rasio beban lintas sumber",
    "bu_tw": "biro tertimbang waktu",
}


def _batched(fn, lf, cut, batch_size, **kw):
    """Jalankan fn(lf_subset, cut_subset) per kelompok nasabah agar hemat RAM (pola sama dengan features.py)."""
    if not batch_size or cut.height <= batch_size:
        return cut.select("customer_id").join(fn(lf, cut, **kw), on="customer_id", how="left")
    cs = cut.sort("customer_id")
    parts = []
    for i in range(0, cs.height, batch_size):
        c = cs.slice(i, batch_size)
        parts.append(fn(lf.filter(pl.col("customer_id").is_in(c["customer_id"].to_list())), c, **kw))
    return cut.select("customer_id").join(pl.concat(parts, how="vertical_relaxed"), on="customer_id", how="left")


# ------------------------------------------------------------------------------------------------ prem2
def premium_prior(ec, cat_medians, before):
    """Prior untuk shrinkage premium, dari item completed SEBELUM `before` (tanpa target):
    m = rata-rata log-premium populasi; k = varians dalam-nasabah ÷ varians antar-nasabah.
    Premium nasabah dengan n item ditarik ke m dengan bobot k/(n+k): sedikit item → banyak ditarik."""
    it = (ec.filter((pl.col("transaction_datetime") < before) & (pl.col("order_status") == "completed"))
            .join(cat_medians.lazy(), on="product_category", how="left")
            .with_columns(lp=(pl.col("item_total") / pl.col("quantity") / pl.col("cat_median_unit")).log()))
    cust = it.group_by("customer_id").agg(m=pl.col("lp").mean(), v=pl.col("lp").var(), n=pl.len()).collect()
    var_w = cust.filter(pl.col("n") > 1)["v"].mean()
    var_b = cust["m"].var() - (var_w / cust["n"]).mean()
    return dict(m=float(it.select(pl.col("lp").mean()).collect().item()), k=float(var_w / max(var_b, 1e-6)),
                var_within=float(var_w), var_between=float(var_b))


def _prem2_one(lf, cut, cat_medians, prior):
    m, k = prior["m"], prior["k"]
    it = (lf.join(cut.lazy(), on="customer_id", how="inner")
            .filter((pl.col("transaction_datetime") < pl.col("cutoff")) & (pl.col("order_status") == "completed"))
            .join(cat_medians.lazy(), on="product_category", how="left")
            .with_columns(lp=(pl.col("item_total") / pl.col("quantity") / pl.col("cat_median_unit")).log(),
                          age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / DAY,
                          cod=pl.col("payment_method") == "COD"))
    return (it.group_by("customer_id").agg(
                ec_prem_n_items=pl.len(),
                _s=pl.col("lp").sum(),
                        ec_prem_pos_share=(pl.col("lp") > 0).mean(),
                ec_prem_q75=pl.col("lp").quantile(0.75),
                ec_prem_value_weighted=(pl.col("lp") * pl.col("item_total")).sum() / pl.col("item_total").sum(),
                _m90=pl.col("lp").filter(pl.col("age_d") < 90).mean(),
                _old=pl.col("lp").filter(pl.col("age_d") >= 90).mean(),
                ec_prem_cod=pl.col("lp").filter(pl.col("cod")).mean(),
                ec_prem_noncod=pl.col("lp").filter(~pl.col("cod")).mean(),
            )
            .with_columns(ec_prem_shrunk=(pl.col("_s") + k * m) / (pl.col("ec_prem_n_items") + k),
                          ec_prem_trend_90d=pl.col("_m90") - pl.col("_old"))
            .drop("_s", "_m90", "_old")
            .collect())


def prem2_features(ec, cut, cat_medians, prior, batch_size=10_000):
    return _batched(_prem2_one, ec, cut, batch_size, cat_medians=cat_medians, prior=prior)


# ------------------------------------------------------------------------------------------------ rhythm
def _rhythm_one(lf, cut):
    o = (lf.join(cut.lazy(), on="customer_id", how="inner")
           .filter(pl.col("transaction_datetime") < pl.col("cutoff"))
           .group_by("order_id").agg(pl.col("customer_id", "transaction_datetime", "cutoff").first())
           .with_columns(age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / DAY)
           .sort("customer_id", "transaction_datetime")
           .with_columns(gap=(pl.col("transaction_datetime").diff().dt.total_seconds() / DAY).over("customer_id"),
                         day=pl.col("transaction_datetime").dt.date(),
                         mon=pl.col("transaction_datetime").dt.truncate("1mo")))
    base = pl.col("age_d") >= 30                       # ritme "normal" = di luar 30 hari terakhir
    f = o.group_by("customer_id").agg(
        _since=pl.col("age_d").min(),
        ec_gap_median_base=pl.col("gap").filter(base).median(),
        ec_gap_mean_base=pl.col("gap").filter(base).mean(),
        ec_gap_cv=pl.col("gap").std() / pl.col("gap").mean(),
        ec_gap_max_365d=pl.col("gap").filter(pl.col("age_d") < 365).max(),
        ec_last_gap=pl.col("gap").last(),
        ec_active_days_30d=pl.col("day").filter(pl.col("age_d") < 30).n_unique(),
        ec_active_months_365d=pl.col("mon").filter(pl.col("age_d") < 365).n_unique(),
        _n7=(pl.col("age_d") < 7).sum(),
        _n14=(pl.col("age_d") < 14).sum(),
        _nbase=((pl.col("age_d") >= 90) & (pl.col("age_d") < 365)).sum(),
    )
    rate = pl.col("_nbase") / 275
    return (f.with_columns(ec_recency_vs_rhythm=pl.col("_since") / pl.col("ec_gap_median_base"),
                           ec_last_gap_vs_rhythm=pl.col("ec_last_gap") / pl.col("ec_gap_median_base"),
                           ec_orders_7d_vs_base=pl.col("_n7") / (rate * 7 + 0.1),
                           ec_orders_14d_vs_base=pl.col("_n14") / (rate * 14 + 0.1))
             .drop("_since", "_n7", "_n14", "_nbase")
             .collect())


def rhythm_features(ec, cut, batch_size=10_000):
    return _batched(_rhythm_one, ec, cut, batch_size)


# ------------------------------------------------------------------------------------------------ rp_loan
def rp_loan_features(rh, cut):
    r = (rh.join(cut, on="customer_id", how="inner")
           .with_columns(late=pl.col("max_days_past_due") > 0,
                         age_d=days_between(pl.col("cutoff"), pl.col("due_date")))
           .sort("customer_id", "due_date")
           .with_columns(inst_no=pl.col("due_date").rank("ordinal").over("loan_id"),
                         recent_rank=pl.col("due_date").rank("ordinal", descending=True).over("customer_id"),
                         w=(0.5 ** (pl.col("age_d") / 180))))
    loans = (r.group_by("customer_id", "loan_id").agg(
                 first_due=pl.col("due_date").min(),
                 late_share=pl.col("late").mean(),
                 first_late=pl.col("late").filter(pl.col("inst_no") == 1).first())
             .sort("customer_id", "first_due"))
    per_loan = loans.group_by("customer_id").agg(
        rp_n_loans_rep=pl.len(),
        rp_worst_loan_late_share=pl.col("late_share").max(),
        rp_best_loan_late_share=pl.col("late_share").min(),
        rp_last_loan_late_share=pl.col("late_share").last(),
        rp_first_loan_late_share=pl.col("late_share").first(),
        rp_first_inst_late_share=pl.col("first_late").cast(pl.Float64).mean(),
        rp_share_loans_with_late=(pl.col("late_share") > 0).mean(),
    )
    per_inst = r.group_by("customer_id").agg(
        rp_missed_cycles=(pl.col("max_days_past_due") / 30).floor().sum(),
        rp_late_share_tw=(pl.col("late").cast(pl.Float64) * pl.col("w")).sum() / pl.col("w").sum(),
        rp_dpd_tw=(pl.col("max_days_past_due") * pl.col("w")).sum() / pl.col("w").sum(),
        rp_late_share_last6=pl.col("late").filter(pl.col("recent_rank") <= 6).mean(),
        rp_dpd_mean_last6=pl.col("max_days_past_due").filter(pl.col("recent_rank") <= 6).mean(),
    )
    return (per_loan.join(per_inst, on="customer_id", how="left")
                    .with_columns(rp_last_vs_first_loan=pl.col("rp_last_loan_late_share") - pl.col("rp_first_loan_late_share")))


# ------------------------------------------------------------------------------------------------ cross
def cross_features(df):
    """Rasio dari kolom tabel fitur v1 (+ profil). df = tabel fitur v1 (train atau test)."""
    inc = pl.col("declared_monthly_income")
    return df.select(
        "application_id",
        xs_income_obs_to_declared=pl.col("ac_income_6m") / inc,
        xs_inst_to_ac_inflow=pl.col("proposed_monthly_installment") / pl.col("ac_inflow_3m"),
        xs_ec_spend90_to_ac_inflow=pl.col("ec_spend_90d") / (3 * pl.col("ac_inflow_3m")),
        xs_ec_spend30_to_ac_income=pl.col("ec_spend_30d") / pl.col("ac_income_3m"),
        xs_ec_spend365_to_income=pl.col("ec_spend_365d") / (12 * inc),
        xs_burden_all_to_income=(pl.col("proposed_monthly_installment") + pl.col("ln_active_installment").fill_null(0)
                                 + pl.col("bu_outstanding_max").fill_null(0) / 12) / inc,
        xs_loan_to_ac_balance=pl.col("loan_amount") / (pl.col("ac_avg_bal_3m") + 1),
    )


# ------------------------------------------------------------------------------------------------ bu_tw
def bu_tw_features(bo, cut, half_life=180):
    b = (bo.join(cut, on="customer_id", how="inner")
           .filter(pl.col("available_datetime") <= pl.col("cutoff"))
           .with_columns(age_d=days_between(pl.col("cutoff"), pl.col("report_datetime")).cast(pl.Float64))
           .with_columns(w=0.5 ** (pl.col("age_d") / half_life), t=-pl.col("age_d") / 365))

    def wmean(c):
        ok = pl.col(c).is_not_null()
        return (pl.col(c) * pl.col("w")).filter(ok).sum() / pl.col("w").filter(ok).sum()

    sc = pl.col("score").is_not_null()
    tt, ss = pl.col("t").filter(sc), pl.col("score").filter(sc)
    f = b.group_by("customer_id").agg(
        bu_tw_score=wmean("score"),
        bu_tw_delinq=wmean("delinquent_accounts_last_12m"),
        bu_tw_inq=wmean("inquiries_last_6m"),
        bu_tw_util=wmean("utilization"),
        bu_all_score_min=pl.col("score").min(),
        bu_all_delinq_max=pl.col("delinquent_accounts_last_12m").max(),
        bu_all_score_A=pl.col("score").filter(pl.col("bureau_provider") == "Bureau_A").mean(),
        bu_all_score_B=pl.col("score").filter(pl.col("bureau_provider") == "Bureau_B").mean(),
        bu_all_score_C=pl.col("score").filter(pl.col("bureau_provider") == "Bureau_C").mean(),
        _n=sc.sum(),
        _cov=((tt - tt.mean()) * (ss - ss.mean())).mean(),
        _var=tt.var(ddof=0),
    )
    return (f.with_columns(bu_score_slope_per_year=pl.when((pl.col("_n") >= 3) & (pl.col("_var") > 0))
                                                  .then(pl.col("_cov") / pl.col("_var")))
             .drop("_n", "_cov", "_var"))


# ------------------------------------------------------------------------------------------------ pintu masuk
def build_block(name, app, T, base, prior=None, ecom_batch_size=10_000):
    """Hitung satu blok kandidat untuk `app` (train/test). base = tabel fitur v1 untuk app yang sama.
    Return: application_id + kolom fitur blok (baris sama dengan base)."""
    cut = app.select("customer_id", pl.col("decision_datetime").alias("cutoff"))
    ids = base.select("application_id", "customer_id")
    if name == "prem2":
        f = prem2_features(T["ecommerce"], cut, T["ecom_cat_medians"], prior, batch_size=ecom_batch_size)
    elif name == "rhythm":
        f = rhythm_features(T["ecommerce"], cut, batch_size=ecom_batch_size)
    elif name == "rp_loan":
        f = rp_loan_features(T["repayment_history"], cut)
    elif name == "bu_tw":
        f = bu_tw_features(T["bureau_observations"], cut)
    elif name == "cross":
        return finalize(base.select("application_id").join(cross_features(base), on="application_id", how="left"))
    else:
        raise ValueError(name)
    return finalize(ids.join(f, on="customer_id", how="left").drop("customer_id"))


def finalize(df):
    """Sama seperti v1: boolean → Int8, ±inf/NaN → null."""
    return df.with_columns(
        [pl.col(c).cast(pl.Int8) for c, t in df.schema.items() if t == pl.Boolean]
        + [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
           for c, t in df.schema.items() if t in (pl.Float32, pl.Float64)])
