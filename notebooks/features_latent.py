"""Blok fitur ESTIMASI LATEN (prefix lt_), diturunkan dari investigasi cara data ini dibangkitkan.

Temuan yang mendasari (lihat ringkasan investigasi generator):
- Skor biro = sifat tetap nasabah + noise acak per laporan (SD ≈ 23). Kovarians skor antar-laporan nasabah yang sama
  ≈ 2.300 baik antar-provider di aplikasi yang sama maupun antar-aplikasi berjarak 2 tahun → tidak ada tren; SEMUA
  laporan (provider mana pun, aplikasi mana pun) adalah pengukuran berulang dari satu nilai laten.
- Keterlambatan cicilan ≈ Bernoulli dengan peluang telat per nasabah (ICC ≈ 0,30).
- Premi harga e-commerce (log harga/median kategori) = premi nasabah (SD ≈ 0,22) + noise per item (SD ≈ 0,66),
  ICC ≈ 0,10 → butuh ± 9 item agar setengah variansnya sinyal.
- Pendapatan deklarasi & pendapatan teramati di rekening = dua pengukuran ber-noise dari pendapatan laten.

Rata-rata mentah dari sedikit pengukuran itu ber-noise besar; estimator terbaik (empirical Bayes) menyusutkannya ke rata-rata
populasi sebanding dengan jumlah pengukuran: post = μ + (x̄ − μ) · n / (n + k), k = var_noise / var_laten.
Fitur "_post" adalah rata-rata posterior, "_psd" simpangan baku posterior (seberapa yakin), "_n" jumlah pengukuran.
Tambahan blok gr_ (ramp_features): lonjakan order menjelang keputusan, dibobot eksponensial terhadap baseline sendiri.

Aturan waktu: semua pengukuran per aplikasi hanya dari informasi SEBELUM decision_datetime (biro: available_datetime ≤ cutoff;
e-commerce: transaksi < cutoff; cicilan & rekening: tabel sudah point-in-time, sama seperti features.py).
Hiperparameter (μ, var laten, var noise) diestimasi TANPA target dari pengukuran pra-cutoff aplikasi train, lalu dipakai apa adanya
untuk test — setara dengan fit scaler/median global.
"""
import numpy as np
import polars as pl


def _anova(df, key, val):
    """Komponen varians satu arah: (mean, var antar-kelompok, var dalam-kelompok)."""
    s = df.group_by(key).agg(n=pl.len(), m=pl.col(val).mean(), v=pl.col(val).var())
    N, k = df.height, s.height
    msw = float(((s["n"] - 1) * s["v"].fill_null(0)).sum() / (N - k))
    grand = float(df[val].mean())
    msb = float((s["n"] * (s["m"] - grand) ** 2).sum() / (k - 1))
    n0 = (N - float((s["n"] ** 2).sum()) / N) / (k - 1)
    return grand, max((msb - msw) / n0, 1e-12), msw


def _beta_prior(n, x):
    """Prior Beta(a, b) untuk peluang telat per nasabah (method of moments, beta-binomial)."""
    mu = x.sum() / n.sum()
    ph = x / n
    s2 = np.average((ph - mu) ** 2, weights=n)
    nb = n.mean()
    rho = (s2 * len(n) / (len(n) - 1) - mu * (1 - mu) / nb) / (mu * (1 - mu) * (1 - 1 / nb))
    rho = float(np.clip(rho, 1e-4, 0.999))
    ab = (1 - rho) / rho
    return mu * ab, (1 - mu) * ab


BUREAU_VARS = {"score": "bureau", "utilization": "util", "inquiries_last_6m": "inq", "delinquent_accounts_last_12m": "delinq"}


def measurements(app, T):
    """Tabel pengukuran per aplikasi (long format) yang dipakai untuk fitur & estimasi hiperparameter."""
    cut = app.select("application_id", "customer_id", pl.col("decision_datetime").alias("cutoff"))
    bo = (T["bureau_observations"].join(cut.select("customer_id", "cutoff", pl.col("application_id").alias("app")),
                                        on="customer_id", how="inner")
          .filter(pl.col("available_datetime") <= pl.col("cutoff")))
    rh = (T["repayment_history"].join(cut.select("customer_id", "application_id"), on="customer_id", how="inner")
          .with_columns(late=(pl.col("max_days_past_due") > 0).cast(pl.Float64),
                        dpd=pl.col("max_days_past_due").cast(pl.Float64)))
    ec = T["ecommerce"]
    ec = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    it = (ec.select("customer_id", "transaction_datetime", "product_category", "quantity", "item_total", "order_status")
            .filter(pl.col("order_status") == "completed")
            .join(cut.lazy(), on="customer_id", how="inner")
            .filter(pl.col("transaction_datetime") < pl.col("cutoff"))
            .join(T["ecom_cat_medians"].lazy(), on="product_category", how="left")
            .with_columns(lp=(pl.col("item_total") / pl.col("quantity") / pl.col("cat_median_unit")).log())
            .select("application_id", "lp")
            .collect())
    ab = (T["account_behavior"].join(cut.select("customer_id", "application_id"), on="customer_id", how="inner")
          .filter(pl.col("income_like_credits") > 0)
          .with_columns(li=pl.col("income_like_credits").cast(pl.Float64).log()))
    return dict(bo=bo.rename({"app": "application_id_cur"}), rh=rh, it=it, ab=ab, cut=cut)


def fit_params(app, T, meas=None):
    """Estimasi hiperparameter dari aplikasi train (tanpa target)."""
    m = meas or measurements(app, T)
    P = {}
    for col, nm in BUREAU_VARS.items():
        d = m["bo"].select("application_id_cur", pl.col(col).cast(pl.Float64).alias("v")).drop_nulls("v")
        P[nm] = _anova(d, "application_id_cur", "v")
    g = m["rh"].group_by("application_id").agg(n=pl.len(), x=pl.col("late").sum())
    P["late"] = _beta_prior(g["n"].to_numpy().astype(float), g["x"].to_numpy().astype(float))
    P["dpd"] = _anova(m["rh"], "application_id", "dpd")
    P["prem"] = _anova(m["it"].drop_nulls("lp"), "application_id", "lp")
    # pendapatan: dua indikator (log deklarasi, log rata-rata teramati) → var laten = kovarians keduanya
    inc = (m["ab"].group_by("application_id").agg(lo=pl.col("li").mean(), n=pl.len(), w=pl.col("li").var())
             .join(app.select("application_id", "customer_id"), on="application_id")
             .join(T["customer_profiles"].select("customer_id", pl.col("declared_monthly_income").cast(pl.Float64).log().alias("ld")),
                   on="customer_id"))
    C = np.cov(inc["ld"].to_numpy(), inc["lo"].to_numpy())
    P["income"] = dict(mu_d=float(inc["ld"].mean()), mu_o=float(inc["lo"].mean()), var_lat=float(C[0, 1]),
                       noise_d=float(C[0, 0] - C[0, 1]), noise_o_month=float(inc["w"].drop_nulls().mean()),
                       noise_o_level=float(max(C[1, 1] - C[0, 1] - inc["w"].drop_nulls().mean() / inc["n"].mean(), 1e-4)))
    return P


def _post(mean, n, mu, vb, vw):
    k = vw / vb
    return mu + (mean - mu) * n / (n + k), np.sqrt(1.0 / (1.0 / vb + n / vw))


def latent_features(app, T, P, meas=None):
    """Return app[application_id] + kolom lt_* (urutan baris sama dengan app)."""
    m = meas or measurements(app, T)
    out = app.select("application_id")
    for col, nm in BUREAU_VARS.items():
        mu, vb, vw = P[nm]
        g = (m["bo"].select("application_id_cur", pl.col(col).cast(pl.Float64).alias("v")).drop_nulls("v")
                    .group_by("application_id_cur").agg(n=pl.len(), s=pl.col("v").mean())
                    .rename({"application_id_cur": "application_id"}))
        post, psd = _post(g["s"].to_numpy(), g["n"].to_numpy(), mu, vb, vw)
        cols = {f"lt_{nm}_post": post}
        if nm == "bureau":
            cols |= {"lt_bureau_psd": psd, "lt_bureau_n": g["n"].to_numpy().astype(float)}
        out = out.join(g.select("application_id").with_columns(**{k: pl.Series(v) for k, v in cols.items()}),
                       on="application_id", how="left")
    a, b = P["late"]
    g = m["rh"].group_by("application_id").agg(n=pl.len(), x=pl.col("late").sum(), d=pl.col("dpd").mean())
    n, x = g["n"].to_numpy().astype(float), g["x"].to_numpy().astype(float)
    mu, vb, vw = P["dpd"]
    dpd_post, _ = _post(g["d"].to_numpy(), n, mu, vb, vw)
    pm = (x + a) / (n + a + b)
    out = out.join(g.select("application_id").with_columns(
        lt_late_post=pl.Series(pm),
        lt_late_post_logit=pl.Series(np.log(pm / (1 - pm))),
        lt_late_psd=pl.Series(np.sqrt(pm * (1 - pm) / (n + a + b + 1))),
        lt_late_upper=pl.Series(pm + 1.645 * np.sqrt(pm * (1 - pm) / (n + a + b + 1))),
        lt_dpd_post=pl.Series(dpd_post)), on="application_id", how="left")
    mu, vb, vw = P["prem"]
    g = m["it"].drop_nulls("lp").group_by("application_id").agg(n=pl.len(), s=pl.col("lp").mean())
    post, psd = _post(g["s"].to_numpy(), g["n"].to_numpy(), mu, vb, vw)
    out = out.join(g.select("application_id").with_columns(lt_prem_post=pl.Series(post), lt_prem_psd=pl.Series(psd),
                                                          lt_prem_n=pl.Series(g["n"].to_numpy().astype(float))),
                   on="application_id", how="left")
    # pendapatan laten: gabungan berbobot presisi dari log deklarasi & log rata-rata teramati (noise teramati turun dgn jumlah bulan)
    I = P["income"]
    g = (app.select("application_id", "customer_id", "proposed_monthly_installment")
            .join(T["customer_profiles"].select("customer_id", pl.col("declared_monthly_income").cast(pl.Float64).log().alias("ld")),
                  on="customer_id", how="left")
            .join(m["ab"].group_by("application_id").agg(lo=pl.col("li").mean(), n=pl.len()), on="application_id", how="left"))
    ld, lo, nmo = g["ld"].to_numpy(), g["lo"].to_numpy(), g["n"].fill_null(0).to_numpy().astype(float)
    pd_ = 1.0 / I["noise_d"]
    po = np.where(nmo > 0, 1.0 / (I["noise_o_level"] + I["noise_o_month"] / np.maximum(nmo, 1)), 0.0)
    pl0 = 1.0 / I["var_lat"]
    lat = (pd_ * (ld - I["mu_d"]) + np.where(nmo > 0, po * (np.nan_to_num(lo) - I["mu_o"]), 0.0)) / (pl0 + pd_ + po)
    inc_hat = np.exp(lat + I["mu_d"])
    out = out.with_columns(lt_income_hat=pl.Series(inc_hat), lt_income_psd=pl.Series(np.sqrt(1.0 / (pl0 + pd_ + po))))
    out = (out.join(app.select("application_id", "proposed_monthly_installment"), on="application_id", how="left")
              .with_columns(lt_dti_hat=pl.col("proposed_monthly_installment") / pl.col("lt_income_hat"))
              .drop("proposed_monthly_installment"))
    assert out["application_id"].equals(app["application_id"])
    num = [c for c, t in out.schema.items() if c.startswith("lt_") and t.is_float()]
    return out.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in num])


RAMP_TAUS = (1, 3, 7, 14, 30)


def ramp_features(app, T):
    """Lonjakan order menjelang keputusan (prefix gr_). Generator menaikkan intensitas order sebelum decision_datetime
    — jauh lebih kuat untuk calon bad (×10 baseline 1 hari terakhir, masih ×2 di hari ke-20) — tanpa mengubah komposisi
    kategori / metode bayar / status. Statistik deteksinya: hitungan order berbobot exp(−umur/τ) dibagi laju baseline
    nasabah sendiri (hari 90–365). Hanya order sebelum cutoff."""
    ec = T["ecommerce"]
    ec = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    cut = app.select("application_id", "customer_id", "decision_datetime")
    o = (ec.select("order_id", "customer_id", "transaction_datetime").unique("order_id")
           .join(cut.lazy(), on="customer_id", how="inner")
           .filter(pl.col("transaction_datetime") < pl.col("decision_datetime"))
           .with_columns(age=(pl.col("decision_datetime") - pl.col("transaction_datetime")).dt.total_seconds() / 86400)
           .group_by("application_id")
           .agg(base=pl.col("age").is_between(90, 365).sum() / 275,
                **{f"ew{t}": (-pl.col("age") / t).exp().filter(pl.col("age") < 120).sum() for t in RAMP_TAUS})
           .collect())
    o = o.select("application_id", *[(pl.col(f"ew{t}") / (pl.col("base") * t + 0.05)).alias(f"gr_ramp{t}") for t in RAMP_TAUS])
    out = app.select("application_id").join(o, on="application_id", how="left").with_columns(pl.col("^gr_.*$").fill_null(0.0))
    assert out["application_id"].equals(app["application_id"])
    return out


def add_cross(df):
    """Setelah lt_* digabung ke tabel fitur v1: beban cicilan total terhadap pendapatan laten."""
    return df.with_columns(lt_total_dti_hat=(pl.col("proposed_monthly_installment") + pl.col("ln_active_installment").fill_null(0))
                                            / pl.col("lt_income_hat"))


def latent_columns(df):
    return [c for c in df.columns if c.startswith("lt_")]
