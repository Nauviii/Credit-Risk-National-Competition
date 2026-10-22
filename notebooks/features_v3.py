"""Kandidat fitur dari diskusi 'reverse engineering generator' (notebook 08).

Dua blok:
- snap   : residu terhadap aturan grid generator. Suku bunga hanya punya 5 tingkat (16–32%) dan sangat ditentukan
           skor biro. Selisih antara tingkat AKTUAL dan tingkat yang DIHARAPKAN dari skor = informasi privat
           pemberi pinjaman yang tidak terlihat di skor. Peta "skor → suku bunga yang diharapkan" dipelajari HANYA
           dari pengajuan lama yang diputuskan SEBELUM keputusan paling awal (1 Jan 2025) → patuh aturan cutoff.
- latent : estimasi sifat laten yang menggerakkan banyak fitur sekaligus ("perilaku" dan "kelayakan kredit").
           Setiap proksi → skor normal (kuantil), lalu digabung dengan bobot komponen utama pertama (PCA) dari
           korelasi antar-proksi. TANPA target: transformasi & bobot di-fit di data train saja (seperti melatih
           scaler), lalu diterapkan ke test.
"""
import numpy as np
import polars as pl

from features import days_between

TIER = 0.04            # jarak antar tingkat suku bunga (16, 20, 24, 28, 32%)


# ------------------------------------------------------------------------------------------------ snap
def pricing_map(pa, bo, before, bin_width=5, min_n=30):
    """Peta skor biro rata-rata pengajuan → suku bunga rata-rata, dari pengajuan DISETUJUI sebelum `before`.
    Return dict(centers, rates, null_rate, table)."""
    sc = bo.group_by("application_id").agg(sc_mean=pl.col("score").mean())
    j = (pa.filter((pl.col("decision_datetime") < before) & pl.col("annual_interest_rate").is_not_null())
           .join(sc, on="application_id", how="left"))
    null_rate = float(j.filter(pl.col("sc_mean").is_null())["annual_interest_rate"].mean())
    tab = (j.drop_nulls("sc_mean")
            .with_columns(b=(pl.col("sc_mean") / bin_width).floor() * bin_width + bin_width / 2)
            .group_by("b").agg(n=pl.len(), rate=pl.col("annual_interest_rate").mean())
            .filter(pl.col("n") >= min_n).sort("b"))
    return dict(centers=tab["b"].to_numpy(), rates=tab["rate"].to_numpy(), null_rate=null_rate, table=tab,
                n_apps=j.height)


def expected_rate(score, pmap):
    s = np.asarray(score, dtype=float)
    out = np.interp(s, pmap["centers"], pmap["rates"])          # di luar rentang: nilai ujung
    return np.where(np.isnan(s), pmap["null_rate"], out)


def snap_features(app_feats, pa, bo, cut, pmap):
    """app_feats: tabel fitur v1 (butuh annual_interest_rate, bu_score_mean). cut: customer_id, cutoff.
    Residu aplikasi ini + residu historis (pengajuan lama yang disetujui, sebelum cutoff)."""
    exp_now = expected_rate(app_feats["bu_score_mean"].to_numpy(), pmap)
    now = app_feats.select("application_id", "customer_id", "annual_interest_rate").with_columns(
        sn_rate_expected=pl.Series(exp_now))
    now = now.with_columns(
        sn_rate_resid=pl.col("annual_interest_rate") - pl.col("sn_rate_expected"),
        sn_tier_resid=((pl.col("annual_interest_rate") - pl.col("sn_rate_expected")) / TIER).round(0))
    # historis: setiap pengajuan lama yang disetujui → residu terhadap skornya SAAT ITU
    sc = bo.group_by("application_id").agg(sc_mean=pl.col("score").mean())
    hist = (pa.filter(pl.col("annual_interest_rate").is_not_null())
              .join(cut, on="customer_id", how="inner")
              .filter(pl.col("decision_datetime") < pl.col("cutoff"))
              .join(sc, on="application_id", how="left")
              .sort("customer_id", "decision_datetime"))
    hist = hist.with_columns(exp=pl.Series(expected_rate(hist["sc_mean"].to_numpy(), pmap)))
    hist = hist.with_columns(resid=pl.col("annual_interest_rate") - pl.col("exp"),
                             age_d=days_between(pl.col("cutoff"), pl.col("decision_datetime")))
    h = hist.group_by("customer_id").agg(
        sn_hist_resid_mean=pl.col("resid").mean(),
        sn_hist_resid_last=pl.col("resid").last(),
        sn_hist_resid_max=pl.col("resid").max(),
        sn_hist_n=pl.len())
    out = (now.join(h, on="customer_id", how="left")
              .with_columns(sn_resid_vs_hist=pl.col("sn_rate_resid") - pl.col("sn_hist_resid_mean"))
              .drop("customer_id", "annual_interest_rate"))
    return out


# ------------------------------------------------------------------------------------------------ latent
BEHAVIOR = ["ec_price_premium_mean", "ec_price_premium_median", "rp_late_share", "rp_mean_dpd",
            "ec_order_surge_30d", "ec_spend_surge_30d", "ec_days_since_last_order", "ac_out_surge_1m"]
CREDIT = ["bu_all_score_mean", "bu_score_mean", "annual_interest_rate", "bu_all_delinq_mean",
          "bu_all_inq_mean", "bu_util_mean"]
ANCHOR = {"behavior": "ec_price_premium_mean", "credit": "annual_interest_rate"}


def _normal_scores(x, ref):
    """Kuantil empiris dari `ref` → skor normal standar. Null tetap null. (Tanpa target.)"""
    from scipy.stats import norm
    ref = np.sort(ref[~np.isnan(ref)])
    ok = ~np.isnan(x)
    out = np.full(len(x), np.nan)
    lo = np.searchsorted(ref, x[ok], side="left"); hi = np.searchsorted(ref, x[ok], side="right")
    u = ((lo + hi) / 2 + 0.5) / (len(ref) + 1)
    out[ok] = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    return out


def latent_fit(train_feats):
    """Fit transformasi kuantil + bobot PC1 per kelompok proksi, HANYA dari fitur train (tanpa target)."""
    fit = {}
    for name, cols in [("behavior", BEHAVIOR), ("credit", CREDIT), ("all", BEHAVIOR + CREDIT)]:
        ref = {c: train_feats[c].cast(pl.Float64).to_numpy() for c in cols}
        Z = np.column_stack([_normal_scores(ref[c], ref[c]) for c in cols])
        # korelasi berpasangan (abaikan null), lalu vektor eigen terbesar = bobot PC1
        k = len(cols); C = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                m = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
                C[i, j] = C[j, i] = np.corrcoef(Z[m, i], Z[m, j])[0, 1] if m.sum() > 100 else 0.0
        vals, vecs = np.linalg.eigh(C)
        w = vecs[:, -1]
        anchor = ANCHOR.get(name, "ec_price_premium_mean")
        if w[cols.index(anchor)] < 0:
            w = -w
        fit[name] = dict(cols=cols, ref=ref, w=w, explained=float(vals[-1] / vals.sum()))
    return fit


def latent_features(feats, fit):
    """Skor laten = rata-rata tertimbang skor normal proksi yang TERSEDIA (bobot = muatan PC1)."""
    out = {"application_id": feats["application_id"]}
    for name, f in fit.items():
        Z = np.column_stack([_normal_scores(feats[c].cast(pl.Float64).to_numpy(), f["ref"][c]) for c in f["cols"]])
        avail = ~np.isnan(Z)
        num = np.nansum(Z * f["w"], axis=1)
        den = (avail * np.abs(f["w"])).sum(axis=1)
        out[f"lt_{name}"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        out[f"lt_{name}_coverage"] = den / np.abs(f["w"]).sum()
    df = pl.DataFrame(out)
    return df.with_columns(lt_behavior_x_credit=pl.col("lt_behavior") * pl.col("lt_credit"))


def loadings_table(fit):
    rows = []
    for name, f in fit.items():
        for c, w in zip(f["cols"], f["w"]):
            rows.append(dict(faktor=name, proksi=c, muatan=float(w)))
    return pl.DataFrame(rows)
