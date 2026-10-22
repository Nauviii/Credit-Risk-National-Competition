"""Uji dua blok fitur hasil investigasi generator data (features_latent.py). Jalankan dari folder notebooks/:
    python uji_blok_laten.py
Waktu: ± 1–3 menit membangun fitur + ± 30–40 menit model (4 varian LightGBM + 1 logreg, masing-masing 3 seed).

- lt_  estimasi laten empirical-Bayes (skor biro gabungan semua laporan, peluang telat, DPD, premi harga, utilisasi,
       inquiry, tunggakan, pendapatan laten & DTI laten)
- gr_  lonjakan order menjelang keputusan (hitungan berbobot eksponensial / baseline sendiri, τ = 1, 3, 7, 14, 30 hari)
Pembanding: lgbm_tuned dan logreg_v1 (fold & seed sama); plasebo = blok lt_+gr_ dengan baris diacak.
Fitur disimpan ke data/features/{train,test}_lt.parquet agar bisa dipakai ulang."""
import json
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import eda_utils as E
import features as F
import features_latent as LT
import modeling as M

SEEDS = (42, 7, 2024)
meta = json.loads((M.FEAT_DIR / "features_v1.json").read_text())
FEAT, CATS = meta["features"], meta["categorical"]
tr = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
te = pl.read_parquet(M.FEAT_DIR / "test_v1.parquet")
y, seg = tr["target"].to_numpy(), tr["segment"].to_numpy()

t = time.time()
if (M.FEAT_DIR / "train_lt.parquet").exists() and "gr_ramp1" in pl.read_parquet_schema(M.FEAT_DIR / "train_lt.parquet"):
    ltr, lte = pl.read_parquet(M.FEAT_DIR / "train_lt.parquet"), pl.read_parquet(M.FEAT_DIR / "test_lt.parquet")
    print("fitur lt_/gr_ dimuat dari disk")
else:
    T = F.load_tables(E.load, E.lazy)
    tr_app, te_app = E.load("train_applications"), E.load("test_applications")
    mtr = LT.measurements(tr_app, T)
    P = LT.fit_params(tr_app, T, mtr)          # hiperparameter dari data train, tanpa target
    print("hiperparameter (mean, var laten, var noise):")
    for k, v in P.items():
        print(f"   {k:7s}", {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else np.round(v, 4))
    ltr = LT.latent_features(tr_app, T, P, mtr).join(LT.ramp_features(tr_app, T), on="application_id")
    lte = LT.latent_features(te_app, T, P).join(LT.ramp_features(te_app, T), on="application_id")
    ltr.write_parquet(M.FEAT_DIR / "train_lt.parquet")
    lte.write_parquet(M.FEAT_DIR / "test_lt.parquet")
    del T
print(f"fitur {ltr.shape} | {time.time() - t:.0f}s")
tr2 = LT.add_cross(tr.join(ltr, on="application_id", how="left"))
te2 = LT.add_cross(te.join(lte, on="application_id", how="left"))
assert tr2["application_id"].equals(tr["application_id"]) and te2["application_id"].equals(te["application_id"])
LTC = LT.latent_columns(tr2)
GR = [c for c in tr2.columns if c.startswith("gr_")]

# ------------------------------------------------------------------ AUC univariat: mentah vs estimasi laten
PAIRS = [("rp_late_share", "lt_late_post"), ("rp_mean_dpd", "lt_dpd_post"), ("ec_price_premium_mean", "lt_prem_post"),
         ("bu_all_score_mean", "lt_bureau_post"), ("bu_all_util_mean", "lt_util_post"), ("bu_all_inq_mean", "lt_inq_post"),
         ("bu_all_delinq_mean", "lt_delinq_post"), ("total_dti", "lt_total_dti_hat"),
         ("ec_order_surge_30d", "gr_ramp7"), ("ec_order_surge_30d", "gr_ramp14")]


def auc(c):
    x = tr2[c].cast(pl.Float64).to_numpy(); m = ~np.isnan(x)
    a = roc_auc_score(y[m], x[m]) if np.unique(x[m]).size > 1 else 0.5
    return max(a, 1 - a), m.mean()


rows = [dict(mentah=r, auc_mentah=auc(r)[0], baru=n, auc_baru=auc(n)[0], terisi=auc(n)[1]) for r, n in PAIRS]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows).with_columns(selisih=pl.col("auc_baru") - pl.col("auc_mentah")).with_columns(pl.col(pl.Float64).round(4)), "\n")

# ------------------------------------------------------------------ uji berpasangan
L, LR = M.run_from_disk("lgbm_tuned", y), M.run_from_disk("logreg_v1", y)
P = {k: v for k, v in L["params"].items() if k != "random_state"}
perm = np.random.default_rng(0).permutation(len(y))
RUNS = {}
for name, fn, base in (
        ("lgbm + lt", lambda: M.train_lgb_cv(tr2, te2, FEAT + LTC, CATS, seeds=SEEDS, params=P, verbose=False), L),
        ("lgbm + gr", lambda: M.train_lgb_cv(tr2, te2, FEAT + GR, CATS, seeds=SEEDS, params=P, verbose=False), L),
        ("lgbm + lt + gr", lambda: M.train_lgb_cv(tr2, te2, FEAT + LTC + GR, CATS, seeds=SEEDS, params=P, verbose=False), L),
        ("lgbm + plasebo", lambda: M.train_lgb_cv(tr.hstack(tr2.select(LTC + GR)[perm]), te2, FEAT + LTC + GR, CATS,
                                                  seeds=SEEDS, params=P, verbose=False), L),
        ("logreg + lt + gr", lambda: M.train_cv("logreg", tr2, te2, FEAT + LTC + GR, CATS, seeds=SEEDS, verbose=False), LR)):
    t = time.time()
    RUNS[name] = r = fn()
    c = M.compare_runs(base, r, y, seg)
    print(f"{name:17s} Δ OOF gabungan {c['delta_oof']:+.4f} | per seed {[round(d, 4) for d in c['per_seed']['delta']]} "
          f"| rata-rata {c['delta_seed_mean']:+.4f} | {M.verdict(c, noise=0.0015)} | {time.time() - t:.0f}s", flush=True)
    print("   per segmen:", c["per_segment"].select("segment", "delta").with_columns(pl.col("delta").round(4)).to_dicts())

R = {k: M.run_from_disk(v, y) for k, v in {"lgbm": "lgbm_tuned", "xgb": "xgb_v1", "cat": "cat_v1", "logreg": "logreg_v1"}.items()}
REF = M.blend(R, y, seeds=list(SEEDS))
best = max(("lgbm + lt", "lgbm + gr", "lgbm + lt + gr"), key=lambda n: np.mean(list(RUNS[n]["ks_by_seed"].values())))
for n, mem in ((f"ensemble, lgbm → {best}", {**R, "lgbm": RUNS[best]}),
               (f"ensemble, lgbm → {best} & logreg → + lt + gr", {**R, "lgbm": RUNS[best], "logreg": RUNS["logreg + lt + gr"]})):
    c = M.compare_runs(REF, M.blend(mem, y, seeds=list(SEEDS)), y)
    print(f"{n:50s} Δ OOF gabungan {c['delta_oof']:+.4f} | per seed {[round(d, 4) for d in c['per_seed']['delta']]} "
          f"| rata-rata {c['delta_seed_mean']:+.4f} | {M.verdict(c, noise=0.0015)}")
for n in ("lgbm + lt", "lgbm + gr", "lgbm + lt + gr", "logreg + lt + gr"):
    M.save_run(n.replace(" + ", "_").replace(" ", ""), tr2, te2, RUNS[n], extra=dict(block=n))
