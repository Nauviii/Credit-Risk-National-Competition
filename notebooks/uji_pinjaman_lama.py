"""Uji data latih tambahan dari pinjaman lama (multi-vintage). Jalankan dari folder notebooks/:
    python uji_pinjaman_lama.py
Waktu: ± 2–5 menit membangun fitur (e-commerce penuh lebih lama) + ± 10–20 menit LightGBM (3 seed × 2 varian).

1. Bangun fitur v1 pada cutoff lama untuk setiap pinjaman lama berlabel → data/features/hist_v1.parquet
2. Uji transfer: model yang dilatih HANYA di pinjaman lama → KS pada target train
3. Augmentasi: tiap fold CV = baris latih fold + SEMUA pinjaman lama (+ indikator is_hist), dibandingkan
   berpasangan dengan lgbm_tuned (fold & seed sama). Dua bobot: 1,0 dan 0,3.
"""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

import eda_utils as E
import features as F
import features_hist as FH
import modeling as M

warnings.simplefilter("ignore")
SEEDS = (42, 7, 2024)
WEIGHTS = (1.0, 0.3)

# ------------------------------------------------------------------ 1. fitur pinjaman lama
t = time.time()
if (M.FEAT_DIR / "hist_v1.parquet").exists():
    H = pl.read_parquet(M.FEAT_DIR / "hist_v1.parquet")
    print("hist_v1.parquet dimuat dari disk")
else:
    T = F.load_tables(E.load, E.lazy)
    apps = pl.concat([E.load("train_applications").select("customer_id", "decision_datetime"),
                      E.load("test_applications").select("customer_id", "decision_datetime")])
    HL = FH.hist_loans(T["previous_applications"], T["repayment_history"], apps)
    print(f"pinjaman lama berlabel: {HL.height:,} | bad rate {HL['target'].mean():.4f}")
    H = FH.build_hist_features(HL, T)
    H.write_parquet(M.FEAT_DIR / "hist_v1.parquet")
    del T
print(f"fitur pinjaman lama {H.shape} | {time.time() - t:.0f}s")

meta = json.loads((M.FEAT_DIR / "features_v1.json").read_text())
FEAT, CATS = meta["features"], meta["categorical"]
tr = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
te = pl.read_parquet(M.FEAT_DIR / "test_v1.parquet")
y, yh, seg = tr["target"].to_numpy(), H["target"].to_numpy(), tr["segment"].to_numpy()
levels = {c: sorted(set(tr[c].drop_nulls().to_list()) | set(te[c].drop_nulls().to_list()) | set(H[c].drop_nulls().to_list()))
          for c in CATS}
X, XH = M._to_pandas(tr, FEAT, levels), M._to_pandas(H, FEAT, levels)
L = M.run_from_disk("lgbm_tuned", y)
P = {**M.LGB_PARAMS, **{k: v for k, v in L["params"].items() if k != "random_state"}}

cov = lambda d, n: d.select(pl.lit(n).alias("data"), pl.len().alias("n"), pl.col("target").mean().alias("bad_rate"),
                            pl.col("ac_n_months").is_not_null().mean().alias("ada_rekening"),
                            (pl.col("ec_n_orders_all") > 0).mean().alias("ada_ecommerce"),
                            pl.col("pa_n").is_not_null().mean().alias("ada_riwayat_aplikasi"),
                            pl.col("rp_n_inst").is_not_null().mean().alias("ada_cicilan"))
print(pl.concat([cov(tr, "train sekarang"), cov(H, "pinjaman lama")]).with_columns(pl.col(pl.Float64).round(3)), "\n")


def fit(Xa, ya, wa, Xv, yv, seed):
    m = lgb.LGBMClassifier(**{**P, "random_state": seed})
    m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xv, yv)], eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


# ------------------------------------------------------------------ 2. transfer
idx = np.random.default_rng(0).permutation(len(yh))
cut = int(0.85 * len(yh))
m = fit(XH.iloc[idx[:cut]], yh[idx[:cut]], None, XH.iloc[idx[cut:]], yh[idx[cut:]], 42)
p = m.predict_proba(X)[:, 1]
print(f"2. TRANSFER — model pinjaman lama → target train: KS {M.ks_score(y, p):.4f} "
      f"(lgbm_tuned OOF seed 42: {L['ks_by_seed'][42]:.4f}); KS internal holdout pinjaman lama "
      f"{M.ks_score(yh[idx[cut:]], m.predict_proba(XH.iloc[idx[cut:]])[:, 1]):.4f}")
for g in np.unique(seg):
    mm = seg == g
    print(f"   {g:24s} transfer {M.ks_score(y[mm], p[mm]):.4f} | lgbm_tuned {M.ks_score(y[mm], L['oof_by_seed'][42][mm]):.4f}")
print()

# ------------------------------------------------------------------ 3. augmentasi
Xi, XHi = X.assign(is_hist=0.0), XH.assign(is_hist=1.0)
for w in WEIGHTS:
    obs, t = {}, time.time()
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            Xa = pd.concat([Xi.iloc[tr_i], XHi])
            ya = np.r_[y[tr_i], yh]
            wa = np.r_[np.ones(len(tr_i)), np.full(len(yh), w)]
            oof[va_i] = fit(Xa, ya, wa, Xi.iloc[va_i], y[va_i], s).predict_proba(Xi.iloc[va_i])[:, 1]
        obs[s] = oof
        print(f"   bobot {w}: seed {s} selesai ({time.time() - t:.0f}s)", flush=True)
    r = dict(oof_by_seed=obs, seeds=list(SEEDS), ks_by_seed={s: M.ks_score(y, o) for s, o in obs.items()})
    c = M.compare_runs(L, r, y, seg)
    print(f"3. AUGMENTASI bobot {w}: Δ OOF gabungan {c['delta_oof']:+.4f} | per seed "
          f"{[round(d, 4) for d in c['per_seed']['delta']]} | rata-rata {c['delta_seed_mean']:+.4f} "
          f"| keputusan {M.verdict(c, noise=0.0015)}")
    print(c["per_segment"].with_columns(pl.col(pl.Float64).round(4)), "\n")
