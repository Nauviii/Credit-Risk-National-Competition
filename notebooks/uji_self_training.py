"""Uji keluarga self-training / post-training pada LightGBM (parameter lgbm_tuned, fold & seed identik).
Jalankan dari folder notebooks/:
    python uji_self_training.py              (± 25–40 menit; 12 model per fold × 5 fold × 3 seed)

Semua varian JUJUR terhadap fold validasi luar: label pseudo / target lunak dibuat oleh "guru" yang dilatih hanya di
fold latih luar (CV dalam 4 fold), jadi label fold validasi tidak pernah bocor ke murid.

A. Kurva belajar      : fold latih disubsampel 25/50/75/100% → seberapa besar manfaat data berlabel tambahan?
                        Ini batas atas semua bentuk pseudo-labeling (label pseudo selalu lebih lemah dari label asli).
B. Refit penuh        : "rata-rata 4 model fold dalam" vs "1 model dilatih ulang di seluruh fold latih" dengan jumlah
                        pohon = rata-rata best_iter dalam × {1,00; 1,15; 1,33}. Simulasi pilihan cara membuat prediksi test.
C. Distilasi-diri     : target lunak = 0,5·y + 0,5·p_guru (OOF dalam) pada baris latih.
D. Pseudo-label lunak : baris latih (y) + SEMUA baris test berlabel p_guru (objective cross-entropy).
E. Pseudo-label keras : baris latih + 40% baris test paling aman berlabel 0.
Pembanding C/D/E: model y-saja dengan protokol identik (early stopping AUC di fold validasi, seperti lgbm_tuned).
"""
import json
import pickle
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold

import modeling as M

warnings.simplefilter("ignore")
SEEDS = (42, 7, 2024)
FRACS = (0.25, 0.5, 0.75)
REFIT_MULT = (1.0, 1.15, 1.33)
ALPHA = 0.5

meta = json.loads((M.FEAT_DIR / "features_v1.json").read_text())
FEAT, CATS = meta["features"], meta["categorical"]
tr = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
te = pl.read_parquet(M.FEAT_DIR / "test_v1.parquet")
y, seg = tr["target"].to_numpy(), tr["segment"].to_numpy()
levels = {c: sorted(set(tr[c].drop_nulls().to_list()) | set(te[c].drop_nulls().to_list())) for c in CATS}
X, Xt = M._to_pandas(tr, FEAT, levels), M._to_pandas(te, FEAT, levels)
L = M.run_from_disk("lgbm_tuned", y)
P = {**M.LGB_PARAMS, **{k: v for k, v in L["params"].items() if k not in ("random_state", "objective")},
     "objective": "cross_entropy"}      # cross-entropy = logloss biner untuk label 0/1, dan menerima label lunak


def fit(Xa, ta, seed, Xv=None, yv=None, n_estimators=None, w=None):
    p = {**P, "random_state": seed}
    if n_estimators:
        p["n_estimators"] = int(n_estimators)
    m = lgb.LGBMRegressor(**p)
    if Xv is not None:
        m.fit(Xa, ta, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(Xa, ta, sample_weight=w)
    return m


def logloss(yv, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))


NAMES = (["y-saja"] + [f"kurva {int(f * 100)}%" for f in FRACS]
         + ["rata-rata 4 model dalam"] + [f"refit ×{m:.2f}" for m in REFIT_MULT] + ["refit ×1.15 + rata-rata dalam"]
         + ["distilasi-diri", "pseudo lunak", "pseudo keras (negatif)"])
oof = {n: {s: np.zeros(len(y)) for s in SEEDS} for n in NAMES}
iters = {n: [] for n in NAMES}
t0 = time.time()
for s in SEEDS:
    for k, (tr_i, va_i) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)):
        Xa, ya, Xv, yv = X.iloc[tr_i], y[tr_i], X.iloc[va_i], y[va_i]
        # baseline dengan protokol identik
        m = fit(Xa, ya, s, Xv, yv); oof["y-saja"][s][va_i] = m.predict(Xv); iters["y-saja"].append(m.best_iteration_)
        # A. kurva belajar (subsampel bersarang dari fold latih)
        perm = np.random.default_rng(s * 100 + k).permutation(len(tr_i))
        for f in FRACS:
            sub = perm[: int(f * len(tr_i))]
            m = fit(Xa.iloc[sub], ya[sub], s, Xv, yv)
            oof[f"kurva {int(f * 100)}%"][s][va_i] = m.predict(Xv); iters[f"kurva {int(f * 100)}%"].append(m.best_iteration_)
        # guru: CV dalam 4 fold di fold latih (fold validasi luar tidak disentuh)
        p_in, p_va, p_te, best = np.zeros(len(tr_i)), np.zeros(len(va_i)), np.zeros(len(Xt)), []
        for a_i, b_i in StratifiedKFold(4, shuffle=True, random_state=s).split(Xa, ya):
            g = fit(Xa.iloc[a_i], ya[a_i], s, Xa.iloc[b_i], ya[b_i])
            p_in[b_i] = g.predict(Xa.iloc[b_i]); p_va += g.predict(Xv) / 4; p_te += g.predict(Xt) / 4
            best.append(g.best_iteration_)
        oof["rata-rata 4 model dalam"][s][va_i] = p_va
        # B. refit penuh tanpa early stopping (fold validasi luar tidak dipakai sama sekali)
        nb = np.mean(best)
        m = fit(Xa, ya, s, n_estimators=round(nb * max(REFIT_MULT)))
        for mult in REFIT_MULT:
            oof[f"refit ×{mult:.2f}"][s][va_i] = m.predict(Xv, num_iteration=round(nb * mult))
        oof["refit ×1.15 + rata-rata dalam"][s][va_i] = (oof["refit ×1.15"][s][va_i] + p_va) / 2
        # C. distilasi-diri
        m = fit(Xa, ALPHA * ya + (1 - ALPHA) * p_in, s, Xv, yv)
        oof["distilasi-diri"][s][va_i] = m.predict(Xv); iters["distilasi-diri"].append(m.best_iteration_)
        # D. pseudo-label lunak: semua baris test, label = p_guru
        m = fit(pd.concat([Xa, Xt]), np.r_[ya, p_te], s, Xv, yv)
        oof["pseudo lunak"][s][va_i] = m.predict(Xv); iters["pseudo lunak"].append(m.best_iteration_)
        # E. pseudo-label keras: 40% test paling aman = 0
        neg = p_te < np.quantile(p_te, 0.4)
        m = fit(pd.concat([Xa, Xt[neg]]), np.r_[ya, np.zeros(neg.sum())], s, Xv, yv)
        oof["pseudo keras (negatif)"][s][va_i] = m.predict(Xv); iters["pseudo keras (negatif)"].append(m.best_iteration_)
        print(f"seed {s} fold {k + 1}/5 selesai | {time.time() - t0:.0f}s", flush=True)

runs = {n: dict(oof_by_seed=oof[n], seeds=list(SEEDS), ks_by_seed={s: M.ks_score(y, oof[n][s]) for s in SEEDS}) for n in NAMES}
pickle.dump(dict(oof=oof, iters=iters), open(M.PRED_DIR / "self_training_res.pkl", "wb"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(f"\nlgbm_tuned dari disk (sanity): KS per seed {np.mean(list(L['ks_by_seed'].values())):.4f} | "
      f"y-saja di sini: {np.mean(list(runs['y-saja']['ks_by_seed'].values())):.4f}")


def row(name, base):
    c = M.compare_runs(runs[base], runs[name], y, seg)
    return dict(varian=name, pembanding=base, ks_per_seed=np.mean(list(runs[name]["ks_by_seed"].values())),
                logloss=np.mean([logloss(y, oof[name][s]) for s in SEEDS]),
                d_per_seed=str([round(d, 4) for d in c["per_seed"]["delta"]]), d_rata2=c["delta_seed_mean"],
                d_oof=c["delta_oof"], keputusan=M.verdict(c, noise=0.0015))


print("\nA. KURVA BELAJAR (fraksi fold latih; 100% = y-saja)")
print(pl.DataFrame([row(f"kurva {int(f * 100)}%", "y-saja") for f in FRACS] + [row("y-saja", "y-saja")])
        .select("varian", "ks_per_seed", "logloss", "d_rata2", "d_oof").with_columns(pl.col(pl.Float64).round(4)))
print("\nB. REFIT PENUH vs RATA-RATA MODEL FOLD (keduanya tidak menyentuh fold validasi luar)")
print(pl.DataFrame([row(n, "rata-rata 4 model dalam") for n in NAMES if n.startswith("refit")])
        .drop("pembanding").with_columns(pl.col(pl.Float64).round(4)))
print("\nC–E. SELF-TRAINING vs y-saja")
print(pl.DataFrame([row(n, "y-saja") for n in ("distilasi-diri", "pseudo lunak", "pseudo keras (negatif)")])
        .drop("pembanding").with_columns(pl.col(pl.Float64).round(4)))
print("\nrata-rata best_iter:", {n: round(float(np.mean(v))) for n, v in iters.items() if v})
