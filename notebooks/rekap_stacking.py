"""Uji stacking (meta-model) vs ensemble saat ini (sub_v4), dari run yang tersimpan di data/preds. Tanpa melatih ulang anggota.

Jalankan dari folder notebooks/:  python rekap_stacking.py   (atau tempel ke satu sel notebook)

- Pembanding (REF) = rata-rata peringkat bobot sama lgbm_tuned + xgb_v1 + cat_v1 + logreg_v1, seed 42, 7, 2024.
- Input meta-model = logit dari peringkat-di-dalam-fold tiap anggota (skala sama untuk GBDT & NN).
- Meta-model dinilai JUJUR: dilatih di 4/5 baris, dinilai di 1/5 sisanya, bergiliran (split berbeda dari fold anggota).
- Anggota NN (realmlp_v1, tabm_v1) dipakai kalau ada di disk.
"""
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import modeling as M

warnings.simplefilter("ignore")
SEEDS = [42, 7, 2024]
NOISE, THIN = 0.0015, 0.002

train = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
y = train["target"].to_numpy()
SEG = train["segment"].to_numpy()
FOLDS = M.fold_table(train)


def rank_in_fold(o, f):
    out = np.zeros(len(o))
    for k in np.unique(f):
        m = f == k
        out[m] = (rankdata(o[m]) - 0.5) / m.sum()
    return out


def load(name):
    if not (M.PRED_DIR / f"{name}_meta.json").exists():
        return None
    r = M.run_from_disk(name, y)
    if not all(s in r["seeds"] for s in SEEDS):
        return None
    obs = {s: rank_in_fold(r["oof_by_seed"][s], FOLDS[f"fold_s{s}"].to_numpy()) for s in SEEDS}
    return dict(oof_by_seed=obs, seeds=list(SEEDS), test_pred=r["test_pred"],
                ks_by_seed={s: M.ks_score(y, o) for s, o in obs.items()})


CORE = {"lgbm": "lgbm_tuned", "xgb": "xgb_v1", "cat": "cat_v1", "logreg": "logreg_v1"}
NN = {"realmlp": "realmlp_v1", "tabm": "tabm_v1"}
R = {k: load(v) for k, v in {**CORE, **NN}.items()}
R = {k: v for k, v in R.items() if v is not None}
assert all(k in R for k in CORE), "run inti harus ada"
ALL = list(R)
print("Anggota tersedia:", ALL)

REF = M.blend({k: R[k] for k in CORE}, y, seeds=SEEDS)
print(f"Pembanding sub_v4 (rank per fold): KS OOF gabungan {REF['ks_oof']:.4f}\n")

# 15 fitur numerik dengan korelasi peringkat terbesar terhadap target (untuk varian 'feature-weighted')
import json
meta = json.loads((M.FEAT_DIR / "features_v1.json").read_text())
num = [c for c in meta["features"] if c not in meta["categorical"]]
corr = {c: abs(spearmanr(train[c].cast(pl.Float64).fill_null(-999).to_numpy(), y).statistic) for c in num}
TOP = sorted(corr, key=lambda c: -np.nan_to_num(corr[c]))[:15]
XTOP = train.select(TOP).cast(pl.Float64).fill_null(-999).to_numpy()
SEGD = np.stack([(SEG == g).astype(float) for g in np.unique(SEG)], 1)


def Z(members, s):
    return np.stack([norm.ppf(R[k]["oof_by_seed"][s]) for k in members], 1)


def meta_oof(X, s, kind):
    out, coefs = np.zeros(len(y)), []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s + 777).split(X, y):
        if kind == "lr":
            mdl = LogisticRegression(C=1.0, max_iter=3000).fit(X[a], y[a])
            coefs.append(mdl.coef_[0])
        else:
            mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=4, min_child_samples=500,
                                     subsample=0.8, subsample_freq=1, verbose=-1, random_state=s).fit(X[a], y[a])
        out[b] = mdl.predict_proba(X[b])[:, 1]
    return out, (np.mean(coefs, axis=0) if coefs else None)


def interact(X):
    return np.hstack([X] + [X * SEGD[:, [j]] for j in range(SEGD.shape[1])] + [SEGD])


CANDS = {"bobot sama, semua anggota": ("blend", ALL)}
for label, members in (("4 inti", list(CORE)), ("semua anggota", ALL)):
    CANDS[f"LR meta, {label}"] = ("lr", members)
CANDS["LR meta + interaksi segmen, semua anggota"] = ("lr_seg", ALL)
CANDS["LightGBM meta, semua anggota"] = ("gbm", ALL)
CANDS["LightGBM meta + 15 fitur teratas"] = ("gbm_feat", ALL)

rows, coef_rows = [], []
for name, (kind, members) in CANDS.items():
    if kind == "blend":
        r = M.blend({k: R[k] for k in members}, y, seeds=SEEDS)
    else:
        obs = {}
        for s in SEEDS:
            X = Z(members, s)
            if kind == "lr_seg":
                X = interact(X)
            if kind == "gbm_feat":
                X = np.hstack([X, XTOP])
            obs[s], coef = meta_oof(X, s, "lr" if kind.startswith("lr") else "gbm")
            if kind == "lr" and coef is not None:
                coef_rows.append(dict(meta=name, seed=s, **{k: float(c) for k, c in zip(members, coef)}))
        r = dict(oof_by_seed=obs, seeds=SEEDS, ks_by_seed={s: M.ks_score(y, o) for s, o in obs.items()})
    c = M.compare_runs(REF, r, y)
    rows.append(dict(kandidat=name, delta_oof_gabungan=c["delta_oof"],
                     **{f"d_s{s}": d for s, d in zip(c["per_seed"]["seed"], c["per_seed"]["delta"])},
                     delta_rata2=c["delta_seed_mean"], seed_naik=f"{c['n_seed_positive']}/{len(SEEDS)}",
                     keputusan=M.verdict(c, thin=THIN, noise=NOISE)))
    print(f"  selesai: {name}", flush=True)

pl.Config.set_tbl_rows(30); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(200)
print("\nStacking vs sub_v4")
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4)))
if coef_rows:
    print("\nKoefisien LR meta (rata-rata 5 fold). Bobot yang mirip satu sama lain = kombinasi optimal ≈ bobot sama:")
    print(pl.DataFrame(coef_rows).with_columns(pl.col(pl.Float64).round(3)))
print(f"\nAturan: tipis = Δ rata-rata per seed ≥ {THIN} dan > noise {NOISE}, semua seed naik; jelas = Δ ≥ 0.005.")
