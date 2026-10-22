"""Utilitas pelatihan, evaluasi, dan submission (dipakai notebook 04 dan seterusnya).

- ks_score            : metrik KS persis seperti kode evaluasi kompetisi (scipy.ks_2samp)
- train_lgb_cv        : LightGBM StratifiedKFold × beberapa seed → OOF, prediksi test, tabel fold, importance
- save_run / load_run : simpan & muat OOF + prediksi test (untuk ensemble nanti)
- make_submission     : tulis CSV + validasi format sesuai aturan kompetisi
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
FEAT_DIR = ROOT / "data" / "features"
PRED_DIR = ROOT / "data" / "preds"
SUB_DIR = ROOT / "submissions"

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    n_estimators=5000, verbose=-1,
)


def ks_score(y, p):
    """KS dua sampel bad vs good — identik dengan fungsi `score` kompetisi."""
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def _to_pandas(df, features, cat_levels):
    X = df.select(features).to_pandas()
    for c, levels in cat_levels.items():
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


def fold_table(train, seeds=(42, 7, 2024, 11, 99), folds=5, y_col="target"):
    """Nomor fold (1..folds) per baris untuk tiap seed — IDENTIK dengan pembagian di train_lgb_cv / train_cv.
    Dipakai untuk membawa fold yang sama ke environment lain (mis. Kaggle) tanpa bergantung versi scikit-learn."""
    y = train[y_col].to_numpy()
    out = {"application_id": train["application_id"]}
    for seed in seeds:
        f = np.zeros(len(y), dtype=np.int8)
        for k, (_, va_i) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            f[va_i] = k + 1
        out[f"fold_s{seed}"] = f
    return pl.DataFrame(out)


def train_lgb_cv(train, test, features, cat_features=(), y_col="target", seeds=(42, 7, 2024), folds=5,
                 params=None, early_stopping=100, verbose=True):
    """Latih LightGBM dengan CV berlapis seed.
    - OOF per seed: setiap baris diprediksi oleh model yang TIDAK melihatnya.
    - Prediksi test: rata-rata semua model (seeds × folds).
    - early_stopping=None → jumlah pohon tetap (params["n_estimators"]), OOF benar-benar tidak tersentuh.
    Return dict: oof (rata-rata seed), oof_by_seed, test_pred, folds (tabel), importance, best_iters."""
    import lightgbm as lgb

    p = {**LGB_PARAMS, **(params or {})}
    cats = [c for c in cat_features if c in features]
    cat_levels = {c: sorted(set(train[c].drop_nulls().to_list()) | set(test[c].drop_nulls().to_list())) for c in cats}
    X, Xt = _to_pandas(train, features, cat_levels), _to_pandas(test, features, cat_levels)
    y = train[y_col].to_numpy()
    oof_by_seed, test_pred = {}, np.zeros(len(Xt))
    fold_rows, gain, best_iters = [], np.zeros(len(features)), []
    t0 = time.time()
    for seed in seeds:
        oof = np.zeros(len(X))
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for k, (tr_i, va_i) in enumerate(skf.split(X, y)):
            m = lgb.LGBMClassifier(**{**p, "random_state": seed})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if early_stopping:   # berhenti saat AUC fold validasi tidak membaik `early_stopping` ronde
                    m.fit(X.iloc[tr_i], y[tr_i], eval_set=[(X.iloc[va_i], y[va_i])], eval_metric="auc",
                          callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
                    n_iter = m.best_iteration_
                else:                # jumlah pohon tetap = params["n_estimators"]; fold validasi tidak disentuh
                    m.fit(X.iloc[tr_i], y[tr_i])
                    n_iter = p["n_estimators"]
            oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
            test_pred += m.predict_proba(Xt)[:, 1] / (len(seeds) * folds)
            gain += m.booster_.feature_importance("gain")
            best_iters.append(n_iter)
            fold_rows.append(dict(seed=seed, fold=k + 1, ks=ks_score(y[va_i], oof[va_i]), best_iter=n_iter))
        oof_by_seed[seed] = oof
        if verbose:
            print(f"seed {seed}: KS OOF {ks_score(y, oof):.4f}  ({time.time() - t0:.0f}s)")
    oof_mean = np.mean(list(oof_by_seed.values()), axis=0)
    imp = (pl.DataFrame({"feature": features, "gain_pct": gain / gain.sum() * 100})
             .sort("gain_pct", descending=True))
    res = dict(oof=oof_mean, oof_by_seed=oof_by_seed, test_pred=test_pred, folds=pl.DataFrame(fold_rows),
               importance=imp, best_iters=best_iters, features=list(features), params=p, seeds=list(seeds))
    res["ks_oof"] = ks_score(y, oof_mean)
    res["ks_by_seed"] = {s: ks_score(y, o) for s, o in oof_by_seed.items()}
    if verbose:
        print(f"KS OOF rata-rata {len(seeds)} seed: {res['ks_oof']:.4f} | per seed: "
              + ", ".join(f"{v:.4f}" for v in res["ks_by_seed"].values()))
    return res


# =====================================================================================
# Model lain untuk ensemble — fold & seed IDENTIK dengan train_lgb_cv (StratifiedKFold, shuffle, random_state=seed)
# =====================================================================================
XGB_PARAMS = dict(n_estimators=5000, learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
                  colsample_bytree=0.4, reg_lambda=10, tree_method="hist", eval_metric="auc",
                  early_stopping_rounds=100, enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, verbosity=0)
CAT_PARAMS = dict(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=10, rsm=0.5, eval_metric="AUC",
                  od_type="Iter", od_wait=100, verbose=0, thread_count=-1, allow_writing_files=False)
LOGREG_C = 0.05


def _fit_xgb(Xtr, ytr, Xva, yva, Xte, seed, params, cats):
    import xgboost as xgb
    m = xgb.XGBClassifier(**{**XGB_PARAMS, **(params or {}), "random_state": seed})
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    gain = m.get_booster().get_score(importance_type="total_gain")
    imp = np.array([gain.get(c, 0.0) for c in Xtr.columns])
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.best_iteration) + 1, imp


def _cat_str(col):
    """Kategori → string, null → "NA" (CatBoost & one-hot tidak menerima NaN di kolom kategorikal)."""
    return col.astype(object).where(col.notna(), "NA").astype(str)


def _cat_frame(X, cats):
    X = X.copy()
    for c in cats:
        X[c] = _cat_str(X[c])
    return X


def _fit_cat(Xtr, ytr, Xva, yva, Xte, seed, params, cats):
    from catboost import CatBoostClassifier
    Xtr, Xva, Xte = _cat_frame(Xtr, cats), _cat_frame(Xva, cats), _cat_frame(Xte, cats)
    m = CatBoostClassifier(**{**CAT_PARAMS, **(params or {}), "random_seed": seed})
    m.fit(Xtr, ytr, cat_features=list(cats), eval_set=(Xva, yva), use_best_model=True)
    imp = np.asarray(m.get_feature_importance(type="PredictionValuesChange"), dtype=float)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], int(m.get_best_iteration()) + 1, imp


def _fit_logreg(Xtr, ytr, Xva, yva, Xte, seed, params, cats):
    """Regresi logistik sebagai anggota 'berbeda jenis': fitur numerik → kuantil normal (tahan outlier),
    null → 0 + indikator null, kategorikal → one-hot. Semua transformasi di-fit HANYA di fold latih."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
    num = [c for c in Xtr.columns if c not in cats]
    pre = ColumnTransformer([
        ("num", make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                              SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=50)), list(cats)),
    ])
    C = (params or {}).get("C", LOGREG_C)
    m = make_pipeline(pre, LogisticRegression(C=C, max_iter=3000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(_cat_frame(Xtr, cats), ytr)
    return m.predict_proba(_cat_frame(Xva, cats))[:, 1], m.predict_proba(_cat_frame(Xte, cats))[:, 1], 0, None


FITTERS = {"xgb": _fit_xgb, "cat": _fit_cat, "logreg": _fit_logreg}


def train_cv(kind, train, test, features, cat_features=(), y_col="target", seeds=(42, 7, 2024), folds=5,
             params=None, verbose=True):
    """Sama seperti train_lgb_cv tetapi untuk model lain: kind ∈ {'xgb', 'cat', 'logreg'}.
    Fold per seed identik dengan train_lgb_cv → OOF bisa dibandingkan & digabung berpasangan."""
    fit = FITTERS[kind]
    cats = [c for c in cat_features if c in features]
    cat_levels = {c: sorted(set(train[c].drop_nulls().to_list()) | set(test[c].drop_nulls().to_list())) for c in cats}
    X, Xt = _to_pandas(train, features, cat_levels), _to_pandas(test, features, cat_levels)
    y = train[y_col].to_numpy()
    oof_by_seed, test_pred, fold_rows, gain, best_iters = {}, np.zeros(len(Xt)), [], np.zeros(len(features)), []
    t0 = time.time()
    for seed in seeds:
        oof = np.zeros(len(X))
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for k, (tr_i, va_i) in enumerate(skf.split(X, y)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_va, p_te, n_iter, imp = fit(X.iloc[tr_i], y[tr_i], X.iloc[va_i], y[va_i], Xt, seed, params, cats)
            oof[va_i] = p_va
            test_pred += p_te / (len(seeds) * folds)
            if imp is not None:
                gain += imp
            best_iters.append(n_iter)
            fold_rows.append(dict(seed=seed, fold=k + 1, ks=ks_score(y[va_i], p_va), best_iter=n_iter))
        oof_by_seed[seed] = oof
        if verbose:
            print(f"[{kind}] seed {seed}: KS OOF {ks_score(y, oof):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    imp_df = (pl.DataFrame({"feature": features, "gain_pct": gain / gain.sum() * 100}).sort("gain_pct", descending=True)
              if gain.sum() > 0 else None)
    res = dict(oof=np.mean(list(oof_by_seed.values()), axis=0), oof_by_seed=oof_by_seed, test_pred=test_pred,
               folds=pl.DataFrame(fold_rows), importance=imp_df, best_iters=best_iters, features=list(features),
               params={"kind": kind, **(params or {})}, seeds=list(seeds))
    res["ks_oof"] = ks_score(y, res["oof"])
    res["ks_by_seed"] = {s: ks_score(y, o) for s, o in oof_by_seed.items()}
    if verbose:
        print(f"[{kind}] KS OOF rata-rata {len(seeds)} seed: {res['ks_oof']:.4f}")
    return res


def run_from_disk(name, y):
    """Muat run tersimpan (save_run) sebagai dict yang kompatibel dengan compare_runs / blend."""
    oof, te, meta = load_run(name)
    seed_cols = [c for c in oof.columns if c.startswith("oof_seed")]
    seeds = [int(c.replace("oof_seed", "")) for c in seed_cols]
    obs = {s: oof[f"oof_seed{s}"].to_numpy() for s in seeds}
    res = dict(oof=oof["oof"].to_numpy(), oof_by_seed=obs, test_pred=te["pred"].to_numpy(), seeds=seeds,
               features=meta.get("features", []), params=meta.get("params", {}), meta=meta)
    res["ks_oof"] = ks_score(y, res["oof"])
    res["ks_by_seed"] = {s: ks_score(y, o) for s, o in obs.items()}
    return res


def rank01(x):
    """Peringkat dinormalisasi ke (0, 1]. KS hanya peduli urutan, jadi menggabung peringkat aman & adil antar model."""
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


def blend(runs, y, weights=None, seeds=None):
    """Rata-rata peringkat tertimbang dari beberapa run, per seed (OOF) dan untuk prediksi test.
    runs: dict nama → run. Return dict kompatibel compare_runs (oof_by_seed, ks_by_seed, oof, test_pred, seeds)."""
    names = list(runs)
    w = np.array([1.0] * len(names) if weights is None else [weights.get(n, 0.0) for n in names], dtype=float)
    w = w / w.sum()
    seeds = seeds or [s for s in runs[names[0]]["seeds"] if all(s in runs[n]["seeds"] for n in names)]
    obs = {s: sum(wi * rank01(runs[n]["oof_by_seed"][s]) for wi, n in zip(w, names) if wi > 0) for s in seeds}
    res = dict(oof_by_seed=obs, seeds=list(seeds), oof=np.mean(list(obs.values()), axis=0),
               test_pred=sum(wi * rank01(runs[n]["test_pred"]) for wi, n in zip(w, names) if wi > 0),
               weights=dict(zip(names, w.round(4).tolist())))
    res["ks_oof"] = ks_score(y, res["oof"])
    res["ks_by_seed"] = {s: ks_score(y, o) for s, o in obs.items()}
    return res


def _weight_grid(names, base_name, step=0.1, max_other=0.7):
    """Semua kombinasi bobot di grid `step`; anggota selain base_name total ≤ max_other, sisanya untuk base_name."""
    import itertools
    others = [n for n in names if n != base_name]
    ticks = np.round(np.arange(0, max_other + 1e-9, step), 4)
    out = []
    for w in itertools.product(ticks, repeat=len(others)):
        if sum(w) <= max_other + 1e-9:
            out.append({base_name: round(1 - sum(w), 4), **dict(zip(others, w))})
    return out


def best_weights(runs, y, base_name, seeds, rows=None, step=0.1, max_other=0.7):
    """Bobot rata-rata peringkat yang memaksimalkan KS rata-rata seed (opsional hanya pada `rows`)."""
    names = list(runs)
    R = {s: np.vstack([rank01(runs[n]["oof_by_seed"][s]) for n in names]) for s in seeds}
    rows = np.arange(len(y)) if rows is None else rows
    best, best_ks = None, -1
    for w in _weight_grid(names, base_name, step, max_other):
        wv = np.array([w[n] for n in names])
        ks = np.mean([ks_score(y[rows], wv @ R[s][:, rows]) for s in seeds])
        if ks > best_ks:
            best, best_ks = w, ks
    return best, best_ks


def nested_blend(runs, y, base_name, seeds, folds=5, step=0.1, max_other=0.7):
    """Estimasi JUJUR manfaat bobot ensemble: bobot dipilih di 4/5 baris, diterapkan ke 1/5 sisanya (bergiliran).
    Memilih bobot di data yang sama dengan yang dinilai selalu optimistis; cara ini menghilangkan optimisme itu.
    Return dict kompatibel compare_runs + daftar bobot yang terpilih di tiap fold."""
    names = list(runs)
    obs, chosen = {}, []
    for s in seeds:
        R = np.vstack([rank01(runs[n]["oof_by_seed"][s]) for n in names])
        out = np.zeros(len(y))
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=s + 1000).split(np.zeros(len(y)), y):
            w, _ = best_weights(runs, y, base_name, [s], rows=tr, step=step, max_other=max_other)
            out[va] = np.array([w[n] for n in names]) @ R[:, va]
            chosen.append(dict(seed=s, **w))
        obs[s] = out
    res = dict(oof_by_seed=obs, seeds=list(seeds), oof=np.mean(list(obs.values()), axis=0),
               fold_weights=pl.DataFrame(chosen))
    res["ks_oof"] = ks_score(y, res["oof"])
    res["ks_by_seed"] = {s: ks_score(y, o) for s, o in obs.items()}
    return res


def segment_ks(y, p, segment):
    """KS keseluruhan + per segmen (n, default rate, KS)."""
    y, p, segment = np.asarray(y), np.asarray(p), np.asarray(segment)
    rows = [dict(segment="SEMUA", n=len(y), default_rate=y.mean(), ks=ks_score(y, p))]
    for s in sorted(set(segment)):
        m = segment == s
        rows.append(dict(segment=s, n=int(m.sum()), default_rate=y[m].mean(), ks=ks_score(y[m], p[m])))
    return pl.DataFrame(rows)


def lift_table(y, p, q=10):
    """Tabel desil skor (desil 1 = paling berisiko): default rate, lift, % default tertangkap (kumulatif)."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    dec = np.empty(len(p), int)
    dec[order] = np.arange(len(p)) * q // len(p) + 1
    df = (pl.DataFrame({"desil": dec, "y": y, "p": p})
            .group_by("desil").agg(n=pl.len(), skor_min=pl.col("p").min(), skor_max=pl.col("p").max(),
                                   n_default=pl.col("y").sum(), default_rate=pl.col("y").mean())
            .sort("desil")
            .with_columns(lift=pl.col("default_rate") / y.mean(),
                          pct_default_kumulatif=pl.col("n_default").cum_sum() / y.sum() * 100,
                          pct_populasi_kumulatif=pl.col("n").cum_sum() / len(y) * 100))
    return df


def ks_curve(y, p):
    """Kurva KS: untuk tiap ambang skor, proporsi bad & good dengan skor ≤ ambang (CDF).
    KS = jarak vertikal maksimum antara dua CDF. Return (ambang, cdf_bad, cdf_good, idx_maks)."""
    y, p = np.asarray(y), np.asarray(p)
    o = np.argsort(p, kind="mergesort")
    ps, ys = p[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]
    cb = np.cumsum(ys)[last] / ys.sum()
    cg = np.cumsum(1 - ys)[last] / (1 - ys).sum()
    return ps[last], cb, cg, int(np.argmax(np.abs(cg - cb)))


def compare_runs(base, new, y, segment=None):
    """Bandingkan dua hasil train_lgb_cv dengan seed & fold SAMA (perbandingan berpasangan).
    Return dict: tabel per seed (KS base, KS baru, Δ), Δ KS OOF gabungan, Δ per segmen, dan ringkasan."""
    seeds = [s for s in base["seeds"] if s in new["seeds"]]
    per_seed = pl.DataFrame({"seed": seeds,
                             "ks_base": [base["ks_by_seed"][s] for s in seeds],
                             "ks_baru": [new["ks_by_seed"][s] for s in seeds]}).with_columns(
        delta=pl.col("ks_baru") - pl.col("ks_base"))
    # OOF gabungan dihitung dari seed yang SAMA di kedua run (base boleh punya seed lebih banyak)
    oof_b = np.mean([base["oof_by_seed"][s] for s in seeds], axis=0)
    oof_n = np.mean([new["oof_by_seed"][s] for s in seeds], axis=0)
    out = dict(per_seed=per_seed,
               delta_oof=ks_score(y, oof_n) - ks_score(y, oof_b),
               delta_seed_mean=float(per_seed["delta"].mean()),
               delta_seed_min=float(per_seed["delta"].min()),
               n_seed_positive=int((per_seed["delta"] > 0).sum()))
    if segment is not None:
        sb, sn = segment_ks(y, oof_b, segment), segment_ks(y, oof_n, segment)
        out["per_segment"] = sb.select("segment", "n", ks_base="ks").join(
            sn.select("segment", ks_baru="ks"), on="segment").with_columns(delta=pl.col("ks_baru") - pl.col("ks_base"))
    return out


def verdict(cmp, clear=0.005, thin=0.002, noise=0.0):
    """Keputusan sesuai aturan main:
    - 'jelas' : Δ rata-rata seed ≥ clear dan semua seed positif
    - 'tipis' : Δ rata-rata seed ≥ max(thin, noise) dan semua seed positif
    - 'tidak' : selain itu (termasuk bila ada seed yang turun)"""
    d, all_pos = cmp["delta_seed_mean"], cmp["n_seed_positive"] == cmp["per_seed"].height
    if all_pos and d >= clear:
        return "jelas"
    if all_pos and d >= max(thin, noise):
        return "tipis"
    return "tidak"


def plot_paired_deltas(cmps, names, noise=0.0, title="", labels=None, thresholds=((0.002, "tipis"), (0.005, "jelas"))):
    """Grafik Δ KS per seed (titik) + rata-rata (garis hitam) untuk beberapa eksperimen vs baseline.
    cmps: dict nama → hasil compare_runs. Pita abu-abu = ±noise (mis. dari placebo)."""
    import matplotlib.pyplot as plt
    import eda_utils as E
    labels = labels or names
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(names) + 1.6))
    seen = set()
    for i, n in enumerate(names):
        ps = cmps[n]["per_seed"]
        for j, (s, v) in enumerate(zip(ps["seed"], ps["delta"])):
            ax.scatter(v, i, color=E.SERIES[j % len(E.SERIES)], s=40, zorder=3,
                       label=None if s in seen else f"seed {s}")
            seen.add(s)
        ax.scatter(ps["delta"].mean(), i, marker="|", s=400, color="black", zorder=4)
    if noise:
        ax.axvspan(-noise, noise, color="grey", alpha=0.15)
    ax.axvline(0, color="black", lw=0.8)
    for x, lab in thresholds:
        ax.axvline(x, color=E.SERIES[1], lw=1, ls="--")
        ax.text(x, -0.5, f" {lab} ≥ {x}", color=E.SERIES[1], fontsize=9, va="bottom")
    ax.set_yticks(range(len(names)), labels)
    ax.set_ylim(len(names) - 0.4, -0.7)
    ax.set_xlabel("Δ KS OOF vs pembanding (per seed, berpasangan) — garis hitam = rata-rata seed")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.tight_layout()
    plt.show()


def save_run(name, train, test, res, extra=None):
    """Simpan OOF & prediksi test + metadata, untuk dibandingkan / di-ensemble nanti."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    train.select("application_id").with_columns(
        pl.Series("oof", res["oof"]), *[pl.Series(f"oof_seed{s}", o) for s, o in res["oof_by_seed"].items()]
    ).write_parquet(PRED_DIR / f"{name}_oof.parquet")
    test.select("application_id").with_columns(pl.Series("pred", res["test_pred"])).write_parquet(PRED_DIR / f"{name}_test.parquet")
    meta = dict(name=name, ks_oof=res["ks_oof"], ks_by_seed={str(k): v for k, v in res["ks_by_seed"].items()},
                n_features=len(res["features"]), features=res["features"], params=res["params"], seeds=res["seeds"],
                mean_best_iter=float(np.mean(res["best_iters"])), **(extra or {}))
    (PRED_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return PRED_DIR / f"{name}_meta.json"


def load_run(name):
    oof = pl.read_parquet(PRED_DIR / f"{name}_oof.parquet")
    test = pl.read_parquet(PRED_DIR / f"{name}_test.parquet")
    meta = json.loads((PRED_DIR / f"{name}_meta.json").read_text())
    return oof, test, meta


def make_submission(test_ids, pred, sample_submission, path):
    """Tulis CSV application_id,default_probability dan validasi semua aturan format kompetisi."""
    pred = np.asarray(pred, dtype=float)
    sub = pl.DataFrame({"application_id": test_ids, "default_probability": np.clip(pred, 0.0, 1.0)})
    ss_ids = set(sample_submission["application_id"].to_list())
    checks = {
        "kolom tepat [application_id, default_probability]": sub.columns == ["application_id", "default_probability"],
        "jumlah baris = sample_submission": sub.height == sample_submission.height,
        "application_id unik": sub["application_id"].n_unique() == sub.height,
        "semua ID test ada & tidak ada ID asing": set(sub["application_id"].to_list()) == ss_ids,
        "probabilitas finite": bool(np.isfinite(sub["default_probability"].to_numpy()).all()),
        "probabilitas dalam [0, 1]": bool(sub["default_probability"].is_between(0, 1).all()),
        "prediksi tidak konstan": sub["default_probability"].n_unique() > 1,
    }
    for k, v in checks.items():
        print(("✅ " if v else "❌ ") + k)
    if not all(checks.values()):
        raise ValueError("Submission TIDAK valid — lihat daftar di atas.")
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    sub.write_csv(path)
    print(f"Tersimpan: {path}  ({sub.height:,} baris)")
    return sub