"""Neural network & foundation model tabular untuk ensemble (notebook 10, dijalankan di Kaggle GPU atau lokal).

Semua model memakai fold IDENTIK dengan LightGBM: tabel fold (application_id, fold_s{seed}) dibuat sekali secara lokal
(modeling.fold_table) lalu dibawa ke Kaggle, sehingga OOF semua model bisa dibandingkan & digabung berpasangan.

Model:
- realmlp : RealMLP-TD (pytabkit) — MLP dengan default yang sudah dituning (robust scaling + smooth clip, embedding PBLD)
- tabm    : TabM-D (pytabkit) — satu model meniru ensemble k MLP (BatchEnsemble) + embedding numerik piecewise-linear
- tabicl  : TabICLv2 — foundation model tabular (in-context learning), tanpa training; butuh GPU untuk 56 ribu baris

Persiapan data (dipakai semua model):
- null → median fold latih + kolom indikator null. Kolom yang pola null-nya identik (mis. semua rp_* untuk nasabah
  tanpa cicilan) berbagi SATU indikator, jadi jumlah kolom tambahan kecil.
- kategori → pandas category (RealMLP/TabM) atau kode integer (TabICL).
"""
import time
import warnings

import numpy as np
import pandas as pd
import polars as pl

import modeling as M


# ------------------------------------------------------------------------------------------------ data
def null_groups(train, features):
    """Kelompokkan kolom berdasarkan pola null yang identik (dihitung di train). Return list (nama_indikator, [kolom])."""
    groups = {}
    for c in features:
        m = train[c].is_null().to_numpy()
        if not m.any():
            continue
        key = np.packbits(m).tobytes()
        groups.setdefault(key, []).append(c)
    return [(f"isnull__{cols[0]}", cols) for cols in groups.values()]


def nn_frame(df, features, cats, cat_levels, groups):
    """pandas DataFrame: fitur asli (null dibiarkan dulu) + indikator null per kelompok."""
    X = M._to_pandas(df, features, cat_levels)
    for name, cols in groups:
        X[name] = df[cols[0]].is_null().cast(pl.Int8).to_numpy().astype(np.float32)
    return X


def impute(X_tr, others, cats):
    """Isi null numerik dengan median fold latih (di-fit hanya di X_tr)."""
    num = [c for c in X_tr.columns if c not in cats]
    med = X_tr[num].median()
    out = [X_tr.copy()] + [o.copy() for o in others]
    for o in out:
        o[num] = o[num].fillna(med).astype(np.float32)
    return out


def to_codes(X, cats):
    """Untuk TabICL: kategori → kode integer (null → -1 → dipetakan ke kode tersendiri)."""
    X = X.copy()
    for c in cats:
        X[c] = X[c].cat.codes.astype(np.float32)
    return X


# ------------------------------------------------------------------------------------------------ model
def _make(kind, seed, device, params):
    params = dict(params or {})
    if kind == "realmlp":
        from pytabkit import RealMLP_TD_Classifier
        return RealMLP_TD_Classifier(device=device, random_state=seed, val_metric_name="1-auc_ovr", verbosity=0, **params)
    if kind == "tabm":
        from pytabkit import TabM_D_Classifier
        return TabM_D_Classifier(device=device, random_state=seed, val_metric_name="1-auc_ovr", verbosity=0, **params)
    if kind == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(device=device, random_state=seed, **params)
    raise ValueError(kind)


def _fit_predict(kind, model, Xtr, ytr, Xva, yva, Xte, cats):
    if kind in ("realmlp", "tabm"):
        model.fit(Xtr, ytr, X_val=Xva, y_val=yva, cat_col_names=list(cats))
    else:                                    # tabicl: in-context, tanpa validasi
        Xtr, Xva, Xte = to_codes(Xtr, cats), to_codes(Xva, cats), to_codes(Xte, cats)
        model.fit(Xtr, ytr)
        p = model.predict_proba(pd.concat([Xva, Xte], axis=0))[:, 1]   # satu kali proses konteks untuk val + test
        return p[:len(Xva)], p[len(Xva):]
    return model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]


def train_nn_cv(kind, train, test, features, cat_features, folds, seeds=(42, 7, 2024), params=None, device="cuda",
                y_col="target", max_folds=None, verbose=True):
    """CV dengan fold dari tabel `folds` (kolom fold_s{seed}, nilai 1..5). Return dict format train_lgb_cv.
    max_folds: untuk uji waktu (mis. 1 → hanya fold pertama dari seed pertama)."""
    cats = [c for c in cat_features if c in features]
    levels = {c: sorted(set(train[c].drop_nulls().to_list()) | set(test[c].drop_nulls().to_list())) for c in cats}
    groups = null_groups(train, features)
    X, Xt = nn_frame(train, features, cats, levels, groups), nn_frame(test, features, cats, levels, groups)
    y = train[y_col].to_numpy()
    assert folds["application_id"].equals(train["application_id"]), "urutan tabel fold harus sama dengan train"
    oof_by_seed, test_pred, rows, n_models = {}, np.zeros(len(Xt)), [], 0
    t0 = time.time()
    for seed in seeds:
        f = folds[f"fold_s{seed}"].to_numpy()
        oof = np.full(len(X), np.nan)
        for k in sorted(set(f)):
            if max_folds is not None and n_models >= max_folds:
                break
            tr_i, va_i = np.flatnonzero(f != k), np.flatnonzero(f == k)
            Xtr, Xva, Xte = impute(X.iloc[tr_i], [X.iloc[va_i], Xt], cats)
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _make(kind, seed, device, params)
                p_va, p_te = _fit_predict(kind, model, Xtr, y[tr_i], Xva, y[va_i], Xte, cats)
            oof[va_i] = p_va
            test_pred += p_te
            n_models += 1
            rows.append(dict(seed=seed, fold=int(k), ks=M.ks_score(y[va_i], p_va), detik=round(time.time() - t, 1)))
            if verbose:
                print(f"[{kind}] seed {seed} fold {k}: KS {rows[-1]['ks']:.4f} ({rows[-1]['detik']}s)", flush=True)
            del model
        oof_by_seed[seed] = oof
        if max_folds is not None and n_models >= max_folds:
            break
    test_pred /= max(n_models, 1)
    complete = {s: o for s, o in oof_by_seed.items() if not np.isnan(o).any()}
    res = dict(oof_by_seed=complete, test_pred=test_pred, folds=pl.DataFrame(rows), importance=None,
               best_iters=[0] * n_models, features=list(features) + [g for g, _ in groups],
               params={"kind": kind, **(params or {})}, seeds=list(complete))
    if complete:
        res["oof"] = np.mean(list(complete.values()), axis=0)
        res["ks_oof"] = M.ks_score(y, res["oof"])
        res["ks_by_seed"] = {s: M.ks_score(y, o) for s, o in complete.items()}
        if verbose:
            print(f"[{kind}] KS OOF rata-rata {len(complete)} seed: {res['ks_oof']:.4f} | total {time.time() - t0:.0f}s")
    return res
