"""Uji anggota ensemble tambahan di luar GBDT/NN: SVM linear, SVM RBF (Nystroem), EBM, Random Forest, Extra Trees,
Ridge, LDA, kNN, Naive Bayes. Fold & seed identik dengan anggota lain; setiap run disimpan sebagai data/preds/extra_<model>_*
(bisa dilanjutkan bila terputus). Jalankan dari folder notebooks/:

    python uji_model_tambahan.py                      # model default (lihat MODELS)
    python uji_model_tambahan.py svm_lin,ebm          # pilih model tertentu

Perkiraan waktu per seed (5 fold) di mesin 2 core: svm_lin 2 mnt, ebm 3,5 mnt, svm_rbf 8 mnt, rf 13 mnt, et 12 mnt,
ridge/lda/knn/nb < 1 mnt. EBM butuh `pip install interpret-core`.
"""
import json
import sys
import time
import warnings

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

import modeling as M

warnings.simplefilter("ignore")
SEEDS = (42, 7, 2024)
MODELS = ["svm_lin", "ridge", "ebm", "svm_rbf", "rf"]

meta = json.loads((M.FEAT_DIR / "features_v1.json").read_text())
FEAT, CATS = meta["features"], meta["categorical"]
train = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
test = pl.read_parquet(M.FEAT_DIR / "test_v1.parquet")
y = train["target"].to_numpy()
levels = {c: sorted(set(train[c].drop_nulls().to_list()) | set(test[c].drop_nulls().to_list())) for c in CATS}
X = M._cat_frame(M._to_pandas(train, FEAT, levels), CATS)
Xt = M._cat_frame(M._to_pandas(test, FEAT, levels), CATS)
NUM = [c for c in FEAT if c not in CATS]


def prep(seed):
    """Sama seperti logreg_v1: kuantil-normal + indikator null + one-hot (di-fit hanya di fold latih)."""
    return ColumnTransformer([
        ("num", make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed),
                              SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), CATS)])


def model(kind, seed):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    if kind == "svm_lin":
        return make_pipeline(prep(seed), LinearSVC(C=0.005, dual=False, max_iter=5000))
    if kind == "svm_rbf":   # kernel RBF diaproksimasi (SVM RBF penuh terlalu lambat untuk 56 ribu baris)
        return make_pipeline(prep(seed), Nystroem(gamma=0.0005, n_components=2000, random_state=seed),
                             LinearSVC(C=0.3, dual=False, max_iter=5000))
    if kind == "ridge":
        return make_pipeline(prep(seed), RidgeClassifier(alpha=300.0))
    if kind == "lda":
        return make_pipeline(prep(seed), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if kind == "knn":
        return make_pipeline(prep(seed), PCA(10, whiten=True, random_state=seed), KNeighborsClassifier(200, n_jobs=-1))
    if kind == "nb":
        return make_pipeline(prep(seed), GaussianNB())
    if kind == "rf":
        return make_pipeline(prep(seed), RandomForestClassifier(300, min_samples_leaf=40, max_features=0.2,
                                                                n_jobs=-1, random_state=seed))
    if kind == "et":
        return make_pipeline(prep(seed), ExtraTreesClassifier(400, min_samples_leaf=30, max_features=0.3,
                                                              n_jobs=-1, random_state=seed))
    if kind == "ebm":       # Explainable Boosting Machine: GAM + 10 interaksi pasangan
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(interactions=10, max_bins=128, outer_bags=4, n_jobs=-1, random_state=seed)
    raise ValueError(kind)


def score(m, Z):
    return m.predict_proba(Z)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Z)


def run(kind):
    name = f"extra_{kind}"
    if (M.PRED_DIR / f"{name}_meta.json").exists():
        r = M.run_from_disk(name, y)
        if all(s in r["seeds"] for s in SEEDS):
            print(f"{name}: dimuat dari disk")
            return r
    oof_by_seed, test_pred, t0 = {}, np.zeros(len(Xt)), time.time()
    for s in SEEDS:
        oof = np.zeros(len(y))
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            m = model(kind, s).fit(X.iloc[tr_i], y[tr_i])
            oof[va_i] = score(m, X.iloc[va_i])
            test_pred += M.rank01(score(m, Xt)) / (5 * len(SEEDS))
        oof_by_seed[s] = oof
        print(f"  {kind} seed {s}: KS {M.ks_score(y, oof):.4f} | {time.time() - t0:.0f}s", flush=True)
    r = dict(oof_by_seed=oof_by_seed, seeds=list(SEEDS), test_pred=test_pred, oof=np.mean(list(oof_by_seed.values()), axis=0),
             features=FEAT, params={"kind": kind}, best_iters=[0])
    r["ks_oof"] = M.ks_score(y, r["oof"])
    r["ks_by_seed"] = {s: M.ks_score(y, o) for s, o in oof_by_seed.items()}
    M.save_run(name, train, test, r, extra=dict(kind=kind, minutes=round((time.time() - t0) / 60, 1)))
    return r


kinds = sys.argv[1].split(",") if len(sys.argv) > 1 else MODELS
CORE = {"lgbm": "lgbm_tuned", "xgb": "xgb_v1", "cat": "cat_v1", "logreg": "logreg_v1"}
R = {k: M.run_from_disk(v, y) for k, v in CORE.items()}
REF = M.blend(R, y, seeds=list(SEEDS))
rows, extra = [], {}
for kind in kinds:
    r = run(kind)
    extra[kind] = r
    c5 = M.compare_runs(REF, M.blend({**R, kind: r}, y, seeds=list(SEEDS)), y)
    c10 = M.compare_runs(REF, M.blend({**R, kind: r}, y, weights={**{k: 0.9 / 4 for k in R}, kind: 0.1},
                                      seeds=list(SEEDS)), y)
    rows.append(dict(model=kind, ks_per_seed=np.mean(list(r["ks_by_seed"].values())),
                     korelasi_dgn_ensemble=np.mean([np.corrcoef(M.rank01(r["oof_by_seed"][s]), M.rank01(REF["oof_by_seed"][s]))[0, 1]
                                                    for s in SEEDS]),
                     d_bobot_sama=c5["delta_seed_mean"], naik_bobot_sama=f"{c5['n_seed_positive']}/{len(SEEDS)}",
                     d_oof_bobot_sama=c5["delta_oof"], d_bobot_10pct=c10["delta_seed_mean"],
                     keputusan=M.verdict(c5, noise=0.0015)))
if len(extra) > 1:
    c = M.compare_runs(REF, M.blend({**R, **extra}, y, seeds=list(SEEDS)), y)
    rows.append(dict(model="SEMUA tambahan sekaligus", d_bobot_sama=c["delta_seed_mean"],
                     naik_bobot_sama=f"{c['n_seed_positive']}/{len(SEEDS)}", d_oof_bobot_sama=c["delta_oof"],
                     keputusan=M.verdict(c, noise=0.0015)))
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(220)
print(f"\nEnsemble saat ini (sub_v4): KS per seed {np.mean(list(REF['ks_by_seed'].values())):.4f} | "
      f"OOF gabungan {REF['ks_oof']:.4f}")
print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(4)))
