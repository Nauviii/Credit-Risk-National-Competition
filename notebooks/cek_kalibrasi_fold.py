"""Cek cepat: apakah OOF sebuah run dirugikan oleh skala prediksi yang berbeda antar fold?
selisih = KS setelah diperingkat DI DALAM tiap fold - KS OOF gabungan biasa. GBDT/logreg ~0; set model +0,003..0,007.
Jalankan dari folder notebooks/ (python cek_kalibrasi_fold.py) atau tempel ke satu sel notebook."""
import numpy as np
import polars as pl
from scipy.stats import rankdata
import modeling as M

train = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
y = train["target"].to_numpy()
folds = M.fold_table(train)

def rank_in_fold(o, f):
    out = np.zeros(len(o))
    for k in np.unique(f):
        m = f == k
        out[m] = rankdata(o[m]) / m.sum()
    return out

rows = []
for name in ["lgbm_tuned", "xgb_v1", "cat_v1", "logreg_v1", "realmlp_v1", "tabm_v1", "tabicl_v1", "setmodel_v1"]:
    if not (M.PRED_DIR / f"{name}_meta.json").exists():
        continue
    r = M.run_from_disk(name, y)
    for s in [s for s in r["seeds"] if s in (42, 7, 2024)]:
        o, f = r["oof_by_seed"][s], folds[f"fold_s{s}"].to_numpy()
        rows.append(dict(run=name, seed=s, ks_gabung=M.ks_score(y, o),
                         ks_rank_per_fold=M.ks_score(y, rank_in_fold(o, f))))
pl.Config.set_tbl_rows(40)
print(pl.DataFrame(rows).with_columns(selisih=pl.col("ks_rank_per_fold") - pl.col("ks_gabung"))
        .with_columns(pl.col(pl.Float64).round(4)))
