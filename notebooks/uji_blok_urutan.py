"""Uji blok fitur urutan (features_seq.py) di lgbm_tuned: blok penuh (55 kolom) vs plasebo (blok yang sama, baris diacak).
Jalankan dari folder notebooks/: python uji_blok_urutan.py  (± 5–10 menit, 3 seed × 2 model)."""
import json, sys, numpy as np, polars as pl
import eda_utils as E, modeling as M, features_seq as Q
from sklearn.metrics import roc_auc_score
meta = json.loads((M.FEAT_DIR/'features_v1.json').read_text()); FEAT, CATS = meta['features'], meta['categorical']
pa, rh = E.load('previous_applications'), E.load('repayment_history')
tr = pl.read_parquet(M.FEAT_DIR/'train_v1.parquet'); te = pl.read_parquet(M.FEAT_DIR/'test_v1.parquet'); y = tr['target'].to_numpy()
sq_tr = Q.seq_features(E.load('train_applications'), pa, rh); sq_te = Q.seq_features(E.load('test_applications'), pa, rh)
SQ = Q.seq_columns(sq_tr)
def auc(c, df):
    x = df[c].cast(pl.Float64).to_numpy(); m = ~np.isnan(x); a = roc_auc_score(y[m], x[m]); return round(max(a, 1-a), 3)
for c in ('rp_late_share', 'rp_late_share_365d', 'rp_late_share_180d', 'rp_mean_dpd'):
    print(c, auc(c, tr), end=' | ')
print()
tr2 = tr.join(sq_tr.drop('customer_id'), on='application_id', how='left'); te2 = te.join(sq_te.drop('customer_id'), on='application_id', how='left')
assert tr2['application_id'].equals(tr['application_id'])
rng = np.random.default_rng(0); perm = rng.permutation(tr2.height)
tr_pl = tr.hstack(sq_tr.select(SQ)[perm].rename({c: c + '_pl' for c in SQ}))
te_pl = te.hstack(sq_te.select(SQ).rename({c: c + '_pl' for c in SQ}))
L = M.run_from_disk('lgbm_tuned', y); P = {k: v for k, v in L['params'].items() if k != 'random_state'}
for name, (a, b, f) in {'+ blok urutan (sq_*)': (tr2, te2, FEAT + SQ), 'plasebo (sq_* diacak)': (tr_pl, te_pl, FEAT + [c + '_pl' for c in SQ])}.items():
    r = M.train_lgb_cv(a, b, f, CATS, seeds=(42, 7, 2024), params=P, verbose=False)
    c = M.compare_runs(L, r, y, tr['segment'].to_numpy())
    print(f"{name:24s} Δ OOF {c['delta_oof']:+.4f} | per seed {[round(d,4) for d in c['per_seed']['delta']]} | rata2 {c['delta_seed_mean']:+.4f}", flush=True)
    if name.startswith('+'):
        print(c['per_segment'].with_columns(pl.col(pl.Float64).round(4)))
        print(r['importance'].filter(pl.col('feature').str.starts_with('sq_')).head(10).with_columns(pl.col('gain_pct').round(2)),
              'total gain sq_ %:', round(r['importance'].filter(pl.col('feature').str.starts_with('sq_'))['gain_pct'].sum(), 1), flush=True)
