"""Set model untuk item e-commerce mentah (notebook 12: uji kelayakan 'graph relasional per nasabah').

Setiap nasabah = himpunan item e-commerce sebelum cutoff. Model membaca SETIAP item (bukan agregat buatan kita),
mengubahnya menjadi vektor, lalu menggabungkan (pooling) vektor-vektor itu dengan bobot yang DIPELAJARI
(attention) + rata-rata + maksimum. Ini setara GNN satu lapis pada pohon nasabah → item.

Fitur per item (semuanya tersedia sebelum cutoff):
- premium harga: log(harga per unit / median kategori sebelum 2025)
- log harga per unit, log jumlah, log total order, jumlah item dalam order yang sama
- umur transaksi (hari sebelum cutoff, diskalakan + log)
- status (completed/cancelled), kategori (embedding), metode bayar (embedding)
"""
import time

import numpy as np
import polars as pl

import modeling as M

CATEGORIES = ["groceries", "fashion", "electronics", "home", "health", "education", "personal_care"]
PAYMENTS = ["COD", "e_wallet", "bank_transfer", "debit_card", "credit_card"]
NUM_COLS = ["lp", "log_unit", "log_qty", "log_order_total", "log_items_in_order", "age_y", "log_age", "cancelled"]


# ------------------------------------------------------------------------------------------------ data
def item_table(ec, app, cat_medians, batch_size=20_000):
    """Item e-commerce sebelum cutoff untuk nasabah di `app`, dengan fitur per item. Diproses per kelompok nasabah
    (hemat RAM untuk file penuh). Return DataFrame terurut (customer_id, waktu)."""
    lf = ec.lazy() if isinstance(ec, pl.DataFrame) else ec
    cut = app.select("customer_id", pl.col("decision_datetime").alias("cutoff")).sort("customer_id")
    parts = []
    for i in range(0, cut.height, batch_size):
        c = cut.slice(i, batch_size)
        sub = (lf.filter(pl.col("customer_id").is_in(c["customer_id"].to_list()))
                 .join(c.lazy(), on="customer_id", how="inner")
                 .filter(pl.col("transaction_datetime") < pl.col("cutoff"))
                 .join(cat_medians.lazy(), on="product_category", how="left")
                 .with_columns(unit=pl.col("item_total") / pl.col("quantity"),
                               age_d=(pl.col("cutoff") - pl.col("transaction_datetime")).dt.total_seconds() / 86400,
                               items_in_order=pl.len().over("order_id"))
                 .select("customer_id", "transaction_datetime",
                         lp=(pl.col("unit") / pl.col("cat_median_unit")).log(),
                         log_unit=pl.col("unit").log(),
                         log_qty=pl.col("quantity").cast(pl.Float64).log1p(),
                         log_order_total=pl.col("order_total").cast(pl.Float64).log1p(),
                         log_items_in_order=pl.col("items_in_order").cast(pl.Float64).log1p(),
                         age_y=pl.col("age_d") / 365,
                         log_age=pl.col("age_d").log1p(),
                         cancelled=(pl.col("order_status") != "completed").cast(pl.Float64),
                         cat=pl.col("product_category").replace_strict({c: i for i, c in enumerate(CATEGORIES)}, default=len(CATEGORIES)),
                         pay=pl.col("payment_method").replace_strict({p: i for i, p in enumerate(PAYMENTS)}, default=len(PAYMENTS)))
                 .collect())
        parts.append(sub)
    return pl.concat(parts).sort("customer_id", "transaction_datetime")


def to_arrays(items, customer_ids):
    """Array datar + offset per nasabah, mengikuti urutan `customer_ids` (urutan baris train/test).
    Nasabah tanpa item → panjang 0 (model memakai vektor 'kosong' yang dipelajari)."""
    order = pl.DataFrame({"customer_id": customer_ids, "_pos": np.arange(len(customer_ids))})
    it = items.join(order, on="customer_id", how="inner").sort("_pos", "transaction_datetime")
    counts = (order.join(it.group_by("_pos").agg(n=pl.len()), on="_pos", how="left")
                   .sort("_pos")["n"].fill_null(0).cast(pl.Int64).to_numpy())
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    num = it.select(NUM_COLS).to_numpy().astype(np.float32)
    cat = it.select("cat", "pay").to_numpy().astype(np.int64)
    return dict(num=num, cat=cat, offsets=offsets, n_items=counts)


def standardize(arr, train_rows_mask_items):
    """Skala fitur numerik item dengan rata-rata/SD dari item nasabah fold latih saja."""
    x = arr["num"]
    mu = np.nanmean(x[train_rows_mask_items], axis=0)
    sd = np.nanstd(x[train_rows_mask_items], axis=0) + 1e-6
    return np.nan_to_num((x - mu) / sd).astype(np.float32)


# ------------------------------------------------------------------------------------------------ model
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(n_num, d=64, n_queries=4):
    torch, nn = _torch()

    class SetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_emb = nn.Embedding(len(CATEGORIES) + 1, 8)
            self.pay_emb = nn.Embedding(len(PAYMENTS) + 1, 4)
            self.enc = nn.Sequential(nn.Linear(n_num + 12, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
            self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.1)
            self.empty = nn.Parameter(torch.zeros(d * (n_queries + 2)))
            self.head = nn.Sequential(nn.Linear(d * (n_queries + 2) + 1, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))

        def forward(self, num, cat, mask, log_n):
            h = self.enc(torch.cat([num, self.cat_emb(cat[..., 0]), self.pay_emb(cat[..., 1])], dim=-1))  # B,L,d
            att = torch.einsum("bld,qd->bql", h, self.queries) / h.shape[-1] ** 0.5
            att = att.masked_fill(~mask[:, None, :], -1e4).softmax(-1)
            pooled_att = torch.einsum("bql,bld->bqd", att, h).flatten(1)
            m = mask[..., None].float()
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
            mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
            z = torch.cat([pooled_att, mean, mx], dim=-1)
            has = (mask.sum(1) > 0)[:, None]
            z = torch.where(has, z, self.empty.expand_as(z))
            return self.head(torch.cat([z, log_n[:, None]], dim=-1)).squeeze(-1)

    return SetModel()



def concat_arrays(a, b):
    """Gabungkan array train (a) dan test (b): baris b menjadi indeks len(a['n_items']) .. seterusnya."""
    return dict(num=np.concatenate([a["num"], b["num"]]), cat=np.concatenate([a["cat"], b["cat"]]),
                offsets=np.concatenate([a["offsets"], b["offsets"][1:] + a["offsets"][-1]]),
                n_items=np.concatenate([a["n_items"], b["n_items"]]))


# ------------------------------------------------------------------------------------------------ skala antar model
def to_pct(p, ref):
    """Persentil tiap nilai `p` di dalam populasi acuan `ref` (0..1; nilai kembar → titik tengah).
    Set model yang berhenti di epoch berbeda punya RENTANG skor berbeda. Kalau skornya langsung digabung
    (OOF dari 5 fold, atau skor stacking dari 4 model dalam), ambang yang sama berarti hal berbeda di tiap
    bagian → KS gabungan turun dan fitur stacking jadi berisik. Memetakan semua skor ke persentil populasi
    acuan yang SAMA (baris test, diprediksi oleh setiap model) menyamakan skalanya."""
    r = np.sort(np.asarray(ref, dtype=float))
    p = np.asarray(p, dtype=float)
    return (np.searchsorted(r, p, "left") + np.searchsorted(r, p, "right")) / (2 * len(r))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def rank_in_fold(oof, fold):
    """Peringkat (0..1) di dalam tiap fold → OOF gabungan bebas dari beda skala antar fold."""
    out = np.zeros(len(oof))
    for k in np.unique(fold):
        m = fold == k
        out[m] = to_pct(oof[m], oof[m])
    return out


def rank_run_in_fold(run, folds, y):
    """Salinan run (format train_lgb_cv / run_from_disk) dengan OOF tiap seed diperingkat di dalam fold.
    Prediksi test tidak diubah (sudah rata-rata semua model, tidak punya masalah ini)."""
    obs = {s: rank_in_fold(o, folds[f"fold_s{s}"].to_numpy()) for s, o in run["oof_by_seed"].items()}
    r = {**run, "oof_by_seed": obs, "oof": np.mean(list(obs.values()), axis=0)}
    r["ks_oof"] = M.ks_score(y, r["oof"])
    r["ks_by_seed"] = {s: M.ks_score(y, o) for s, o in obs.items()}
    return r


# ------------------------------------------------------------------------------------------------ pelatihan
def _batch(arr, num_std, idx, device):
    torch, _ = _torch()
    lens = arr["n_items"][idx]
    L = max(int(lens.max()), 1)
    B, F = len(idx), num_std.shape[1]
    num = np.zeros((B, L, F), np.float32); cat = np.zeros((B, L, 2), np.int64); mask = np.zeros((B, L), bool)
    for b, i in enumerate(idx):
        s, e = arr["offsets"][i], arr["offsets"][i + 1]
        n = e - s
        if n:
            num[b, :n] = num_std[s:e]; cat[b, :n] = arr["cat"][s:e]; mask[b, :n] = True
    t = lambda a: torch.from_numpy(a).to(device)
    return t(num), t(cat), t(mask), t(np.log1p(lens).astype(np.float32))


def _length_batches(idx, lens, bs, rng, pool=32):
    """Batch berisi nasabah dengan jumlah item mirip → padding jauh lebih sedikit (~2-3x lebih cepat).
    Diacak dulu, diurutkan per kelompok `pool` batch, lalu urutan batch diacak lagi."""
    idx = rng.permutation(idx)
    out = []
    for i in range(0, len(idx), bs * pool):
        c = idx[i:i + bs * pool]
        c = c[np.argsort(lens[c], kind="stable")]
        out += [c[j:j + bs] for j in range(0, len(c), bs)]
    return [out[o] for o in rng.permutation(len(out))]


def _predict(model, arr, num_std, idx, device, bs=1024):
    torch, _ = _torch()
    idx = np.asarray(idx)
    out = np.zeros(len(idx))
    if len(idx) == 0:
        return out
    order = np.argsort(arr["n_items"][idx], kind="stable")
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sel = order[i:i + bs]
            out[sel] = torch.sigmoid(model(*_batch(arr, num_std, idx[sel], device))).cpu().numpy()
    return out


def fit_set(arr, y, tr_i, va_i, pred_sets, seed, device="cpu", epochs=30, patience=4, bs=256, lr=2e-3):
    """Latih SATU set model di baris tr_i, early stopping pada AUC baris va_i, lalu prediksi tiap himpunan indeks
    di pred_sets (indeks ke `arr`; boleh berisi baris test). Return (list prediksi, epoch terbaik)."""
    torch, nn = _torch()
    from sklearn.metrics import roc_auc_score
    item_owner = np.repeat(np.arange(len(arr["n_items"])), arr["n_items"])
    num_std = standardize(arr, np.isin(item_owner, tr_i))          # skala dari item fold latih saja
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build_model(num_std.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf, yt = nn.BCEWithLogitsLoss(), torch.from_numpy(y.astype(np.float32))
    best, best_state, best_ep, bad = -1, None, 0, 0
    for ep in range(epochs):
        model.train()
        for idx in _length_batches(tr_i, arr["n_items"], bs, rng):
            opt.zero_grad()
            lossf(model(*_batch(arr, num_std, idx, device)), yt[idx].to(device)).backward()
            opt.step()
        auc = roc_auc_score(y[va_i], _predict(model, arr, num_std, va_i, device))
        if auc > best:
            best, best_ep, bad = auc, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return [_predict(model, arr, num_std, ix, device) for ix in pred_sets], best_ep


def train_set_cv(arr, y, folds, seeds=(42, 7, 2024), test_idx=None, device="cpu", calibrate=True, verbose=True, **cfg):
    """CV set model dengan fold dari modeling.fold_table (identik dengan model lain).
    arr: array train (+ test di belakangnya, lihat concat_arrays); y: target train; test_idx: indeks baris test di arr.
    calibrate=True: OOF tiap fold = persentil terhadap prediksi test model fold itu (skala sama antar fold),
    prediksi test = rata-rata persentil. calibrate=False = perilaku notebook 12 (probabilitas mentah).
    Return dict format train_lgb_cv (oof_by_seed, test_pred, ks_by_seed, ...) → bisa disimpan dengan M.save_run."""
    n = len(y)
    test_idx = np.array([], dtype=np.int64) if test_idx is None else np.asarray(test_idx)
    oof_by_seed, test_pred, rows, eps, t0 = {}, np.zeros(len(test_idx)), [], [], time.time()
    for seed in seeds:
        f = folds[f"fold_s{seed}"].to_numpy()
        oof = np.zeros(n)
        for k in sorted(set(f)):
            tr_i, va_i = np.flatnonzero(f != k), np.flatnonzero(f == k)
            t = time.time()
            (p_va, p_te), ep = fit_set(arr, y, tr_i, va_i, [va_i, test_idx], seed * 100 + int(k), device=device, **cfg)
            if calibrate:
                ref = p_te if len(test_idx) else p_va
                oof[va_i] = to_pct(p_va, ref)
                if len(test_idx):
                    test_pred += to_pct(p_te, p_te) / (len(seeds) * len(set(f)))
            else:
                oof[va_i] = p_va
                if len(test_idx):
                    test_pred += p_te / (len(seeds) * len(set(f)))
            eps.append(ep)
            rows.append(dict(seed=seed, fold=int(k), ks=M.ks_score(y[va_i], p_va), epoch_terbaik=ep,
                             detik=round(time.time() - t, 1)))
            if verbose:
                print(f"[set] seed {seed} fold {k}: KS {rows[-1]['ks']:.4f} | epoch terbaik {ep} | {rows[-1]['detik']}s",
                      flush=True)
        oof_by_seed[seed] = oof
    res = dict(oof_by_seed=oof_by_seed, test_pred=test_pred, seeds=list(seeds), folds=pl.DataFrame(rows),
               oof=np.mean(list(oof_by_seed.values()), axis=0), importance=None, best_iters=eps,
               features=NUM_COLS + ["cat", "pay"], params={"kind": "set_model", "calibrate": calibrate, **cfg})
    res["ks_oof"] = M.ks_score(y, res["oof"])
    res["ks_by_seed"] = {s: M.ks_score(y, o) for s, o in oof_by_seed.items()}
    if verbose:
        print(f"[set] KS OOF {len(seeds)} seed: {res['ks_oof']:.4f} | per seed: "
              + ", ".join(f"{v:.4f}" for v in res["ks_by_seed"].values()) + f" | total {time.time() - t0:.0f}s")
    return res


# ------------------------------------------------------------------------------------------------ stacking bersarang
def stacked_lgb_cv(train, test, features, cats, arr, folds, seeds, params, test_idx, inner_folds=4, device="cpu",
                   calibrate=True, y_col="target", verbose=True, **cfg):
    """LightGBM (fitur v1 + `set_score`) dengan stacking BERSARANG, tanpa kebocoran label:
    untuk fold luar k, set_score baris latih = OOF dari CV dalam yang hanya memakai baris latih fold k;
    set_score baris validasi & test = gabungan model-model dalam itu.
    calibrate=True (perbaikan notebook 12b): semua skor dipetakan ke persentil populasi test.
      - baris latih: skor model dalam j → persentil terhadap prediksi test model j
      - baris validasi & test: rata-rata logit 4 model dalam → persentil terhadap rata-rata logit di test
      Jadi baris latih, validasi, dan test punya skala & sebaran yang sama.
    calibrate=False = versi notebook 12 (probabilitas mentah; skala berbeda antar model dalam).
    Sekalian dilatih PLASEBO: LightGBM yang sama dengan set_score diacak (kolom tanpa informasi), untuk mengukur
    seberapa besar KS berubah hanya karena menambah satu kolom (pengambilan kolom acak LightGBM ikut berubah).
    Return (res_stack, res_placebo), format train_lgb_cv."""
    import warnings
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    y = train[y_col].to_numpy()
    cats = [c for c in cats if c in features]
    levels = {c: sorted(set(train[c].drop_nulls().to_list()) | set(test[c].drop_nulls().to_list())) for c in cats}
    X, Xt = M._to_pandas(train, features, levels), M._to_pandas(test, features, levels)
    p = {**M.LGB_PARAMS, **params}
    out = {m: dict(oof_by_seed={}, test_pred=np.zeros(len(Xt)), rows=[], iters=[]) for m in ("stack", "placebo")}
    t0 = time.time()
    for seed in seeds:
        f = folds[f"fold_s{seed}"].to_numpy()
        oofs = {m: np.zeros(len(y)) for m in out}
        for k in sorted(set(f)):
            tr_i, va_i = np.flatnonzero(f != k), np.flatnonzero(f == k)
            score, score_t = np.zeros(len(y)), np.zeros(len(Xt))
            lg_va, lg_te = np.zeros(len(va_i)), np.zeros(len(Xt))
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed + int(k))
            for j, (a, b) in enumerate(inner.split(tr_i, y[tr_i])):
                (p_b, p_va, p_te), _ = fit_set(arr, y, tr_i[a], tr_i[b], [tr_i[b], va_i, test_idx],
                                               seed * 1000 + int(k) * 10 + j, device=device, **cfg)
                if calibrate:
                    score[tr_i[b]] = to_pct(p_b, p_te)
                    lg_va += _logit(p_va) / inner_folds
                    lg_te += _logit(p_te) / inner_folds
                else:
                    score[tr_i[b]] = p_b
                    score[va_i] += p_va / inner_folds
                    score_t += p_te / inner_folds
            if calibrate:
                score[va_i] = to_pct(lg_va, lg_te)
                score_t = to_pct(lg_te, lg_te)
            # diagnostik: KS skor set model sendiri di baris latih (OOF dalam, digabung) vs baris validasi
            ks_tr, ks_va = M.ks_score(y[tr_i], score[tr_i]), M.ks_score(y[va_i], score[va_i])
            rng = np.random.default_rng(seed * 10 + int(k))
            cols = {"stack": (score, score_t), "placebo": (rng.permutation(score), rng.permutation(score_t))}
            for m, (s_tr, s_te) in cols.items():
                Xs, Xts = X.assign(set_score=s_tr), Xt.assign(set_score=s_te)
                mdl = lgb.LGBMClassifier(**{**p, "random_state": seed})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl.fit(Xs.iloc[tr_i], y[tr_i], eval_set=[(Xs.iloc[va_i], y[va_i])], eval_metric="auc",
                            callbacks=[lgb.early_stopping(100, verbose=False)])
                oofs[m][va_i] = mdl.predict_proba(Xs.iloc[va_i])[:, 1]
                out[m]["test_pred"] += mdl.predict_proba(Xts)[:, 1] / (len(seeds) * len(set(f)))
                gain = dict(zip(Xs.columns, mdl.booster_.feature_importance("gain")))
                out[m]["iters"].append(mdl.best_iteration_)
                out[m]["rows"].append(dict(seed=seed, fold=int(k), ks=M.ks_score(y[va_i], oofs[m][va_i]),
                                           gain_set_score_pct=gain["set_score"] / sum(gain.values()) * 100,
                                           ks_skor_latih=ks_tr, ks_skor_val=ks_va))
            if verbose:
                r = out["stack"]["rows"][-1]
                print(f"[stack] seed {seed} fold {k}: KS {r['ks']:.4f} (plasebo {out['placebo']['rows'][-1]['ks']:.4f}) "
                      f"| gain set_score {r['gain_set_score_pct']:.1f}% | KS skor latih/val {ks_tr:.4f}/{ks_va:.4f} "
                      f"| {time.time() - t0:.0f}s", flush=True)
        for m in out:
            out[m]["oof_by_seed"][seed] = oofs[m]
    res = {}
    for m, o in out.items():
        r = dict(oof_by_seed=o["oof_by_seed"], test_pred=o["test_pred"], seeds=list(seeds), folds=pl.DataFrame(o["rows"]),
                 oof=np.mean(list(o["oof_by_seed"].values()), axis=0), importance=None, best_iters=o["iters"],
                 features=list(features) + ["set_score"], params={**p, "stacking": m, "inner_folds": inner_folds, "calibrate": calibrate})
        r["ks_oof"] = M.ks_score(y, r["oof"])
        r["ks_by_seed"] = {s: M.ks_score(y, v) for s, v in o["oof_by_seed"].items()}
        res[m] = r
    if verbose:
        print(f"[stack] KS OOF {res['stack']['ks_oof']:.4f} | plasebo {res['placebo']['ks_oof']:.4f} "
              f"| total {time.time() - t0:.0f}s")
    return res["stack"], res["placebo"]