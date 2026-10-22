"""Rekap semua kandidat vs ensemble saat ini (sub_v4), dari run yang sudah tersimpan di data/preds. Tanpa training.

Jalankan dari folder notebooks/:  python rekap_kandidat.py   (atau tempel ke satu sel notebook)

- Pembanding (REF) = rata-rata peringkat bobot sama lgbm_tuned + xgb_v1 + cat_v1 + logreg_v1, seed 42, 7, 2024.
- Semua perbandingan berpasangan di seed yang sama (42, 7, 2024).
- RANK_PER_FOLD = True: OOF SEMUA run diperingkat di dalam tiap fold dulu (perlakuan sama; menghilangkan beda skala
  antar fold pada model NN/set model, lihat cek_kalibrasi_fold.py). Untuk GBDT/logreg efeknya ≈ 0.
- Run yang tidak ada di disk otomatis dilewati.
"""
import numpy as np
import polars as pl
from scipy.stats import rankdata

import modeling as M

SEEDS = [42, 7, 2024]
RANK_PER_FOLD = True
NOISE, THIN = 0.0015, 0.002

train = pl.read_parquet(M.FEAT_DIR / "train_v1.parquet")
y = train["target"].to_numpy()
FOLDS = M.fold_table(train)


def rank_in_fold(o, f):
    out = np.zeros(len(o))
    for k in np.unique(f):
        m = f == k
        out[m] = rankdata(o[m]) / m.sum()
    return out


def load(names, seeds=SEEDS):
    """Muat run (boleh beberapa file, mis. run asli + run seed tambahan) dan ambil seed `seeds` saja."""
    names = [names] if isinstance(names, str) else names
    found = {}
    test_pred = None
    for name in names:
        if name and (M.PRED_DIR / f"{name}_meta.json").exists():
            r = M.run_from_disk(name, y)
            found.update({s: o for s, o in r["oof_by_seed"].items() if s in seeds and s not in found})
            test_pred = r["test_pred"] if test_pred is None else test_pred
    if not all(s in found for s in seeds):
        return None
    if RANK_PER_FOLD:
        obs = {s: rank_in_fold(found[s], FOLDS[f"fold_s{s}"].to_numpy()) for s in seeds}
    else:
        obs = {s: found[s] for s in seeds}
    out = dict(oof_by_seed=obs, seeds=list(seeds), test_pred=test_pred)
    out["ks_by_seed"] = {s: M.ks_score(y, o) for s, o in obs.items()}
    out["ks_oof"] = M.ks_score(y, np.mean(list(obs.values()), axis=0))
    return out


def stage_b_config(kind):
    """Konfigurasi yang masuk tahap B notebook 13 (punya file t13_<cfg>_s11_99)."""
    for p in sorted(M.PRED_DIR.glob(f"t13_{kind}_*_s11_99_meta.json")):
        return p.name.replace("_s11_99_meta.json", "")
    return None


# ------------------------------------------------------------------ run yang tersedia
CORE = {"lgbm": "lgbm_tuned", "xgb": "xgb_v1", "cat": "cat_v1", "logreg": "logreg_v1"}
EXTRA = {"realmlp": "realmlp_v1", "tabm": "tabm_v1", "tabicl": "tabicl_v1", "set": "setmodel_v1",
         "lgbm_stack": "lgbm_stack_set_v2", "xgb_tuned": stage_b_config("xgb"), "cat_tuned": stage_b_config("cat")}
R = {k: load(v) for k, v in CORE.items()}
assert all(R.values()), "run inti (lgbm_tuned, xgb_v1, cat_v1, logreg_v1) harus ada"
R.update({k: load(v) for k, v in EXTRA.items() if v})
R = {k: v for k, v in R.items() if v is not None}
print("Run tersedia:", {k: (CORE | EXTRA)[k] for k in R})

REF = M.blend({k: R[k] for k in CORE}, y, seeds=SEEDS)
print(f"Pembanding sub_v4: KS OOF gabungan {REF['ks_oof']:.4f} (rank per fold: {RANK_PER_FOLD})\n")

# ------------------------------------------------------------------ 1. model tunggal
rows = [dict(model=k, ks_rata2_per_seed=np.mean(list(r["ks_by_seed"].values())), ks_oof_gabungan=r["ks_oof"],
             vs_sub_v4=r["ks_oof"] - REF["ks_oof"]) for k, r in R.items()]
print("1. Model tunggal")
print(pl.DataFrame(rows).sort("ks_oof_gabungan", descending=True).with_columns(pl.col(pl.Float64).round(4)), "\n")

# ------------------------------------------------------------------ 2. kombinasi sederhana (satu perubahan)
SWAP = {"lgbm_stack": "lgbm", "xgb_tuned": "xgb", "cat_tuned": "cat"}    # pengganti anggota
ADD = ["realmlp", "tabm", "tabicl", "set"]                               # anggota tambahan
single = {}
for k in SWAP:
    if k in R:
        single[f"{SWAP[k]} → {k}"] = {**{m: R[m] for m in CORE}, SWAP[k]: R[k]}
for k in ADD:
    if k in R:
        single[f"+ {k}"] = {**{m: R[m] for m in CORE}, k: R[k]}


def row(name, members):
    b = M.blend(members, y, seeds=SEEDS)
    c = M.compare_runs(REF, b, y)
    return dict(kandidat=name, delta_oof_gabungan=c["delta_oof"],
                **{f"d_s{s}": d for s, d in zip(c["per_seed"]["seed"], c["per_seed"]["delta"])},
                delta_rata2=c["delta_seed_mean"], seed_naik=f"{c['n_seed_positive']}/{len(SEEDS)}",
                keputusan=M.verdict(c, thin=THIN, noise=NOISE))


tab = pl.DataFrame([row(n, m) for n, m in single.items()]).sort("delta_oof_gabungan", descending=True)
print("2. Satu perubahan vs sub_v4")
print(tab.with_columns(pl.col(pl.Float64).round(4)), "\n")

# ------------------------------------------------------------------ 3. gabungan semua yang positif
pos = tab.filter(pl.col("delta_oof_gabungan") > 0)["kandidat"].to_list()
if len(pos) > 1:
    members = {m: R[m] for m in CORE}
    for n in pos:
        if " → " in n:
            old, new = n.split(" → ")
            members[old] = R[new]
        else:
            k = n.removeprefix("+ ")
            members[k] = R[k]
    print(f"3. Gabungan semua perubahan yang positif: {pos}")
    print("   (optimistis: kandidat dipilih SETELAH melihat hasilnya)")
    print(pl.DataFrame([row("semua yang positif", members)]).with_columns(pl.col(pl.Float64).round(4)))
else:
    print("3. Kurang dari 2 perubahan yang positif → tidak ada gabungan untuk diuji.")

# ------------------------------------------------------------------ 4. konfirmasi di seed BARU (11, 99)
# Kandidat di atas dipilih dengan seed 42/7/2024 (termasuk konfigurasi tuning di notebook 13). Seed 11 & 99 tidak ikut
# memilih apa pun, jadi di sinilah selisih yang "benar" kelihatan. Hanya run yang punya seed 11 & 99 yang bisa diuji
# (stacking set_score & model NN hanya punya 42/7/2024).
NEW = [11, 99]
RN = {"lgbm": load(["lgbm_tuned", "lgbm_tuned_s10"], NEW), "xgb": load(["xgb_v1_s11_99", "xgb_v1_s10"], NEW),
      "cat": load(["cat_v1_s11_99", "cat_v1_s10"], NEW), "logreg": load(["logreg_v1_s10", "logreg_v1_ext10"], NEW)}
for k in ("xgb", "cat"):
    cfg = EXTRA[f"{k}_tuned"]
    if cfg:
        RN[f"{k}_tuned"] = load([f"{cfg}_s11_99", f"{cfg}_s10"], NEW)
if all(RN[k] is not None for k in CORE):
    REF_N = M.blend({k: RN[k] for k in CORE}, y, seeds=NEW)
    cands = {}
    for k in ("xgb_tuned", "cat_tuned"):
        if RN.get(k) is not None:
            cands[f"{k[:3]} → {k}"] = {**{m: RN[m] for m in CORE}, k[:3]: RN[k]}
    if len(cands) == 2:
        cands["xgb & cat tuned"] = {**{m: RN[m] for m in CORE}, "xgb": RN["xgb_tuned"], "cat": RN["cat_tuned"]}
    rows4 = []
    for n, mem in cands.items():
        c = M.compare_runs(REF_N, M.blend(mem, y, seeds=NEW), y)
        rows4.append(dict(kandidat=n, delta_oof_gabungan=c["delta_oof"],
                          **{f"d_s{s}": d for s, d in zip(c["per_seed"]["seed"], c["per_seed"]["delta"])},
                          seed_naik=f"{c['n_seed_positive']}/{len(NEW)}"))
    print("\n4. Konfirmasi di seed baru 11 & 99 (tidak ikut memilih apa pun)")
    print(pl.DataFrame(rows4).with_columns(pl.col(pl.Float64).round(4)) if rows4 else "   tidak ada run tuning dengan seed 11 & 99")
else:
    print("\n4. Run seed 11 & 99 belum lengkap (jalankan notebook 13) → konfirmasi dilewati.")
print(f"\nAturan: tipis = Δ rata-rata per seed ≥ {THIN} dan > noise {NOISE}, semua seed naik; jelas = Δ ≥ 0.005.")
