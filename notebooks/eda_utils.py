"""Helper EDA untuk kompetisi credit risk (dipakai oleh semua notebook 02_*).

Isi:
- load / lazy           : baca Parquet hasil notebook 00
- ks_stat, auc_stat     : KS & AUC univariat (KS sama persis dengan scipy.ks_2samp)
- binned, iv_stat       : target rate per bin + WoE/IV
- signal_table          : peringkat kekuatan sinyal banyak fitur sekaligus
- ks_within             : KS fitur di dalam tiap strata (mis. per suku bunga)
- plot_rates            : small multiples target rate per bin (+ panel jumlah)
- heatmap_rate          : target rate 2 dimensi
- cv_lgb                : baseline LightGBM 5-fold → KS out-of-fold
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
PQ_DIR = ROOT / "data" / "parquet"

# ---------------------------------------------------------------- palet & gaya
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_RATE = SERIES[0]          # warna utama: target rate
C_COUNT = "#d6d5ce"         # batang jumlah: netral & resesif
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)


def setup_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 110,
        "font.family": "sans-serif",
    })


# ---------------------------------------------------------------- data
def load(name, columns=None):
    return pl.read_parquet(PQ_DIR / f"{name}.parquet", columns=columns)


def lazy(name):
    return pl.scan_parquet(PQ_DIR / f"{name}.parquet")


# ---------------------------------------------------------------- format angka
def fmt_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "null"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:.1f}M"
    if a >= 1e6:
        return f"{v/1e6:.1f}jt"
    if a >= 1e3:
        return f"{v/1e3:.0f}rb"
    if a >= 10 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------- metrik univariat
def _to_numeric(y, x):
    """String/kategori → diganti bad rate kategorinya (optimistis, wajar untuk kardinalitas kecil)."""
    x = np.asarray(x)
    if x.dtype == object or x.dtype.kind in "US":
        isnull = pd_isnull(x)
        xs = np.where(isnull, "__null__", x.astype(str))
        out = np.full(len(x), np.nan)
        for c in np.unique(xs[~isnull]):
            m = xs == c
            out[m] = y[m].mean()
        return out
    return x.astype(float)


def ks_stat(y, x):
    """KS dua sampel (bad vs good) dari satu skor/fitur. Null = nilai terendah tersendiri."""
    y = np.asarray(y, dtype=float)
    x = _to_numeric(y, np.asarray(x))
    nan = np.isnan(x)
    if nan.all():
        return 0.0
    x = np.where(nan, np.nanmin(x) - 1, x)
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(xs) != 0), len(xs) - 1]
    cb = np.cumsum(ys)[last] / ys.sum()
    cg = np.cumsum(1 - ys)[last] / (1 - ys).sum()
    return float(np.max(np.abs(cb - cg)))


def auc_stat(y, x):
    """AUC pada baris non-null. > 0.5 = nilai tinggi → risiko tinggi."""
    y = np.asarray(y, dtype=float)
    x = _to_numeric(y, np.asarray(x))
    m = ~np.isnan(x)
    y, x = y[m], x[m]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bin_masks(s, bins=10, min_share=0.003, min_n=150):
    """Daftar (label, mask) untuk satu kolom.
    - Kategori/boolean: satu bin per nilai.
    - Numerik diskret (≤ 24 nilai unik): satu bin per nilai; ekor yang tipis digabung jadi "≥ v".
    - Numerik kontinu: bin kuantil; massa titik besar (≥10%) di nilai minimum/maksimum (mis. banyak nol)
      dijadikan bin tersendiri "= v" agar tidak tercampur.
    - Null selalu bin terpisah bernama "null"."""
    n = len(s)
    if s.dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean):
        xs = s.cast(pl.String).to_numpy()
        isnull = pd_isnull(xs)
        out = [(str(v), xs == v) for v in sorted(set(xs[~isnull]))]
        if isnull.any():
            out.append(("null", isnull))
        return out
    x = s.cast(pl.Float64).to_numpy()
    isnull = np.isnan(x)
    pos = np.flatnonzero(~isnull)
    xv = x[pos]
    out = []

    def mask_of(sel):
        m = np.zeros(n, bool)
        m[pos[sel]] = True
        return m

    if len(xv):
        u, cnt = np.unique(xv, return_counts=True)
        floor = max(min_n, min_share * n)
        if len(u) <= max(bins, 24):
            i, acc = len(u) - 1, cnt[-1]               # gabungkan ekor atas yang tipis → "≥ v"
            while acc < floor and i > 1:
                i -= 1
                acc += cnt[i]
            j, acc = 0, cnt[0]                         # gabungkan ekor bawah yang tipis → "≤ v"
            while acc < floor and j < i - 1:
                j += 1
                acc += cnt[j]
            out.append((fmt_num(u[0]) if j == 0 else f"≤{fmt_num(u[j])}", mask_of(xv <= u[j])))
            for v in u[j + 1:i]:
                out.append((fmt_num(v), mask_of(xv == v)))
            if i > j:
                out.append((fmt_num(u[i]) if i == len(u) - 1 else f"≥{fmt_num(u[i])}", mask_of(xv >= u[i])))
        else:
            rest = np.ones(len(xv), bool)
            head, tail = [], []
            if (xv == u[0]).mean() >= 0.10:
                head = [(f"={fmt_num(u[0])}", mask_of(xv == u[0]))]
                rest &= xv != u[0]
            if (xv == u[-1]).mean() >= 0.10:
                tail = [(f"={fmt_num(u[-1])}", mask_of(xv == u[-1]))]
                rest &= xv != u[-1]
            out += head
            xr = xv[rest]
            if len(xr):
                k = max(bins - len(head) - len(tail), 2)
                edges = np.unique(np.quantile(xr, np.linspace(0, 1, k + 1)))
                if len(edges) == 1:
                    out.append((fmt_num(edges[0]), mask_of(rest)))
                else:
                    j = np.clip(np.searchsorted(edges, xr, side="right") - 1, 0, len(edges) - 2)
                    ridx = np.flatnonzero(rest)
                    for q in range(len(edges) - 1):
                        sel = np.zeros(len(xv), bool)
                        sel[ridx[j == q]] = True
                        if sel.any():
                            out.append((f"{fmt_num(edges[q])}–{fmt_num(edges[q+1])}", mask_of(sel)))
            out += tail
    if isnull.any():
        out.append(("null", isnull))
    return out


def _bin_stats(masks, yy, exp=None):
    base = yy.mean()
    tot_b, tot_g = yy.sum(), (1 - yy).sum()
    out = []
    for order, (lab, m) in enumerate(masks):
        n = int(m.sum())
        if n == 0:
            continue
        b = yy[m].sum()
        p = b / n
        z = 1.96
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        dg, db = ((n - b) + 0.5) / (tot_g + 0.5), (b + 0.5) / (tot_b + 0.5)
        woe = np.log(dg / db)
        row = dict(bin=lab, order=order, n=n, share=n / len(yy), bad=int(b), rate=p,
                   ci_lo=c - h, ci_hi=c + h, lift=p / base, woe=woe, iv=(dg - db) * woe)
        if exp is not None:
            e = exp[m].sum()
            row.update(oe=b / e, oe_lo=max(b - 1.96 * np.sqrt(b), 0) / e, oe_hi=(b + 1.96 * np.sqrt(b)) / e)
        out.append(row)
    return pl.DataFrame(out)


def binned(df, col, y="target", bins=10):
    """Tabel target rate per bin (+ bin null): n, share, rate, CI 95% Wilson, lift, WoE, IV."""
    return _bin_stats(_bin_masks(df[col], bins), df[y].to_numpy().astype(float))


def pd_isnull(a):
    a = np.asarray(a)
    if a.dtype != object:
        return np.isnan(a) if a.dtype.kind == "f" else np.zeros(len(a), bool)
    return np.array([v is None or (isinstance(v, float) and v != v) for v in a])


def iv_stat(df, col, y="target", bins=10):
    return float(binned(df, col, y, bins)["iv"].sum())


def signal_table(df, cols, y="target", bins=10):
    """Satu baris per fitur: null%, #unik, KS, AUC (arah), IV. Diurutkan berdasarkan KS."""
    yy = df[y].to_numpy()
    rows = []
    for c in cols:
        s = df[c]
        x = s.cast(pl.String).to_numpy() if s.dtype in (pl.String, pl.Categorical, pl.Boolean) else s.cast(pl.Float64).to_numpy()
        a = auc_stat(yy, x)
        rows.append(dict(feature=c, type="cat" if s.dtype in (pl.String, pl.Categorical, pl.Boolean) else "num",
                         null_pct=round(s.is_null().mean() * 100, 2), n_unique=s.n_unique(),
                         ks=round(ks_stat(yy, x), 4), auc=round(a, 4),
                         arah="↑ nilai → ↑ risiko" if a > 0.5 else "↑ nilai → ↓ risiko",
                         iv=round(iv_stat(df, c, y, bins), 4)))
    return pl.DataFrame(rows).sort("ks", descending=True)


def ks_noise_floor(y, n_perm=200, q=0.95, seed=0):
    """KS yang bisa muncul dari fitur ACAK MURNI pada label y ini (persentil ke-q).
    Fitur dengan KS di bawah angka ini tidak bisa dibedakan dari noise."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return float(np.quantile([ks_stat(y, rng.random(len(y))) for _ in range(n_perm)], q))


def _as_array(s):
    return s.cast(pl.String).to_numpy() if s.dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean) else s.cast(pl.Float64).to_numpy()


def ks_within(df, feature, strata, y="target", min_n=300, n_perm=30, seed=0):
    """KS fitur DI DALAM tiap strata (mis. tiap suku bunga) dibanding noise floor strata tsb.
    excess > 0 secara konsisten = fitur punya sinyal yang BELUM tertangkap oleh strata."""
    rows = []
    for (g,), part in df.group_by(strata):
        yy = part[y].to_numpy()
        if part.height < min_n or yy.sum() < 20:
            continue
        ks = ks_stat(yy, _as_array(part[feature]))
        floor = ks_noise_floor(yy, n_perm=n_perm, q=0.95, seed=seed)
        rows.append(dict(stratum=g, n=part.height, ks=ks, noise_p95=floor, excess=ks - floor))
    t = pl.DataFrame(rows).sort("stratum")
    w = t["n"] / t["n"].sum()
    summary = dict(feature=feature, ks_overall=ks_stat(df[y].to_numpy(), _as_array(df[feature])),
                   ks_within=float((t["ks"] * w).sum()), noise_within=float((t["noise_p95"] * w).sum()))
    summary["excess_within"] = summary["ks_within"] - summary["noise_within"]
    return summary, t


def binned_oe(df, col, expected, y="target", bins=10):
    """Observed / Expected per bin. expected = kolom prediksi risiko pembanding (mis. rata-rata
    target per suku bunga). O/E ≈ 1 di semua bin → fitur tidak menambah info di atas pembanding.
    CI = pendekatan Poisson untuk jumlah kejadian teramati."""
    return _bin_stats(_bin_masks(df[col], bins), df[y].to_numpy().astype(float), df[expected].to_numpy().astype(float))


# ---------------------------------------------------------------- plot
def _rate_panel(fig, spec, tab, title, base, show_ylabel=True, is_cat=False, ks=None):
    gs = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax, axn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    xs = np.arange(tab.height)
    is_null = (tab["bin"] == "null").to_numpy()
    rate, lo, hi = (tab[c].to_numpy() * 100 for c in ("rate", "ci_lo", "ci_hi"))
    ax.axhline(base * 100, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    nn = ~is_null
    if is_cat:   # kategori tidak berurutan → titik + error bar, tanpa garis penghubung
        ax.errorbar(xs[nn], rate[nn], yerr=[rate[nn] - lo[nn], hi[nn] - rate[nn]], fmt="o", color=C_RATE,
                    ms=6, lw=1.5, capsize=0, zorder=3)
    else:
        ax.fill_between(xs[nn], lo[nn], hi[nn], color=C_RATE, alpha=0.12, lw=0, zorder=2)
        ax.plot(xs[nn], rate[nn], color=C_RATE, lw=2, marker="o", ms=4.5, zorder=3)
    if is_null.any():
        ax.errorbar(xs[is_null], rate[is_null], yerr=[rate[is_null] - lo[is_null], hi[is_null] - rate[is_null]],
                    fmt="s", color=SERIES[1], ms=5, lw=1.2, capsize=0, zorder=3)
    ax.set_title(title, fontsize=9, loc="left")
    if ks is not None:
        ax.set_title(f"KS {ks:.3f}", fontsize=8.5, loc="right", color=INK2, fontweight="normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_xlim(-0.6, tab.height - 0.4)
    ax.set_ylim(bottom=0)
    if show_ylabel:
        ax.set_ylabel("target rate (%)")
    axn.bar(xs, tab["share"].to_numpy() * 100, color=C_COUNT, width=0.75)
    axn.set_xticks(xs)
    labels = tab["bin"].to_list()
    rot = 0 if max(len(l) for l in labels) <= 6 and len(labels) <= 8 else 40
    axn.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=7)
    axn.set_xlim(-0.6, tab.height - 0.4)
    axn.grid(False)
    axn.tick_params(axis="y", labelsize=6.5)
    if show_ylabel:
        axn.set_ylabel("% baris", fontsize=7.5)
    return ax


def plot_rates(df, cols, y="target", bins=10, ncols=3, titles=None, width=13.5, row_h=3.3, suptitle=None):
    """Small multiples: target rate per bin (garis + CI 95%), garis putus = rata-rata,
    kotak oranye = bin null, panel bawah = porsi baris per bin."""
    base = df[y].mean()
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(width, row_h * nrows))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.28, hspace=0.62)
    for i, c in enumerate(cols):
        tab = binned(df, c, y, bins)
        ks = ks_stat(df[y].to_numpy(), df[c].cast(pl.String).to_numpy() if df[c].dtype in (pl.String, pl.Categorical, pl.Boolean)
                     else df[c].cast(pl.Float64).to_numpy())
        t = (titles or {}).get(c, c)
        is_cat = df[c].dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean)
        _rate_panel(fig, outer[i // ncols, i % ncols], tab, t, base,
                    show_ylabel=(i % ncols == 0), is_cat=is_cat, ks=ks)
    if suptitle:
        fig.suptitle(suptitle, x=0.06, ha="left", fontsize=12, fontweight="bold", color=INK)
    plt.show()


def plot_oe(df, cols, expected, y="target", bins=10, ncols=3, titles=None, width=13.5, row_h=2.9, suptitle=None):
    """Small multiples O/E: garis 1.0 = tidak ada informasi tambahan di atas `expected`."""
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, row_h * nrows), squeeze=False)
    for i, c in enumerate(cols):
        ax = axes[i // ncols][i % ncols]
        t = binned_oe(df, c, expected, y, bins)
        xs = np.arange(t.height)
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
        oe, lo, hi = (t[k].to_numpy() for k in ("oe", "oe_lo", "oe_hi"))
        if df[c].dtype in (pl.String, pl.Categorical, pl.Enum, pl.Boolean):
            ax.errorbar(xs, oe, yerr=[oe - lo, hi - oe], fmt="o", color=SERIES[2], ms=6, lw=1.5, capsize=0)
            ax.set_xlim(-0.6, t.height - 0.4)
        else:
            nn = (t["bin"] != "null").to_numpy()
            ax.fill_between(xs[nn], lo[nn], hi[nn], color=SERIES[2], alpha=0.15, lw=0)
            ax.plot(xs[nn], oe[nn], color=SERIES[2], marker="o", ms=4.5)
            if (~nn).any():   # bin null: titik terpisah, tidak disambung garis
                ax.errorbar(xs[~nn], oe[~nn], yerr=[oe[~nn] - lo[~nn], hi[~nn] - oe[~nn]], fmt="s", color=SERIES[1], ms=5, lw=1.2)
        ax.set_xticks(xs)
        labels = t["bin"].to_list()
        rot = 0 if max(len(l) for l in labels) <= 6 and len(labels) <= 8 else 40
        ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center", fontsize=7)
        ax.set_title((titles or {}).get(c, c), fontsize=9.5)
        if i % ncols == 0:
            ax.set_ylabel("observed / expected")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, x=0.06, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    plt.show()


def qbin(expr, q):
    """Nomor bin kuantil (0..q-1) berbasis ranking — pengganti `qcut` yang tidak bergantung versi polars
    (qcut deprecated di polars 2.x). Nilai kembar masuk bin yang sama; null tetap null.
    Bisa dipakai sebagai kolom maupun kunci .over(...)."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return (((e.rank("min") - 1) * q) // e.count()).cast(pl.Int32)


def heatmap_rate(df, row, col, y="target", row_bins=None, col_bins=None, min_n=50, title=None, figsize=(8, 4.2)):
    """Target rate (%) per kombinasi dua variabel. Sel dengan n < min_n dikosongkan.
    row_bins / col_bins = jumlah bin kuantil untuk variabel kontinu (label = rentang nilai min–maks bin)."""
    d = df
    labels = {row: {}, col: {}}
    for c, b in ((row, row_bins), (col, col_bins)):
        if b is not None:
            d = d.with_columns(qbin(pl.col(c), b).alias(f"__{c}"))
            for k, lo, hi in d.group_by(f"__{c}").agg(lo=pl.col(c).min(), hi=pl.col(c).max()).drop_nulls(f"__{c}").iter_rows():
                labels[c][k] = fmt_num(lo) if lo == hi else f"{fmt_num(lo)}–{fmt_num(hi)}"
        else:
            d = d.with_columns(pl.col(c).alias(f"__{c}"))
    rk, ck = f"__{row}", f"__{col}"
    g = d.drop_nulls([rk, ck]).group_by(rk, ck).agg(n=pl.len(), rate=pl.col(y).mean())
    rows_ = sorted(g[rk].unique().to_list(), key=_sort_key)
    cols_ = sorted(g[ck].unique().to_list(), key=_sort_key)
    M = np.full((len(rows_), len(cols_)), np.nan)
    N = np.zeros_like(M)
    for r in g.iter_rows(named=True):
        i, j = rows_.index(r[rk]), cols_.index(r[ck])
        N[i, j] = r["n"]
        if r["n"] >= min_n:
            M[i, j] = r["rate"] * 100
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols_)))
    xl = [labels[col].get(v, _short(v)) for v in cols_]
    rot = 30 if max(len(v) for v in xl) > 6 else 0
    ax.set_xticklabels(xl, rotation=rot, ha="right" if rot else "center")
    ax.set_yticks(range(len(rows_)))
    ax.set_yticklabels([labels[row].get(v, _short(v)) for v in rows_])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    vmax = np.nanmax(M)
    for i in range(len(rows_)):
        for j in range(len(cols_)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.1f}%\n(n={int(N[i,j]):,})", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("target rate (%)", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(title or f"Target rate: {row} × {col}")
    plt.show()


def _sort_key(v):
    if v is None:
        return (2, 0, "")
    try:
        s = str(v).strip("([]) ").split(",")[0]
        return (0, float(s), str(v))
    except ValueError:
        return (1, 0, str(v))


def _short(v):
    s = str(v)
    if s.startswith(("(", "[")):
        try:
            a, b = (float(v) for v in s.strip("([])").split(","))
        except ValueError:
            return s
        if np.isinf(a):
            return f"≤{fmt_num(b)}"
        if np.isinf(b):
            return f">{fmt_num(a)}"
        return f"{fmt_num(a)}–{fmt_num(b)}"
    return s


# ---------------------------------------------------------------- baseline model
def cv_lgb(df, features, y="target", cat_features=(), folds=5, seed=42, params=None, return_model=False):
    """LightGBM 5-fold stratified → (oof_pred, ks_oof, auc_oof, importance gain).
    Early stopping memakai fold validasi (sedikit optimistis) — cukup untuk membandingkan sumber data."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    X = df.select(features).to_pandas()
    for c in cat_features:
        X[c] = X[c].astype("category")
    yy = df[y].to_numpy()
    p = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             n_estimators=3000, verbose=-1, random_state=seed)
    p.update(params or {})
    oof = np.zeros(len(yy))
    imp = np.zeros(len(features))
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    import warnings
    for tr_i, va_i in skf.split(X, yy):
        m = lgb.LGBMClassifier(**p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # LightGBM ≥4.6 memberi deprecation warning untuk eval_set
            m.fit(X.iloc[tr_i], yy[tr_i], eval_set=[(X.iloc[va_i], yy[va_i])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_i] = m.predict_proba(X.iloc[va_i])[:, 1]
        imp += m.booster_.feature_importance("gain")
    imp_df = pl.DataFrame({"feature": features, "gain_pct": imp / imp.sum() * 100}).sort("gain_pct", descending=True)
    return oof, ks_stat(yy, oof), auc_stat(yy, oof), imp_df