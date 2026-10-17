"""Density movies and snapshot maps per model (supplementary material, Ecological Modelling).

Usage:  python scripts/figures/density_movies.py [--seed 1] [--fps 12] [--every 2] [--only C1,null] [--nosim]
Simulates one run per model (30x30 grid; seed = SeedSequence([20261017, seed]), as in
the snapshot panel of the paper figure), caches weekly maps in results/movies/cache/*.npz and renders
MP4 movies, PNG snapshots and a comparison mosaic into results/movies/. Does not modify u25/.
"""
from __future__ import annotations
import argparse, os, sys
from multiprocessing import Pool
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "movies")
CACHE = os.path.join(OUT, "cache")
COMP_S = 0.2        # competent = s >= 0.2 (manuscript); non-breeders have s < obs_s_low = 0.2
SEED_BASE = 20261017
WD_COLOR = "#d4006a"      # withdrawn non-breeders
REF_COLOR = "#0088cc"     # retreat sites (refuges)
PH_COL = {"A": "#f1f0ec", "B": "#e2e1db", "C": "#cfcec7", "D": "#b9b8b0"}   # same ramp as the manuscript figures
PH_NAME = {"A": "A establishment", "B": "B growth", "C": "C stagnation", "D": "D decline"}

# key -> (preset, overrides, T, title, output file name or None for the default movie_<key>.mp4)
# C1 and its well-mixed version use the manuscript configuration (em_wm_run.py point 0, em_snapshots.py): the
# preset calhoun_C1 already carries the retreat sites (refuge_frac 0.3, capacity 2, pref 100, strict) which act only
# as capacity when jump_prob = 1 and jump_mode = "uniform". No other override.
MODELS = {
    "C0": ("calhoun_C0", {}, 600, "C0", None),
    "C1p": ("calhoun_C1p", {}, 600, "C1'", "movie_C1p.mp4"),
    "C1": ("calhoun_C1", {}, 600, "C1 (spatial)", "video_S2_C1_spatial.mp4"),
    "C2": ("calhoun_C2", {}, 700, "C2", None),
    "short_lived": ("notebook", dict(a_max=60.0, repro_age_max=50), 600, "Short-lived control", None),
    "long_lived": ("longevo", {}, 600, "Long-lived control (alpha=1.4)", None),
    "null": ("longevo", dict(alpha=4.0), 600, "Null model (long-lived, alpha=4, no rules)", None),
    "C1_wellmixed": ("calhoun_C1", dict(jump_prob=1.0, jump_mode="uniform"), 600, "C1 well-mixed",
                     "video_S3_C1_wellmixed.mp4"),
}
MOSAIC = ["C1", "C1_wellmixed", "short_lived", "null"]
MOSAIC_FILE = "video_S1_comparison.mp4"


def movie_name(key):
    return MODELS[key][4] or f"movie_{key}.mp4"


def simulate_model(args):
    name, seed = args
    from u25.params import preset
    from u25.model import Universe
    pre, ov, T, _, _ = MODELS[name]
    p = preset(pre, T=T, grid_size=30, **ov)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, seed]))
    U = Universe(p, rng=rng)
    G = p.grid_size
    dens, comp, bo = [], [], []

    def grab(U):
        dens.append(U.density_map().astype(np.int16))
        # withdrawn non-breeders (phi_BO of the package, as in em_snapshots.py): adult, s < obs_s_low, isolated
        # (m_i <= obs_iso_m) for >= obs_iso_weeks consecutive weeks. A lone last survivor (N <= 1) is not counted.
        b = (U.age >= p.repro_age_min) & (U.s < p.obs_s_low) & (U.iso_run >= p.obs_iso_weeks)
        if U.N <= 1:
            b = np.zeros_like(b)
        bo.append(np.bincount((U.x * G + U.y)[b], minlength=G * G).reshape(G, G).astype(np.int16))
        c = np.bincount(U.x * G + U.y, weights=(U.s >= COMP_S).astype(float), minlength=G * G).reshape(G, G)
        comp.append(c.astype(np.int16))
    grab(U)
    series, summ = U.run(T, callback=grab)
    refuge = (U.refuge.reshape(G, G) if U.refuge is not None else np.zeros((G, G), bool))
    n = len(dens)
    np.savez_compressed(os.path.join(CACHE, f"{name}.npz"), dens=np.array(dens), comp=np.array(comp), bo=np.array(bo),
                        refuge=refuge, t=series["t"].to_numpy(), N=series["N"].to_numpy(),
                        births=series["births"].to_numpy() if "births" in series else np.zeros(n), T=T)
    return name, int(series["N"].iloc[-1]), int(series["N"].max())


def load(name):
    d = np.load(os.path.join(CACHE, f"{name}.npz"))
    return {k: d[k] for k in d.files}


def NORM(vmax):
    from matplotlib.colors import PowerNorm
    return PowerNorm(0.5, vmin=0, vmax=vmax, clip=True)


def cbar_ticks(vmax):
    return [v for v in (0, 1, 4, 9, 16, 25) if v <= vmax * 0.93]    # no tick near vmax: avoids overlap


def draw_wd(a, wmap, store=None):
    """Withdrawn non-breeders: dot in each cell with >= 1, size ~ sqrt(n)."""
    ys, xs = np.nonzero(wmap)
    sz = 9 + 14 * np.sqrt(wmap[ys, xs]) if len(xs) else []
    if store is None:
        return a.scatter(xs, ys, s=sz, marker="o", c=WD_COLOR, edgecolors="white", linewidths=0.4, zorder=5)
    store.set_offsets(np.c_[xs, ys] if len(xs) else np.empty((0, 2))); store.set_sizes(sz if len(xs) else [])
    return store


def draw_refuges(a, refuge, s=30):
    ys, xs = np.nonzero(refuge)
    if len(xs):
        a.scatter(xs, ys, s=s, marker="s", facecolors="none", edgecolors=REF_COLOR, linewidths=1.1, zorder=4)


def legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker="o", ls="", mfc=WD_COLOR, mec="white", label="cell with withdrawn non-breeders (size ~ n)"),
            Line2D([], [], marker="s", ls="", mfc="none", mec=REF_COLOR, mew=1.4, label="retreat site")]


def phases(d):
    """Operational phase boundaries A|B|C|D, identical to those used in the paper figures
    (manuscript Figs. 1-3): A->B first week with births; B->C growth rate of ln N (centred, 4 weeks) falls below
    half of its maximum (first 60 weeks); C->D start of the plateau, N >= 0.98 max N; D ends at extinction
    (first N < 1). Returns (tA, tBC, tD, tE); NaN when a boundary does not exist."""
    t = np.asarray(d["t"], float); N = np.asarray(d["N"], float); B = np.asarray(d["births"], float)
    nan = np.nan
    iB = np.flatnonzero(B > 0)
    tA = t[iB[0]] if iB.size else nan
    lN = np.log(np.maximum(N, 1e-12))
    r = np.full(N.size, nan); r[2:-2] = (lN[4:] - lN[:-4]) / 4
    tBC = nan
    m = t <= 60
    if np.isfinite(r[m]).any():
        imax = int(np.nanargmax(np.where(m, r, nan)))
        j = np.flatnonzero((np.arange(N.size) > imax) & (r < 0.5 * r[imax]))
        if j.size:
            tBC = t[j[0]]
    iD = int(np.flatnonzero(N >= 0.98 * N.max())[0]); tD = t[iD]
    ext = np.flatnonzero((N < 1) & (t > tD))
    tE = t[ext[0]] if ext.size else nan
    return tA, tBC, tD, tE


def phase_bounds(d, T):
    tA, tBC, tD, tE = phases(d)
    end = tE if np.isfinite(tE) else float(T)
    ts = [0.0, tA, tBC, tD, end]
    out = []
    for k, ph in enumerate("ABCD"):
        lo, hi = ts[k], ts[k + 1]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out.append((ph, lo, hi))
    return out


def t_end(d):
    """Week of extinction (first N = 0) or None if the colony survives the run."""
    z = np.flatnonzero(d["N"] < 1)
    return int(d["t"][z[0]]) if z.size else None


def snapshot_times(d):
    N = d["N"].astype(float)
    iend = int(np.flatnonzero(N > 0)[-1]); imax = int(np.argmax(N)); pk = N[imax]
    ig = int(np.argmax(N >= 0.5 * pk))
    after = np.flatnonzero((np.arange(N.size) > imax) & (N <= 0.5 * pk))
    idec = int(after[0]) if after.size else int(imax + 0.8 * (iend - imax))
    ist = (imax + idec) // 2
    idx = [0, ig, imax, ist, idec, iend]
    names = ["founding", "growth (N = 50% of peak)", "peak", "stagnation (mid peak-decline)", "decline (N = 50% of peak)",
             "last live week"]
    for k in range(1, 6):
        idx[k] = max(idx[k], idx[k - 1])
    return idx, names


def fig_snaps(name, d, vmax):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx, names = snapshot_times(d)
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.6), constrained_layout=True)
    for a, i, nm in zip(ax.ravel(), idx, names):
        im = a.imshow(d["dens"][i], cmap="Greys", norm=NORM(vmax), interpolation="nearest")
        draw_refuges(a, d["refuge"], 18)
        draw_wd(a, d["bo"][i])
        a.set_title(f"{nm}\nweek {d['t'][i]}, N={int(d['N'][i])}, withdrawn={int(d['bo'][i].sum())}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="animals per cell (square-root scale)")
    cb.set_ticks(cbar_ticks(vmax))
    fig.suptitle(f"{MODELS[name][3]} (seed fixed)")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.savefig(os.path.join(OUT, f"snapshots_{name}.png"), dpi=110); plt.close(fig)


def make_movie(name, d, vmax, fps, every):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    N = d["N"]; t = d["t"]; T = int(d["T"])
    te = t_end(d)
    margin = 10
    xmax = (te + margin) if te is not None else T
    last = min(xmax, N.size - 1)
    frames = list(range(0, last + 1, every))
    bounds = phase_bounds(d, T)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1.3], hspace=0.38, wspace=0.12)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    im1 = a1.imshow(d["dens"][0], cmap="Greys", norm=NORM(vmax), interpolation="nearest")
    cf = np.where(d["dens"][0] > 0, d["comp"][0] / np.maximum(d["dens"][0], 1), np.nan)
    cmap2 = plt.get_cmap("viridis").copy(); cmap2.set_bad("0.15")
    im2 = a2.imshow(cf, cmap=cmap2, vmin=0, vmax=1, interpolation="nearest")
    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([]); draw_refuges(a, d["refuge"], 30)
    bsc = [draw_wd(a, d["bo"][0]) for a in (a1, a2)]
    Wt = d["bo"].sum(axis=(1, 2))
    a1.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
    a1.set_title("Density (animals per cell)", fontsize=10)
    a2.set_title(f"Fraction of competent animals (s >= {COMP_S}) per cell; dark grey = empty", fontsize=10)
    cb1 = fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02, label="animals per cell (square-root scale)")
    cb1.set_ticks(cbar_ticks(vmax))
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02, label="competent fraction")
    sel = t <= xmax
    Nmax = N[sel].max()
    a3.plot(t[sel], N[sel], color="0.25", lw=1.2)
    a4 = a3.twinx(); a4.plot(t[sel], Wt[sel], color=WD_COLOR, lw=1.2)
    a4.set_ylabel("Withdrawn non-breeders", color=WD_COLOR)
    a4.set_ylim(0, max(Wt[sel].max(), 1) * 1.15); pb, = a4.plot([0], [Wt[0]], "o", color=WD_COLOR, mec="k")
    for ph, lo, hi in bounds:
        a3.axvspan(lo, min(hi, xmax), color=PH_COL[ph], alpha=0.9, lw=0, zorder=0)
        if min(hi, xmax) - lo > 0.025 * xmax:
            a3.text((lo + min(hi, xmax)) / 2, Nmax * 1.02, ph, ha="center", fontsize=9)
    vl = a3.axvline(0, color="crimson", lw=1.5); pt, = a3.plot([0], [N[0]], "o", color="crimson")
    a3.set_xlim(0, xmax); a3.set_ylim(0, Nmax * 1.15); a3.set_xlabel("Week"); a3.set_ylabel("Population N(t)")
    a3.set_title("Phases: " + "; ".join(PH_NAME[p] for p, _, _ in bounds), fontsize=8, pad=3, loc="left", color="0.3")
    ttl = fig.suptitle("", fontsize=13)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, movie_name(name))
    w = FFMpegWriter(fps=fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, path, 100):
        for i in frames:
            im1.set_data(d["dens"][i])
            im2.set_data(np.where(d["dens"][i] > 0, d["comp"][i] / np.maximum(d["dens"][i], 1), np.nan))
            for sc in bsc: draw_wd(None, d["bo"][i], sc)
            pb.set_data([t[i]], [Wt[i]])
            vl.set_xdata([t[i], t[i]]); pt.set_data([t[i]], [N[i]])
            ttl.set_text(f"{MODELS[name][3]}  |  week {t[i]}  |  N = {int(N[i])}  |  withdrawn = {int(Wt[i])}")
            w.grab_frame()
    plt.close(fig)
    return path


def make_mosaic(names, d_all, vmax, fps, every, fname=MOSAIC_FILE):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    T = 600
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06], wspace=0.22, hspace=0.18, left=0.04, right=0.91, top=0.89, bottom=0.1)
    axs = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    cax = fig.add_subplot(gs[:, 2])
    ims, txt, bs = [], [], []
    for a, n in zip(axs, names):
        d = d_all[n]
        im = a.imshow(d["dens"][0], cmap="Greys", norm=NORM(vmax), interpolation="nearest"); ims.append(im)
        a.set_xticks([]); a.set_yticks([])
        draw_refuges(a, d["refuge"], 22)
        txt.append(a.set_title("", fontsize=10)); bs.append(draw_wd(a, d["bo"][0]))
    cb = fig.colorbar(ims[0], cax=cax, label="animals per cell (square-root scale)")
    cb.set_ticks(cbar_ticks(vmax))
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, fontsize=9, frameon=False)
    ttl = fig.suptitle("", fontsize=13)
    path = os.path.join(OUT, fname)
    w = FFMpegWriter(fps=fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, path, 100):
        for i in range(0, T + 1, every):
            for k, (im, tx, n) in enumerate(zip(ims, txt, names)):
                d = d_all[n]; j = min(i, d["dens"].shape[0] - 1)
                Nj = int(d["N"][j]) if i < d["N"].size else 0
                im.set_data(d["dens"][j] if Nj > 0 else np.zeros_like(d["dens"][0]))
                bj = d["bo"][j] if Nj > 0 else np.zeros_like(d["bo"][0])
                draw_wd(None, bj, bs[k])
                tx.set_text(f"{MODELS[n][3]}  (N={Nj}, withdrawn={int(bj.sum())})")
            ttl.set_text(f"Week {i}")
            w.grab_frame()
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--nosim", action="store_true", help="reuse cache")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    names = [n for n in MODELS if not a.only or n in a.only.split(",")]
    os.makedirs(CACHE, exist_ok=True)
    if not a.nosim:
        with Pool(a.procs) as pool:
            for r in pool.imap_unordered(simulate_model, [(n, a.seed) for n in names]):
                print("sim", r, flush=True)
    d_all = {n: load(n) for n in MODELS if os.path.exists(os.path.join(CACHE, f"{n}.npz"))}
    allv = np.concatenate([d['dens'].ravel() for d in d_all.values()]); vmax = float(np.ceil(np.percentile(allv[allv > 0], 99.5)))
    print("common vmax =", vmax)
    for n in names:
        fig_snaps(n, d_all[n], vmax); make_movie(n, d_all[n], vmax, a.fps, a.every); print("ok", n, flush=True)
    if all(m in d_all for m in MOSAIC) and (not a.only or all(m in names for m in MOSAIC)):
        make_mosaic(MOSAIC, d_all, vmax, a.fps, a.every)


if __name__ == "__main__":
    main()
