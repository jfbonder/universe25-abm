"""Figures of the Ecological Modelling paper (manuscript/em/figs/), regenerated from the deposited data.

    python3 scripts/em/em_figs.py [names]          # from the repository root; or: make em-figs

No simulations are run: every figure is read from results_v2/ (production ensembles of the PRE paper, as in
make figs) and from results_em/ (ensembles and summaries added for the Ecological Modelling paper; see README.md).
The figure sizes and the numbers quoted in the captions are written to results_em/em_figs_numbers.json.

  mechanism    fig_em_mechanism    causal scheme (no data; the paper uses manuscript/em/figs/fig_causal.tex instead)
  trajectory   fig_em_trajectory   reference colony (spatial C1, n = 200) and Calhoun's (1973) censuses, phases A-D
  wellmixed    fig_em_wellmixed    spatial against well mixed (n = 200 each): same sequence and extinction
  snapshots    fig_em_s_snapshots  maps of one run, spatial and well mixed (SM; data: wellmixed_c1/snapshots_seed1.npz)
  behaviour    fig_em_behaviour    classes of non-breeders, failure of care, late cohorts, transplants
  patterns     fig_em_patterns     patterns x model-structure matrix (data: pattern_tables/matrix.csv)
  robustness   fig_em_robustness   background mortality, size, ablations, variants, Sobol', feasible region (SM)
  conditions   fig_em_s_recovery   adult recovery, one panel (SM; not cited in the text)
  recovery     fig_em_recovery     adult-recovery sweep and threshold eps^c(L)
  ablations    fig_em_s_ablations  pattern that fails under each ablation (SM)

Calhoun's censuses: Table S2 of the PRE supplement (manuscript/pre/supplement.tex, tab:calhoun-numbers).
fig_em_s_occupancy.pdf (SM) comes from scripts/em/f10_occupancy.py, and fig_causal.pdf from the TikZ source
manuscript/em/figs/fig_causal.tex (make em-figs builds both).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                  # repository root
for p in (str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "figures"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.lines import Line2D     # noqa: E402

import figs_results as FR               # noqa: E402  (utilidades y carga de datos; no ejecuta figuras)
from pipeline import survival           # noqa: E402

OUT = ROOT / "manuscript" / "em" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
EM = ROOT / "results_em"
NUMS_OUT = EM / "em_figs_numbers.json"
NUM = json.loads((ROOT / "results_v2" / "results_numbers.json").read_text())
RES2 = ROOT / "results_v2"
WMDIR = EM / "wellmixed_c1"
K_GAMMA = 8 * 900          # m_c |G| (L = 30)
G_CELLS = 900

# ----------------------------------------------------------------------------- estilo (Elsevier)
W1, W15, W2 = 3.54, 5.51, 7.48      # 90, 140, 190 mm
CAT = FR.CAT                        # paleta categórica validada (dataviz): orden fijo
INK, INK2, MUTED, GRID = FR.INK, FR.INK2, FR.MUTED, FR.GRID
C_SP, C_WM, C_WMNR, C_C1P, C_LL, C_SL = CAT[0], CAT[1], CAT[4], CAT[2], CAT[5], CAT[6]
C_CAL = INK                         # datos de Calhoun
PH_COL = {"A": "#f1f0ec", "B": "#e2e1db", "C": "#cfcec7", "D": "#b9b8b0"}   # fases: rampa neutra ordenada

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold", "mathtext.fallback": "cm",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "axes.titlelocation": "left",
})

NUMS: dict = {}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w_mm, h_mm = (bb.width + 0.06) * 25.4, (bb.height + 0.06) * 25.4
    NUMS.setdefault("_sizes_mm", {})[name] = [round(w_mm, 1), round(h_mm, 1)]
    plt.close(fig)
    warn = "  <-- más ancho que 190 mm" if w_mm > 190.5 else ""
    print("fig", OUT / f"{name}.pdf", f"{w_mm:.0f} x {h_mm:.0f} mm{warn}")


def panel(ax, letter, title=None, pad=None):
    t = title if title is not None else ax.get_title(loc="left")
    kw = {} if pad is None else dict(pad=pad)
    ax.set_title(f"({letter}) {t}" if t else f"({letter})", loc="left", color=INK, fontsize=8, **kw)


def wilson(k, n):
    return survival.wilson(int(k), int(n))


def xe(f, lo, hi):
    return FR.xe(f, lo, hi)


# ----------------------------------------------------------------------------- datos de Calhoun (Tabla S2 de PRE)
# (día, N, etiqueta); semana = día / 7. La extinción del día 1780 es una proyección de Calhoun.
CALHOUN_N = [(0, 8, "4 pairs introduced"), (315, 620, "end of B"), (560, 2200, "peak"), (736, 2056, ""),
             (1444, 122, ""), (1588, 27, "27 survivors")]
CALHOUN_PHASES = [("A", 0, 104), ("B", 104, 315), ("C", 315, 560), ("D", 560, 1780)]   # días
CALHOUN_EVENTS = {"last surviving young": 600, "last conception": 920, "projected extinction": 1780}
CALHOUN_AGE = (1331, 776)        # 1 Mar 1972 (día 1331): edad media de los sobrevivientes 776 días


# ----------------------------------------------------------------------------- series y fases
SER_COLS = ["point", "rep", "t", "N", "births", "pups", "conceptions", "S_adult", "mean_age", "N_adult",
            "frac_withdrawn", "frac_crowded_low", "phi_BO", "low_s_frac", "phi_w", "coh_f_born",
            "coh_f_ever_conceived"]
LBAR = 6.55     # tamaño medio de camada (litter_sizes · litter_probs)


def load_series(path, point, cols=SER_COLS):
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    s = pd.read_parquet(path, columns=[c for c in cols if c in have])
    return s[s.point == point].copy()


DEP = EM / "series_deposit"     # reduced series (em_f9_analysis.py deposit)


def base_series(point=0):
    """C1 espacial (f2b_base). Se lee la serie reducida versionada si existe (mismas filas, columnas de SER_COLS,
    float32); si no, el series.parquet completo (no versionado)."""
    dep = DEP / "em_series_C1_spatial.parquet"
    return load_series(dep if dep.exists() else RES2 / "f2b_base" / "series.parquet", point)


def wm_series(point=0):
    dep = DEP / "em_series_C1_wellmixed.parquet"
    return load_series(dep if dep.exists() else WMDIR / "series.parquet", point)


def kinetic_traj():
    k = pd.read_csv(EM / "kinetic" / "traj_C1_exact.csv")
    k0 = pd.DataFrame(dict(t=[0], nu=[k.nu_pre.iloc[0]], births=[0.0], S=[1.0]))
    return pd.concat([k0, k[["t", "nu", "births", "S"]]], ignore_index=True)


def phase_times(t, N, B):
    """Fronteras operativas de las fases de Calhoun en una corrida (o en la órbita fluida):
    A -> B: primera semana con nacimientos (t_A); B -> C: la tasa de crecimiento de ln N (diferencia centrada
    a 4 semanas) cae por debajo de la mitad de su máximo (Calhoun: el tiempo de duplicación pasa de ~55 a ~145
    días); C -> D: comienzo de la meseta, N >= 0.98 max N (t'_pk, "population increase abruptly ceased");
    fin de D: extinción (primer N < 1; en la órbita fluida, |G| nu < 1/2)."""
    t = np.asarray(t, float); N = np.asarray(N, float); B = np.asarray(B, float)
    tA = t[np.flatnonzero(B > 0)[0]]
    lN = np.log(np.maximum(N, 1e-12))
    r = np.full(N.size, np.nan)
    r[2:-2] = (lN[4:] - lN[:-4]) / 4
    m = t <= 60
    imax = int(np.nanargmax(np.where(m, r, np.nan)))
    j = np.flatnonzero((np.arange(N.size) > imax) & (r < 0.5 * r[imax]))
    tBC = t[j[0]]
    iD = int(np.flatnonzero(N >= 0.98 * N.max())[0])
    tD = t[iD]
    ext = np.flatnonzero((N < 1) & (t > tD))
    tE = t[ext[0]] if ext.size else np.nan
    iB = np.flatnonzero(B > 0)
    tlast = t[iB[-1]] if iB.size else np.nan
    iBC = int(np.searchsorted(t, tBC))
    rB = (lN[iBC] - lN[int(np.searchsorted(t, tA))]) / max(tBC - tA, 1)
    rC = (lN[iD] - lN[iBC]) / max(tD - tBC, 1)
    return dict(tA=tA, tBC=tBC, tD=tD, tE=tE, tlast=tlast, dB=7 * np.log(2) / rB, dC=7 * np.log(2) / rC)


def ens_phases(S):
    rows = []
    for r, g in S.groupby("rep"):
        g = g.sort_values("t")
        rows.append(dict(rep=r, **phase_times(g.t, g.N, g.births)))
    return pd.DataFrame(rows)


def qband(S, col, tmax, alive_only=False, scale=1.0, min_alive=0.5):
    g = S[S.t <= tmax]
    if alive_only:
        nrep = g.rep.nunique()
        gl = g[g.N > 0]
        q = gl.groupby("t")[col].quantile([0.1, 0.5, 0.9]).unstack()
        alive = gl.groupby("t").rep.nunique() / nrep
        q = q.reindex(sorted(g.t.unique()))
        alive = alive.reindex(q.index).fillna(0)
        q.loc[alive < min_alive] = np.nan
    else:
        q = g.groupby("t")[col].quantile([0.1, 0.5, 0.9]).unstack()
    return q.index.to_numpy(float), q[0.1].to_numpy() * scale, q[0.5].to_numpy() * scale, q[0.9].to_numpy() * scale


def ribbon(ax, y, h, bounds, label, end=None, fs=7, label_x=None):
    """Cinta de fases A-D en coordenadas de datos x, y en coordenadas de ejes."""
    tr = ax.get_xaxis_transform()
    for ph, a, b in bounds:
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        ax.add_patch(Rectangle((a, y), b - a, h, transform=tr, facecolor=PH_COL[ph], edgecolor="white", lw=0.8,
                               clip_on=False, zorder=3))
        if b - a > 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]):
            ax.text((a + b) / 2, y + h / 2, ph, transform=tr, ha="center", va="center", fontsize=fs, color=INK,
                    zorder=4, clip_on=False)
    if label:
        lx = label_x if label_x is not None else ax.get_xlim()[1]
        ax.text(lx, y + h / 2, " " + label, transform=tr, ha="left", va="center", fontsize=fs, color=INK2,
                clip_on=False)


def med(D, c):
    return float(np.nanmedian(D[c]))


def bounds_from(P):
    """Medianas de las fronteras de un ensamble -> lista (fase, inicio, fin)."""
    return [("A", 0, med(P, "tA")), ("B", med(P, "tA"), med(P, "tBC")), ("C", med(P, "tBC"), med(P, "tD")),
            ("D", med(P, "tD"), med(P, "tE"))]


def vphases(ax, bnd, letters=True, y=0.985, xmax=None):
    x1 = xmax if xmax is not None else ax.get_xlim()[1]
    for ph, a, b in bnd:
        if ph != "A" and a < x1:
            ax.axvline(a, color=MUTED, lw=0.6, ls=":", zorder=1)
        if letters and np.isfinite(a) and np.isfinite(b) and ph != "A":
            bb = min(b, x1)
            if bb - a > 0.04 * x1:
                ax.text((a + bb) / 2, y, ph, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=7, color=INK2)


def ribbon_axis(ax, rows, xlim, fs=7):
    """Eje de cintas de fases: rows = [(etiqueta, bounds, marcas)], arriba hacia abajo."""
    ax.set_xlim(*xlim); ax.set_ylim(0, len(rows)); ax.axis("off")
    for i, (label, bnd, marks) in enumerate(rows):
        y = len(rows) - 1 - i + 0.12
        for ph, a, b in bnd:
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            ax.add_patch(Rectangle((a, y), b - a, 0.76, facecolor=PH_COL[ph], edgecolor="white", lw=0.8))
            if b - a > 0.025 * (xlim[1] - xlim[0]):
                ax.text((a + b) / 2, y + 0.38, ph, ha="center", va="center", fontsize=fs, color=INK)
        ax.text(xlim[0] - 0.008 * (xlim[1] - xlim[0]), y + 0.38, label, ha="right", va="center", fontsize=fs,
                color=INK2)
        for x, mk, col in marks:
            ax.plot([x], [y + 0.88], marker=mk, ms=3.6, color=col, clip_on=False, zorder=5, ls="none")


# ============================================================================= 1. esquema causal
def fig_mechanism():
    """Esquema causal: cadena en serpentina (fila 1 de izquierda a derecha, fila 2 de derecha a izquierda), con
    cada regla escrita sobre la flecha en la que actúa, la realimentación de densidad (punteada) y la rama
    espacial (retiro y nidos vacíos), que es la única parte que necesita espacio."""
    fig, ax = plt.subplots(figsize=(W2, 2.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 32.6); ax.axis("off")
    bw, bh = 18.4, 7.6
    X = [0.6, 31.0, 61.0]
    Y1, Y2 = 21.0, 2.6

    def node(x, y, text, phase, ec=INK, fc="white", tc=INK, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec,
                                    lw=lw))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=7.2, color=tc, linespacing=1.15)
        if phase:
            ax.text(x + bw - 0.5, y + 0.5, phase, ha="right", va="bottom", fontsize=7, color=INK2,
                    fontweight="bold")

    def arr(p, q, ls="-", col=INK2, rad=0.0, lw=1.0):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, shrinkA=2,
                                                        shrinkB=2, mutation_scale=8,
                                                        connectionstyle=f"arc3,rad={rad}"))

    def lab(x, y, text, col=C_SP, ha="center", va="center"):
        ax.text(x, y, text, ha=ha, va=va, fontsize=7, color=col, style="italic", linespacing=1.15)

    node(X[0], Y1, "Competent adults\nbreed: the colony\ngrows", "B")
    node(X[1], Y1, "Local density\nexceeds $m_c$\nanimals per cell", "B→C")
    node(X[2], Y1, "Adults lose\nreproductive\ncompetence $s$", "C")
    node(X[2], Y2, "Young raised among\nincompetent adults\nstay incompetent", "C")
    node(X[1], Y2, "Births stop near\nthe peak", "D")
    node(X[0], Y2, "A sterile, ageing\ncolony dies out;\nno recovery", "D")
    # rama espacial
    xs = 87.0
    ax.add_patch(FancyBboxPatch((xs, Y1 - 1.0), 12.6, bh + 2.0, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc="#fbfbf9", ec=MUTED, lw=0.7, ls="--"))
    ax.text(xs + 6.3, Y1 + bh / 2, "Withdrawn class\nand empty nests\n(spatial variant\nonly)", ha="center",
            va="center", fontsize=7, color=INK2, linespacing=1.1)
    # cadena
    yc1, yc2 = Y1 + bh / 2, Y2 + bh / 2
    arr((X[0] + bw, yc1), (X[1], yc1)); lab((X[0] + bw + X[1]) / 2, yc1 + 2.4, "growth", col=INK2)
    arr((X[1] + bw, yc1), (X[2], yc1)); lab((X[1] + bw + X[2]) / 2, yc1 + 2.9, "crowding\ndamage")
    arr((X[2] + bw / 2, Y1), (X[2] + bw / 2, Y2 + bh))
    lab(X[2] + bw / 2 + 1.0, (Y1 + Y2 + bh) / 2, "R2: young learn competence\nfrom neighbours; pups inherit\n"
        "a fraction $w$ of the mother's $s$", ha="left")
    arr((X[2], yc2), (X[1] + bw, yc2)); lab((X[1] + bw + X[2]) / 2, yc2 + 3.2, "fecundity and\npup survival\n"
                                                                              "fall with $s$", col=INK2)
    arr((X[1], yc2), (X[0] + bw, yc2)); lab((X[0] + bw + X[1]) / 2, yc2 + 3.0, "R1: no recovery\nafter the window")
    # realimentación: los adultos estériles mantienen la densidad
    arr((X[0] + bw * 0.3, Y2 + bh), (X[1] + bw * 0.3, Y1), ls=(0, (3, 2)), col=MUTED, rad=-0.3)
    ax.text(X[0] + bw * 0.74, (Y1 + Y2 + bh) / 2 - 0.6, "long-lived sterile adults\nkeep the density high",
            ha="left", va="center", fontsize=7, color=INK2, style="italic", linespacing=1.1)
    # rama espacial
    arr((X[2] + bw, yc1), (xs, yc1), ls=(0, (3, 2)), col=MUTED)
    lab((X[2] + bw + xs) / 2 + 0.1, yc1 - 3.9, "aversion,\nretreat\nsites", col=INK2)
    ax.text(0, 32.5, "Rules (blue) act on individuals, every week, without any resource limit; phase letters "
            "follow Calhoun (A adjustment, B growth, C stagnation, D death).", fontsize=7, color=INK2, va="top")
    save(fig, "fig_em_mechanism")


# ============================================================================= 2. trayectoria y Calhoun
def _norm_band(S, P, col, xg, scale_col=None, right=0.0):
    """Bandas 10-50-90 % de col (opcionalmente dividida por su máximo por corrida) en el tiempo normalizado
    x = t / t_D de cada corrida (t_D = comienzo de la meseta, frontera C -> D)."""
    tD = P.set_index("rep").tD
    Y = []
    for r, g in S.groupby("rep"):
        g = g.sort_values("t")
        y = g[col].to_numpy(float)
        if scale_col == "max":
            y = y / np.nanmax(y)
        Y.append(np.interp(xg, g.t.to_numpy(float) / tD[r], y, right=right))
    Y = np.array(Y)
    return np.nanquantile(Y, 0.1, axis=0), np.nanquantile(Y, 0.5, axis=0), np.nanquantile(Y, 0.9, axis=0), Y


def fig_trajectory():
    """Fig. 2 (Fase 9a, B14): eje de tiempo normalizado por el instante del pico de cada sistema (t / t_pk, con
    t_pk = frontera C -> D: comienzo de la meseta en el modelo, día 560 en Universe 25), para comparar formas y
    no sugerir un ajuste temporal. (b) y (c) son del modelo solo, en semanas."""
    S = base_series(0)
    P = ens_phases(S)
    bnd = bounds_from(P)
    NUMS["trajectory"] = {k: dict(median=med(P, k), p10=float(P[k].quantile(0.1)), p90=float(P[k].quantile(0.9)))
                          for k in ("tA", "tBC", "tD", "tE", "tlast", "dB", "dC")}
    NUMS["trajectory"]["n"] = int(P.shape[0])
    Pn = P.assign(**{k: P[k] / P.tD for k in ("tA", "tBC", "tE", "tlast")}).assign(tD=1.0)
    bndn = bounds_from(Pn)
    NUMS["trajectory"]["normalized"] = {k: float(np.nanmedian(Pn[k])) for k in ("tA", "tBC", "tE", "tlast")}
    TPK_CAL = 560.0
    xmax = 5.6
    fig = plt.figure(figsize=(W2, 5.2))
    gs0 = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.62)
    gt = gs0[0].subgridspec(2, 1, height_ratios=[0.30, 1.30], hspace=0.06)
    gs = gs0[1].subgridspec(1, 3, wspace=0.34)
    rb = fig.add_subplot(gt[0])
    cb = [(ph, d0 / TPK_CAL, d1 / TPK_CAL) for ph, d0, d1 in CALHOUN_PHASES]
    ribbon_axis(rb, [("Universe 25", cb, [(CALHOUN_EVENTS["last surviving young"] / TPK_CAL, "v", C_CAL),
                                          (CALHOUN_EVENTS["last conception"] / TPK_CAL, "v", C_CAL)]),
                     ("model (C1)", bndn, [(NUMS["trajectory"]["normalized"]["tlast"], "v", C_SP)])], (0, xmax),
                fs=7)
    rb.set_title("(a) shape of the trajectory: time relative to the peak of each system", loc="left",
                 fontsize=8, color=INK, pad=2)
    a = fig.add_subplot(gt[1])
    xg = np.linspace(0, xmax, 561)
    lo, md, hi, Y = _norm_band(S, P, "N", xg, "max")
    a.fill_between(xg, lo, hi, color=C_SP, alpha=0.18, lw=0)
    for k in range(10):
        a.plot(xg, Y[k], color=C_SP, lw=0.35, alpha=0.5)
    a.plot(xg, md, color=C_SP, lw=1.7, label="model (C1): median, 10–90% band, 10 of 200 runs")
    cw = np.array([d for d, _, _ in CALHOUN_N]) / TPK_CAL
    cn = np.array([n for _, n, _ in CALHOUN_N]) / 2200
    a.plot(cw, cn, "s--", color=C_CAL, ms=4.4, lw=0.9, mfc="white", mew=1.0,
           label="Universe 25 census (Calhoun, 1973)")
    ext_w = CALHOUN_EVENTS["projected extinction"] / TPK_CAL
    a.plot([cw[-1], ext_w], [cn[-1], 0], ls=":", color=C_CAL, lw=0.9)
    a.plot([ext_w], [0], "x", color=C_CAL, ms=4.5, mew=1.0, clip_on=False, ls="none",
           label="Universe 25: projected extinction (day 1780)")
    a.text(1588 / TPK_CAL - 0.06, cn[-1] + 0.05, "27 old survivors", fontsize=7, color=INK2, ha="right",
           va="bottom")
    a.text(1.04, 1.02, "peak, $N=2200$", fontsize=7, color=INK2, va="bottom")
    a.set_ylim(-0.02, 1.12); a.set_xlim(0, xmax)
    a.set_ylabel("population / its maximum")
    a.set_xlabel("time / time of the peak ($t/t_{\\rm pk}$; model: start of the plateau, median 34 weeks; "
                 "Universe 25: day 560)")
    a.legend(loc="upper center", fontsize=7, handlelength=2.4, bbox_to_anchor=(0.5, -0.2), ncol=3,
             columnspacing=1.2)
    tl = med(P, "tlast")
    # (b) nacimientos
    b = fig.add_subplot(gs[0])
    t, lo, md, hi = qband(S, "births", 120)
    b.fill_between(t, lo, hi, color=C_SP, alpha=0.18, lw=0)
    b.plot(t, md, color=C_SP, label="surviving births")
    t2, _, mp, _ = qband(S, "pups", 120)
    b.plot(t2, mp, color=C_SP, ls="--", lw=1.0, label="pups born")
    b.set_xlim(0, 80); b.set_ylim(0, None)
    vphases(b, bnd)
    b.set_xlabel("week"); b.set_ylabel("per week")
    b.legend(loc="center right", fontsize=7)
    panel(b, "b", "births stop at the peak")
    # (c) estado social adulto y fracción deteriorada
    c = fig.add_subplot(gs[1])
    t, lo, md, hi = qband(S, "S_adult", 120, alive_only=True)
    c.fill_between(t, lo, hi, color=C_SP, alpha=0.18, lw=0)
    c.plot(t, md, color=C_SP, label="adults: mean $s$")
    t, lo, md, hi = qband(S, "low_s_frac", 120, alive_only=True)
    c.plot(t, md, color=INK2, ls="--", lw=1.0, label="share with $s<0.2$")
    c.set_ylim(0, 1.02); c.set_xlim(0, 80)
    vphases(c, bnd)
    c.set_xlabel("week"); c.set_ylabel("competence, share")
    c.legend(loc="center right", fontsize=7, bbox_to_anchor=(1.02, 0.5))
    panel(c, "c", "competence falls before the peak")
    # (d) edad media y Calhoun
    d = fig.add_subplot(gs[2])
    Sa = S[S.N > 0]
    lo, md_, hi, _ = _norm_band(Sa, P, "mean_age", xg, right=np.nan)
    alive = np.array([(np.interp(xg, g.t.to_numpy(float) / P.set_index("rep").tD[r], (g.N > 0).to_numpy(float),
                                 right=0.0)) for r, g in S.sort_values("t").groupby("rep")]).mean(0)
    ok = alive >= 0.5
    d.fill_between(xg[ok], lo[ok], hi[ok], color=C_SP, alpha=0.18, lw=0)
    d.plot(xg[ok], md_[ok], color=C_SP, label="model: mean age")
    d.plot([CALHOUN_AGE[0] / TPK_CAL], [CALHOUN_AGE[1] / 7], "s", ms=4.5, mfc="white", mec=C_CAL, mew=1.0,
           label="Universe 25: mean age (day 1331)")
    d.plot([1588 / TPK_CAL], [987 / 7], "^", ms=4.5, mfc="white", mec=C_CAL, mew=1.0,
           label="Universe 25: 27 alive, all older")
    d.set_xlim(0, xmax); d.set_ylim(0, 250)
    d.set_xlabel("$t/t_{\\rm pk}$"); d.set_ylabel("age (weeks)")
    d.legend(loc="upper left", fontsize=7, bbox_to_anchor=(-0.02, 1.03), handlelength=1.5)
    panel(d, "d", "the colony ages after births stop")
    save(fig, "fig_em_trajectory")


# ============================================================================= 3. espacial frente a bien mezclada
def km_curve(ev, t):
    km = survival.kaplan_meier(np.asarray(t, float), np.asarray(ev, bool))
    return km


def fig_wellmixed():
    """Espacial (C1, n = 200) frente a bien mezclada (C1 con saltos uniformes, n = 200; con y sin refugios) y la
    ecuación cinética (límite fluido de la variante bien mezclada sin refugios, Fase 6)."""
    Ssp, Swm, Swn = base_series(0), wm_series(0), wm_series(1)
    kin = kinetic_traj()
    Psp, Pwm, Pwn = ens_phases(Ssp), ens_phases(Swm), ens_phases(Swn)
    Pk = phase_times(kin.t, kin.nu * G_CELLS, kin.births)
    # T_fl: |G| nu < 1/2
    ik = np.flatnonzero((kin.nu * G_CELLS < 0.5) & (kin.t > Pk["tD"]))
    Pk["tE"] = float(kin.t.iloc[ik[0]]) if ik.size else np.nan
    NUMS["wellmixed"] = {nm: {k: dict(median=med(P, k), p10=float(P[k].quantile(0.1)),
                                      p90=float(P[k].quantile(0.9))) for k in ("tA", "tBC", "tD", "tE", "tlast",
                                                                               "dB", "dC")}
                         for nm, P in (("spatial", Psp), ("wellmixed", Pwm), ("wellmixed_norefuges", Pwn))}
    NUMS["wellmixed"]["kinetic"] = {k: float(v) for k, v in Pk.items()}
    series = [("spatial", Ssp, C_SP, "-"), ("well-mixed", Swm, C_WM, "-"), ("well-mixed, no retreat sites", Swn, C_WMNR,
                                                                              (0, (4, 1.5)))]
    # Fase 10b (recorte aprobado por el autor): la fila (f) de mapas pasa al SM (fig_snapshots)
    fig = plt.figure(figsize=(W2, 4.45))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.36, top=0.93)
    # (a) densidad
    a = fig.add_subplot(gs[0, 0])
    for nm, S, col, ls in series:
        t, lo, md_, hi = qband(S, "N", 210, scale=1 / G_CELLS)
        if ls == "-":
            a.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        a.plot(t, md_, color=col, ls=ls, lw=1.4 if ls == "-" else 1.1, label=nm)
    a.axhline(8, color=MUTED, lw=0.6, ls=":"); a.text(208, 8.15, "$m_c$", fontsize=7, color=INK2, ha="right",
                                                      va="bottom")
    a.set_xlim(0, 210); a.set_ylim(0, 9.2)
    a.set_xlabel("week"); a.set_ylabel("animals per cell")
    fig.legend(*a.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3,
               fontsize=7, handlelength=2.4, columnspacing=1.6)
    panel(a, "a", "population")
    # (b) nacimientos
    b = fig.add_subplot(gs[0, 1])
    for nm, S, col, ls in series:
        t, lo, md_, hi = qband(S, "births", 80, scale=1 / G_CELLS)
        if ls == "-":
            b.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        b.plot(t, md_, color=col, ls=ls, lw=1.4 if ls == "-" else 1.1)
    b.set_xlim(0, 80); b.set_ylim(0, None)
    b.set_xlabel("week"); b.set_ylabel("births per cell and week")
    panel(b, "b", "births")
    # (c) estado social adulto
    c = fig.add_subplot(gs[0, 2])
    for nm, S, col, ls in series:
        t, lo, md_, hi = qband(S, "S_adult", 80, alive_only=True)
        if ls == "-":
            c.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        c.plot(t, md_, color=col, ls=ls, lw=1.4 if ls == "-" else 1.1)
    c.set_xlim(0, 80); c.set_ylim(0, 1.02)
    c.set_xlabel("week"); c.set_ylabel("mean competence of adults")
    panel(c, "c", "loss of competence")
    # (d) cronología de fases
    d = fig.add_subplot(gs[1, 0:2])
    pos = d.get_position()
    d.set_position([pos.x0 + 0.085, pos.y0, pos.width - 0.085, pos.height])
    rows = [("Universe 25", [(ph, d0 / 7, d1 / 7) for ph, d0, d1 in CALHOUN_PHASES], None, C_CAL),
            ("spatial", bounds_from(Psp), Psp, C_SP), ("well-mixed", bounds_from(Pwm), Pwm, C_WM),
            ("well-mixed,\nno retreat sites", bounds_from(Pwn), Pwn, C_WMNR)]
    nR = len(rows)
    for i, (lab, bnd, P, col) in enumerate(rows):
        y = nR - 1 - i
        for ph, x0, x1 in bnd:
            d.add_patch(Rectangle((x0, y - 0.32), x1 - x0, 0.64, facecolor=PH_COL[ph], edgecolor="white", lw=0.8))
            if x1 - x0 > 7:
                shortC = any(p_ == "C" and b_ - a_ <= 7 for p_, a_, b_ in bnd)
                xm = x0 + (0.7 if (ph == "D" and shortC) else 0.5) * (x1 - x0)
                d.text(xm, y, ph, ha="center", va="center", fontsize=7, color=INK)
            elif ph == "C" and x1 - x0 > 0:
                d.text(x1 + 1.5, y, f"< C, {x1 - x0:.0f} weeks", ha="left", va="center", fontsize=7, color=INK)
        if P is not None:
            for k in ("tBC", "tD", "tE"):
                lo_, hi_ = P[k].quantile(0.1), P[k].quantile(0.9)
                d.plot([lo_, hi_], [y - 0.43, y - 0.43], color=col, lw=1.8, solid_capstyle="butt")
            tl = med(P, "tlast")
            d.plot([tl], [y + 0.43], "v", ms=3.6, color=col)
        d.plot([-3], [y], "s", ms=4.5, color=col, clip_on=False)
        d.text(-7, y, lab, ha="right", va="center", fontsize=7, color=INK)
    d.plot([CALHOUN_EVENTS["last surviving young"] / 7], [nR - 1 + 0.43], "v", ms=3.6, color=C_CAL)
    d.set_yticks([])
    d.set_ylim(-0.7, nR - 0.4); d.set_xlim(-6, 262)
    d.set_xlabel("weeks since colonization")
    d.grid(axis="y", visible=False)
    d.spines["left"].set_visible(False); d.tick_params(axis="y", length=0)
    d.text(262, -0.66, "bars: 10–90% of colonies (weekly resolution);  triangles: last birth", fontsize=7,
           color=INK2, ha="right", va="bottom")
    panel(d, "d", "phase sequence: without space phase C is compressed")
    # (e) supervivencia: C1 y control de vida corta en el espacio, b.m. con apareamiento en la celda y b.m. con
    # apareamiento global (Fase 9a, A1)
    e = fig.add_subplot(gs[1, 2])
    NUMS["wellmixed"]["km"] = {}
    for nm, (tt, T, col, ls, lw) in survival_groups().items():
        km = _km(tt, T)
        e.step(km.t, km.S, where="post", color=col, ls=ls, lw=lw)
        if ls == "-":
            e.fill_between(km.t, km.lo, km.hi, step="post", color=col, alpha=0.15, lw=0)
        ev = np.isfinite(tt)
        m = survival.km_median(km)
        NUMS["wellmixed"]["km"][nm] = dict(n=int(tt.size), extinct=int(ev.sum()), T=T, median=float(m["median"]),
                                           lo=float(m["lo"]), hi=float(m["hi"]))
    e.set_xlim(0, 1000); e.set_ylim(-0.02, 1.02)
    e.text(1000, 1.0, "$L=60$", fontsize=7, color=CAT[2], ha="right", va="top")
    e.set_xlabel("week"); e.set_ylabel("P(colony alive)")
    e.legend([Line2D([], [], color=c_, lw=1.3) for c_ in (C_SP, C_WM, CAT[2])] +
             [Line2D([], [], color=INK2, lw=1.3), Line2D([], [], color=INK2, lw=1.3, ls=(0, (1, 1.2)))],
             ["space", "w.m.,\ncell mating", "w.m.,\nglobal mating", "C1", "short-lived\ncontrol"],
             loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=7, handlelength=1.5, labelspacing=0.5)
    panel(e, "e", "extinction and its control")
    save(fig, "fig_em_wellmixed")


def fig_snapshots():
    """(SM, Fase 10b) la antigua Fig. 4f del cuerpo: mapas de una corrida, espacial y bien mezclada."""
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 1, top=0.80)
    snapshot_row(fig, gs[0, 0])
    save(fig, "fig_em_s_snapshots")


SNAP = WMDIR / "snapshots_seed1.npz"


def snapshot_row(fig, spec):
    """(f) mapas de una corrida (em_snapshots.py): ocupación por celda (gris, raíz común) y adultos retirados
    (s < 0.2 y aislados >= 4 semanas; puntos, área proporcional a n) en el pico de N y en el declive a la mitad."""
    if not SNAP.exists():
        print("sin", SNAP)
        return
    Z = np.load(SNAP)
    G = int(Z["G"])
    gsr = spec.subgridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.06], wspace=0.12)
    maps = []
    for nm in ("sp", "wm"):
        N = Z[f"{nm}_N"]
        ipk = int(np.argmax(N))
        ihalf = ipk + int(np.flatnonzero(N[ipk:] <= 0.5 * N.max())[0])
        for i, tag in ((ipk, "peak"), (ihalf, "half-peak decline")):
            maps.append((nm, i + 1, tag, Z[f"{nm}_occ"][i].reshape(G, G), Z[f"{nm}_wd"][i].reshape(G, G),
                         Z[f"{nm}_ref"].reshape(G, G), int(N[i])))
    occ_all = np.concatenate([m[3].ravel() for m in maps])
    vmax = float(np.quantile(occ_all[occ_all > 0], 0.995))
    from matplotlib.colors import PowerNorm, LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("grey", ["#ffffff", "#d3d2cc", "#8a8984", "#3d3c39", "#0b0b0b"])
    norm = PowerNorm(0.5, vmin=0, vmax=vmax)
    NUMS.setdefault("wellmixed", {})["snapshots"] = []
    for k, (nm, wk, tag, O, W, R, Nt) in enumerate(maps):
        ax = fig.add_subplot(gsr[k])
        im = ax.imshow(O.T, origin="lower", cmap=cm, norm=norm, interpolation="nearest")
        yy, xx = np.nonzero(W.T)
        ax.scatter(xx, yy, s=4.0 * np.sqrt(W.T[yy, xx]), color=CAT[6], lw=0, zorder=3)
        ry, rx = np.nonzero((R.T) & (O.T == 0))
        for x_, y_ in zip(rx, ry):
            ax.add_patch(Rectangle((x_ - 0.5, y_ - 0.5), 1, 1, fill=False, ec=CAT[2], lw=0.5, zorder=2))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_color(MUTED); s_.set_linewidth(0.6)
        f0 = float((O == 0).mean()); nwd = int(W.sum())
        ax.set_title(f"{'spatial' if nm == 'sp' else 'well-mixed'}, {tag}\nweek {wk}, $N={Nt}$", fontsize=7,
                     loc="center", color=INK, pad=2)
        ax.text(0.5, -0.03, f"empty cells {100 * f0:.0f}%;  withdrawn {nwd}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=INK2)
        NUMS["wellmixed"]["snapshots"].append(dict(model=nm, week=wk, phase=tag, N=Nt, empty=f0, withdrawn=nwd))
        if k == 0:
            ax.text(-0.04, 1.2, "one colony (seed 1): occupancy and withdrawn adults", transform=ax.transAxes,
                    fontsize=8, color=INK, ha="left", va="bottom")
    cax = fig.add_subplot(gsr[4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 1, 4, 9, 16][: 1 + sum(v <= vmax for v in (1, 4, 9, 16))])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    cb.set_label("animals per cell", fontsize=7)


def _km(tt, T):
    tt = np.asarray(tt, float)
    ev = np.isfinite(tt)
    t = np.where(ev, tt, T)
    km = survival.kaplan_meier(t, ev)
    if t.max() > km.t.max():
        last = km.iloc[-1].copy(); last["t"] = t.max()
        km = pd.concat([km, last.to_frame().T], ignore_index=True)
    return km


def survival_groups():
    """{nombre: (t_ext con NaN si sobrevive, T, color, estilo, ancho)}; L = 30 salvo indicación."""
    base = pd.read_csv(RES2 / "f2b_base" / "summary.csv")
    sp1k = pd.read_csv(EM / "spatial_reversible_null_T1000" / "summary.csv")
    wm = _wm_summary()
    eco = pd.read_csv(EM / "wellmixed_cell_reference" / "per_run.csv")
    gl = pd.read_csv(EM / "wellmixed_global_mating" / "per_run.csv")
    dot = (0, (1, 1.2))
    g = lambda c, L: gl[(gl.cfg == c) & (gl.L == L)].t_ext.to_numpy(float)   # noqa: E731
    return {"C1, space": (base[base.point == 0].t_ext.to_numpy(float), 600, C_SP, "-", 1.3),
            "SL, space": (sp1k[sp1k.point == 1].t_ext.to_numpy(float), 1000, C_SP, dot, 1.3),
            "C1, w.m. cell": (wm[wm.point == 0].t_ext.to_numpy(float), 600, C_WM, "-", 1.3),
            "SL, w.m. cell": (eco[eco.cfg == "WM_SL"].t_ext.to_numpy(float), 600, C_WM, dot, 1.3),
            "C1, w.m. global": (g("C1_ref", 30), 1000, CAT[2], "-", 1.3),
            "SL, w.m. global": (g("SL", 30), 1000, CAT[2], dot, 1.3),
            "SL, w.m. global, L=60": (g("SL", 60), 1000, CAT[2], dot, 0.8)}


def _wm_summary():
    from pipeline import f2b
    return f2b.load(WMDIR)["summary"]


# ============================================================================= 4. patrones de comportamiento
def composition(S):
    """Media entre colonias vivas (mientras >= 50 % vivas) de las fracciones de adultos por clase."""
    nrep = S.rep.nunique()
    g = S[(S.N > 0) & S.phi_w.notna()]
    m = g.groupby("t")[["phi_w", "frac_crowded_low", "frac_withdrawn", "phi_BO"]].mean()
    alive = g.groupby("t").rep.nunique() / nrep
    m = m[alive.reindex(m.index) >= 0.5]
    m["other"] = (m.phi_w - m.frac_crowded_low - m.frac_withdrawn).clip(lower=0)
    m["healthy"] = 1 - m.phi_w
    return m


def pup_survival(S, tmax=60):
    g = S[S.t <= tmax].groupby("t")[["births", "conceptions"]].sum()
    surv = g.births / (g.conceptions * LBAR)
    surv[g.conceptions < 20] = np.nan
    return surv.index.to_numpy(float), surv.to_numpy(float)


def cohort_fertility(S, min_born=30):
    """Fracción de las hembras de cada semana de nacimiento que alguna vez conciben (totales finales)."""
    last = S.sort_values("t").groupby(["rep", "t"])[["coh_f_born", "coh_f_ever_conceived"]].last()
    c = last.groupby("t").sum()
    c = c[c.coh_f_born >= min_born]
    f = c.coh_f_ever_conceived / c.coh_f_born
    lo, hi = zip(*[wilson(k, n) for k, n in zip(c.coh_f_ever_conceived, c.coh_f_born)])
    return c.index.to_numpy(float), f.to_numpy(float), np.array(lo), np.array(hi), c.coh_f_born.to_numpy()


def fig_behaviour():
    Ssp, Swm = base_series(0), wm_series(0)
    P = ens_phases(Ssp)
    bnd = bounds_from(P)
    fig = plt.figure(figsize=(W2, 4.9))
    gs0 = fig.add_gridspec(2, 1, hspace=0.78, height_ratios=[1, 1])
    gt = gs0[0].subgridspec(1, 2, wspace=0.18)
    gb = gs0[1].subgridspec(1, 3, wspace=0.62, width_ratios=[1, 1, 1.1])
    cls = [("healthy", "competent ($s\\geq0.2$)", "#e4e3df"),
           ("other", "non-breeders ($s<0.2$), intermediate density", "#b7d3f6"),
           ("frac_crowded_low", "crowded non-breeders ($m>m_c$)", CAT[1]),
           ("frac_withdrawn", "withdrawn non-breeders ($m\\leq2$)", CAT[6])]
    NUMS["behaviour"] = {}
    for j, (S, title) in enumerate(((Ssp, "spatial model: crowded and withdrawn non-breeders"),
                                    (Swm, "well-mixed model: isolation is brief and random"))):
        ax = fig.add_subplot(gt[j])
        m = composition(S)
        m = m[m.index <= 150]
        ys = np.vstack([m[k].to_numpy() for k, _, _ in cls])
        ax.stackplot(m.index, ys, colors=[c for _, _, c in cls], labels=[l for _, l, _ in cls], lw=0,
                     edgecolor="white")
        ax.plot(m.index, 1 - m.phi_BO, color=INK, lw=0.9, ls=(0, (3, 1.5)),
                label="withdrawn for $\\geq4$ weeks in a row (below the line)")
        ax.set_xlim(0, 150); ax.set_ylim(0, 1)
        ax.set_xlabel("week"); ax.set_ylabel("share of adults" if j == 0 else "")
        if j == 1:
            ax.tick_params(labelleft=False)
        ax.grid(False)
        vphases(ax, bnd if j == 0 else bounds_from(ens_phases(S)), y=1.075)
        panel(ax, "ab"[j], title, pad=10)
        k60 = m.loc[m.index >= 60].iloc[0]
        NUMS["behaviour"]["spatial" if j == 0 else "wellmixed"] = dict(
            week=int(k60.name), crowded=float(k60.frac_crowded_low), withdrawn=float(k60.frac_withdrawn),
            persistent=float(k60.phi_BO), deteriorated=float(k60.phi_w))
        if j == 0:
            hl, ll = ax.get_legend_handles_labels()
    fig.legend(hl[::-1], ll[::-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.535), fontsize=7,
               handlelength=1.6, columnspacing=1.2)
    # (c) supervivencia de las crías
    c = fig.add_subplot(gb[0])
    for S, col, nm in ((Ssp, C_SP, "spatial"), (Swm, C_WM, "well-mixed")):
        t, sv = pup_survival(S)
        c.plot(t, sv, color=col, label=nm)
    if "pups" in Ssp:     # control del estimador births / (conceptions * Lbar) con el conteo exacto de crías
        g = Ssp[Ssp.t <= 60].groupby("t")[["births", "pups"]].sum()
        ok = g.pups >= 100
        t_, sv_ = pup_survival(Ssp)
        ex = (g.births / g.pups)[ok]
        NUMS.setdefault("behaviour", {})["pup_survival_estimator_maxdiff"] = float(
            np.nanmax(np.abs(pd.Series(sv_, index=t_).reindex(ex.index) - ex)))
    c.set_ylim(0, 1); c.set_xlim(0, 60)
    vphases(c, bnd, y=1.1)
    c.set_xlabel("week"); c.set_ylabel("pups surviving birth")
    c.legend(loc="lower left", fontsize=7)
    panel(c, "c", "failure of maternal care", pad=11)
    # (d) cohortes: fracción de hembras que alguna vez conciben
    dd = fig.add_subplot(gb[1])
    for S, col, nm, mk in ((Ssp, C_SP, "spatial", "o"), (Swm, C_WM, "well-mixed", "s")):
        t, f, lo, hi, nb = cohort_fertility(S)
        dd.plot(t, f, color=col, lw=1.1, label=nm)
        dd.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        NUMS["behaviour"][f"cohort_{nm}"] = {f"{int(a)}": float(b) for a, b in zip(t, f)}
    dd.axhline(0.18, color=C_CAL, lw=0.8, ls="--")
    dd.text(59, 0.2, "two-cell universe: 18%", fontsize=7, color=INK2, ha="right", va="bottom")
    dd.set_xlim(0, 60); dd.set_ylim(0, 1.02)
    vphases(dd, bnd, y=1.1)
    dd.set_xlabel("week of birth"); dd.set_ylabel("females that ever conceive")
    dd.legend(loc="upper right", fontsize=7)
    panel(dd, "d", "late-born cohorts never breed", pad=11)
    # (e) transplantes (Marsden 1972)
    e = fig.add_subplot(gb[2])
    A = FR.trans_long()

    def pt(arm, ph):
        r = A[(A.brazo == arm) & (A.fase == ph)].iloc[0]
        return float(r.P_T), float(r.lo), float(r.hi), int(r.k), int(r.n)
    items = [("phase B", "C1", "B"), ("phase D", "C1", "D"),
             ("phase D, healthy\npartners", "C1", "MIX_F"),
             ("phase D, adults\nallowed to recover", "X: madre C1, colonia -R1 -R2 rec=0.1", "D")]
    y = np.arange(len(items))[::-1]
    NUMS["behaviour"]["transplants"] = {}
    for yy, (lab, arm, ph) in zip(y, items):
        f, lo, hi, k, n = pt(arm, ph)
        if ph == "MIX_F":
            f2, lo2, hi2, k2, n2 = pt(arm, "MIX_M")
            k, n = k + k2, n + n2
            f = k / n; lo, hi = wilson(k, n)
        col = C_SP if "recover" not in lab else MUTED
        e.errorbar([f], [yy], xerr=xe([f], [lo], [hi]), fmt="o", ms=4, color=col, elinewidth=0.9, capsize=0)
        e.text(1.04, yy, f"{k}/{n}", fontsize=7, color=INK2, va="center", ha="left")
        NUMS["behaviour"]["transplants"][lab.replace("\n", " ")] = dict(k=k, n=n, f=f)
    e.set_yticks(y); e.set_yticklabels([i[0] for i in items], fontsize=7)
    e.set_xlim(-0.03, 1.03); e.set_ylim(-0.6, len(items) - 0.4)
    e.set_xlabel("P(transplanted group founds a colony)")
    e.grid(axis="y", visible=False)
    panel(e, "e", "transplants")
    save(fig, "fig_em_behaviour")


# ============================================================================= Fase 9a: matriz (A2, A5) y ablaciones
F9 = EM / "pattern_tables"


def _fmt(f):
    if not np.isfinite(f):
        return ""
    if f >= 0.9995:
        return "1"
    if f <= 0.0005:
        return "0"
    return f"{f:.2f}".lstrip("0")


def _matrix_axes(fig, rect, nr, nc, gaps):
    ax = fig.add_axes(rect)
    xs = np.arange(nc) + np.array([sum(g <= j for g in gaps) * 0.35 for j in range(nc)])
    ax.set_xlim(-0.55, xs[-1] + 0.55); ax.set_ylim(nr - 0.5, -0.5); ax.axis("off")
    return ax, xs


def _draw_cell(ax, x, y, status, f=np.nan, fm=np.nan, bold=False, sub=None):
    cmap = FR.CMAP
    if status != "ok":
        ax.add_patch(Rectangle((x - 0.46, y - 0.44), 0.92, 0.88, facecolor="white", edgecolor=GRID,
                               hatch="//////", lw=0.4))
        ax.text(x, y, {"trunc": "trunc.", "ne": "–", "na": "n.a."}.get(status, status), ha="center",
                va="center", fontsize=7, color=INK2, bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
        return
    ax.add_patch(Rectangle((x - 0.46, y - 0.44), 0.92, 0.88, facecolor=cmap(f), edgecolor="white", lw=0.5))
    tc = "white" if f > 0.55 else INK
    if sub:
        ax.text(x, y - 0.13, _fmt(f), ha="center", va="center", fontsize=7, color=tc,
                fontweight="bold" if bold else "normal")
        ax.text(x, y + 0.24, sub, ha="center", va="center", fontsize=7, color=tc if f > 0.55 else INK2)
    else:
        ax.text(x, y, _fmt(f), ha="center", va="center", fontsize=7, color=tc, fontweight="bold" if bold else "normal")


def fig_patterns():
    """Fig. 4 (Fase 9a): fracción de corridas que pasan cada patrón (color y número) y, debajo, la fracción
    MARGINAL ("m"), una sola fuente por columna, evaluador primario; datos de em_f9_analysis.py (matrix.csv)."""
    M = pd.read_csv(F9 / "matrix.csv")
    lay = json.load(open(F9 / "matrix_layout.json"))
    rows, cols = lay["rows"], lay["cols"]
    nr, nc = len(rows), len(cols)
    groups = []
    for j, c in enumerate(cols):
        if not groups or groups[-1][0] != c[1]:
            groups.append([c[1], j, j])
        groups[-1][2] = j
    gaps = [g[1] for g in groups[1:]]
    fig = plt.figure(figsize=(W2, 5.05))
    x0, w = 0.335, 0.655
    ax, xs = _matrix_axes(fig, [x0, 0.085, w, 0.775], nr, nc, gaps)
    Mi = M.set_index(["col", "row"])
    NUMS["patterns"] = {}
    for i, (code, lab, ph, note) in enumerate(rows):
        for j, (key, grp, head, src) in enumerate(cols):
            r = Mi.loc[(key, code)]
            st = r.status
            if st == "ok" and r.k_na == r.n:
                st = "na"
            if st != "ok":
                _draw_cell(ax, xs[j], i, st)
                continue
            fm = r.f_pm
            sub = f"m{_fmt(r.k_marg / r.n)}" if r.k_marg / r.n >= 0.01 else None
            _draw_cell(ax, xs[j], i, "ok", r.f_pass, fm, bold=(code == "O17p"), sub=sub)
            NUMS["patterns"][f"{key}:{code}"] = dict(n=int(r.n), pass_=int(r.k_pass), marg=int(r.k_marg))
    phases = [r[2] for r in rows]
    starts = [0] + [i for i in range(1, nr) if phases[i] != phases[i - 1]] + [nr]
    xl = -0.55
    for k in range(len(starts) - 1):
        i0, i1 = starts[k], starts[k + 1] - 1
        if k > 0:
            ax.plot([-6.0, xs[-1] + 0.5], [i0 - 0.5, i0 - 0.5], color=MUTED, lw=0.6, clip_on=False)
        ax.text(-5.75, (i0 + i1) / 2, phases[i0], ha="center", va="center", fontsize=7.5, color=INK,
                fontweight="bold", clip_on=False)
    for i, (code, lab, ph, note) in enumerate(rows):
        mk = {"constr": "$^{\\ddagger}$", "post": "$^{\\S}$"}.get(note, "")
        ax.text(-1.35, i, lab + mk, ha="right", va="center", fontsize=7, color=INK,
                fontweight="bold" if code == "O17p" else "normal")
        ax.text(xl, i, {"ceiling": "", "phC": "O3c", "O11aw": "O11$_{a_w}$"}.get(code, code), ha="right",
                va="center", fontsize=7, color=MUTED)
    for j, (key, grp, head, src) in enumerate(cols):
        ax.text(xs[j], -0.62, head, ha="center", va="bottom", fontsize=7, color=INK, linespacing=0.95)
    for grp, j0, j1 in groups:
        ax.plot([xs[j0] - 0.42, xs[j1] + 0.42], [-1.95, -1.95], color=INK2, lw=0.8, clip_on=False)
        ax.text((xs[j0] + xs[j1]) / 2, -2.05, grp, ha="center", va="bottom", fontsize=7.5, color=INK, clip_on=False)
    ax.text(-1.35, -0.62, "Calhoun's pattern", ha="right", va="bottom", fontsize=7, color=INK)
    cax = fig.add_axes([0.04, 0.03, 0.15, 0.018])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=FR.CMAP, norm=plt.Normalize(0, 1)), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    fig.text(0.04, 0.062, "share of runs that pass", fontsize=7, color=INK2)
    fig.text(0.245, 0.04, "m: fraction MARGINAL (one step from the threshold).  Hatched: trunc., runs stopped by the "
             "cap on $N$;\nn.a., criterion undefined (no run reaches the event).  $\\ddagger$: holds by construction.  "
             "$\\S$: added after the\npreregistration (O3c: phase-C clause of O3).  †: every run stopped at weeks 26–29 "
             "(median 27) by the cap on $N$.", fontsize=7, color=INK2, va="center", linespacing=1.25)
    save(fig, "fig_em_patterns")


def fig_ablations():
    """Fig. S (A2): qué patrón falla en cada ablación de C1 (f2b_abl1, f2b_abl2; evaluador
    primario; m = MARGINAL). n = 120 en las ablaciones simples (abl1) y 40 en las dobles y el núcleo (abl2)."""
    A = pd.read_csv(F9 / "ablations.csv")
    labs = list(dict.fromkeys(A.label))
    rows = list(dict.fromkeys(A.row))
    NAMES = {"O10b": "Socially functional growth", "O4": "Peak far below capacity", "O10": "Adults deteriorate "
             "before peak", "O14": "Aggregation with free space", "O13": "Crowded and withdrawn coexist",
             "O13p": "Withdrawn class persists", "O11": "Late cohorts fail to mature", "O5": "Natality ceases",
             "O6": "No recovery of natality", "O7": "Extinction", "O8": "Long senescent decline",
             "O9": "Old female-biased remnant", "O17p": "All essential patterns"}
    nr, nc = len(rows), len(labs)
    fig = plt.figure(figsize=(W2, 3.6))
    ax, xs = _matrix_axes(fig, [0.23, 0.07, 0.76, 0.74], nr, nc, [1, 8])
    Ai = A.set_index(["label", "row"])
    for i, code in enumerate(rows):
        for j, lab in enumerate(labs):
            r = Ai.loc[(lab, code)]
            sub = f"m{_fmt(r.k_marg / r.n)}" if r.k_marg / r.n >= 0.01 else None
            _draw_cell(ax, xs[j], i, "ok", r.f_pass, r.f_pm, bold=(code == "O17p"), sub=sub)
        ax.text(-0.6, i, f"{NAMES[code]}  ", ha="right", va="center", fontsize=7, color=INK,
                fontweight="bold" if code == "O17p" else "normal")
        ax.text(-0.6, i, "", ha="right")
    for j, lab in enumerate(labs):
        ax.text(xs[j], -0.62, lab, ha="center", va="bottom", fontsize=7, color=INK, linespacing=0.95)
    for (j0, j1, g) in ((1, 7, "one rule removed or modified"), (8, nc - 1, "two rules removed, or all four")):
        ax.plot([xs[j0] - 0.42, xs[j1] + 0.42], [-1.75, -1.75], color=INK2, lw=0.8, clip_on=False)
        ax.text((xs[j0] + xs[j1]) / 2, -1.85, g, ha="center", va="bottom", fontsize=7.5, color=INK, clip_on=False)
    fig.text(0.23, 0.012, "Fraction of runs that pass (colour and number; $n=120$ per single ablation, $n=40$ per "
             "double ablation and core;\nprimary evaluator, $L=30$, $T=600$); m: fraction MARGINAL.", fontsize=7,
             color=INK2, linespacing=1.2)
    save(fig, "fig_em_s_ablations")


# ============================================================================= 6. condiciones del mecanismo
E1 = EM / "adult_recovery" / "summary.csv"


def _pooled_alive(df, col):
    """Colonias vivas al final (1 - P_ext) sumadas sobre filas; devuelve k, n, f, lo, hi."""
    n = int(df.n.sum())
    k = int(np.round(((1 - df[col]) * df.n).sum()))
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, f=k / n if n else np.nan, lo=lo, hi=hi)


def fig_conditions():
    """Figura suplementaria mínima (decisión de brevedad, Fase 8a): el contenido completo (recuperación adulta,
    hidra, P3) está en fig_adult_recovery y fig_hydra del artículo matemático, que se citan. Un solo panel,
    "adapted from": P(extinción a 1000 semanas) frente al factor de recuperación adulta, L = 30, espacial y bien
    mezclado, aprendizaje social puro (lambda = 1) y recuperación asocial (lambda = 0)."""
    E = pd.read_csv(E1)
    E = E[(E.model == "C1") & (E.variant == "cell") & (E.L == 30)]
    fig, ax = plt.subplots(figsize=(W1, 2.35))
    ax.axvspan(-0.001, 0.02, color=PH_COL["B"], lw=0, zorder=0)
    ax.text(0.0105, 0.42, "collapse\nin every\nrun", ha="center", va="center", fontsize=7, color=INK2)
    NUMS["conditions"] = {}
    for space, col, mk, nm in (("sp", C_SP, "o", "spatial"), ("wm", C_WM, "s", "well-mixed")):
        for lam, ls in ((1.0, "-"), (0.0, "--")):
            g = E[(E.space == space) & (E.lam == lam)].sort_values("out")
            x, f = g.out.to_numpy(float), g.P_ext_1000.to_numpy(float)
            ax.plot(x, f, color=col, ls=ls, marker=mk, ms=2.8, lw=1.1,
                    label=f"{nm}, {'social learning only' if lam == 1 else 'individual recovery'}")
            mid = (f > 0) & (f < 1)
            ax.errorbar(x[mid], f[mid], yerr=xe(f[mid], g.P_ext_1000_lo.to_numpy()[mid],
                                                   g.P_ext_1000_hi.to_numpy()[mid]),
                        fmt="none", ecolor=col, elinewidth=0.6, alpha=0.7)
            NUMS["conditions"][f"{space}_lam{lam:g}_L30"] = {f"{a:g}": float(b) for a, b in zip(x, f)}
            NUMS["conditions"][f"{space}_lam{lam:g}_L30_n"] = int(g.n.iloc[0])
    ax.set_xscale("symlog", linthresh=0.02, linscale=0.35)
    ax.set_xlim(0, 1.05); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1]); ax.set_xticklabels(["0", ".02", ".05", ".1", ".2", ".5", "1"])
    ax.set_xlabel("adult recovery factor $\\varepsilon_{\\rm ad}$")
    ax.set_ylabel("P(extinct by week 1000)")
    ax.legend(loc="lower left", fontsize=7, handlelength=2.4, bbox_to_anchor=(0.0, 0.0))
    save(fig, "fig_em_s_recovery")


# ============================================================================= 7. robustez estadística
def fig_robustness():
    fig = plt.figure(figsize=(W2, 5.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.08], hspace=0.62, wspace=0.55,
                          width_ratios=[1, 1, 1.05])
    NUMS["robustness"] = {}
    # (a) mortalidad de fondo
    a = fig.add_subplot(gs[0, 0])
    Mu = pd.DataFrame(NUM["robmu"])
    for j, (cand, name, col, mk) in enumerate((("C1", "C1", C_SP, "o"), ("C1p", "C1$'$ (minimal)", C_C1P, "s"))):
        d = Mu[Mu.candidato == cand].sort_values(["mu_bg", "n"]).drop_duplicates("mu_bg", keep="last")
        x = d.mu_bg.to_numpy(float) * 1e3      # un punto por mu_bg: el de mayor n (200 en 0, 0.003, 0.008; 60 si no)
        a.errorbar(x, d.O17p, yerr=xe(d.O17p, d.lo, d.hi), fmt=mk + "-", ms=3.6, color=col, lw=1.1,
                   elinewidth=0.7, capsize=0, label=name)
    a.axhline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("background mortality ($10^{-3}$ per week)"); a.set_ylabel("share of Calhoun-like runs")
    a.set_ylim(-0.03, 1.03); a.legend(loc="upper right", fontsize=7)
    panel(a, "a", "extra mortality")
    # (b) tamaño
    b = fig.add_subplot(gs[0, 1])
    Sz = pd.DataFrame(NUM["size"]["tabla"])
    b.errorbar(Sz.L ** 2, Sz.O17p, yerr=xe(Sz.O17p, Sz.lo, Sz.hi), fmt="o-", color=C_SP, ms=3.6, lw=1.0,
               elinewidth=0.7, capsize=0, label="Calhoun-like runs")
    b.errorbar(Sz.L ** 2, Sz.spread, yerr=xe(Sz.spread, Sz.spread_lo, Sz.spread_hi), fmt="s--", color=INK2,
               ms=3.4, lw=0.9, elinewidth=0.7, capsize=0, label="spread of $T_{\\rm ext}$ (IQR/median)")
    b.plot(Sz.L ** 2, Sz.BO4, "^:", color=CAT[4], ms=3.6, lw=0.9, label="persistent withdrawn share")
    b.set_xscale("log"); b.set_xticks([225, 400, 900, 2025, 3600]); b.set_xticklabels(["225", "400", "900",
                                                                                         "2025", "3600"])
    b.minorticks_off()
    b.set_xlabel("cells $L^2$ (same density)"); b.set_ylim(0, 1.05)
    b.legend(loc="center", fontsize=7, bbox_to_anchor=(0.55, 0.58))
    panel(b, "b", "system size")
    # (c) ablaciones (como fig_pre_rules b): evaluador primario y preregistrado
    c = fig.add_subplot(gs[0, 2])
    abl = [("abl1", "C1", "C1"), ("abl1", "C1 -R1", "$-$R1"), ("abl1", "C1 -R2", "$-$R2"),
           ("abl1", "C1 -AV", "$-$AV"), ("abl1", "C1 -R7", "$-$R7"),
           ("abl2", "C1 -R1 -R2", "$-$R1 $-$R2"), ("abl2", "C1 -R1 -R2 -AV -R7", "no rules"),
           ("abl1", "C1 w=0 rec=0.15", "$w=0$")]
    T = {"abl1": FR.tab("abl1"), "abl2": FR.tab("abl2")}
    Tv = {"abl1": FR.ptab("abl1", "v1_"), "abl2": FR.ptab("abl2", "v1_")}
    yy = np.arange(len(abl))[::-1]
    rows_ = []
    for st, lab, name in abl:
        r = T[st][T[st].label == lab].iloc[0]
        rv = Tv[st][Tv[st].label == lab].iloc[0]
        rows_.append(dict(name=name, n=int(r["n"]), f=float(r.f_O17p), lo=float(r.lo_O17p), hi=float(r.hi_O17p),
                          fv1=float(rv.f_O17p)))
    A_ = pd.DataFrame(rows_)
    c.errorbar(A_.f, yy, xerr=xe(A_.f, A_.lo, A_.hi), fmt="o", ms=3.6, color=C_SP, elinewidth=0.9, capsize=0,
               label="primary evaluator")
    c.plot(A_.fv1, yy, "D", ms=3.2, mfc="white", mec=CAT[1], mew=1.0, ls="none", label="preregistered")
    for y_, (_, r) in zip(yy, A_.iterrows()):
        c.text(1.05, y_, f"$n$={r.n}", fontsize=7, color=INK2, va="center", ha="left")
    c.axvline(0.8, color=MUTED, lw=0.6, ls=":")
    c.set_yticks(yy); c.set_yticklabels(A_.name, fontsize=7)
    c.set_xlim(-0.03, 1.03); c.set_xlabel("share of Calhoun-like runs"); c.grid(axis="y", visible=False)
    c.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.0, 0.18))
    NUMS["robustness"]["ablations"] = rows_
    panel(c, "c", "rule ablations")
    # (d) variantes
    d = fig.add_subplot(gs[1, 0])
    Rb = pd.DataFrame(NUM["rob"])
    names = {"C1 R7 legacy (v1)": "single-pass retreat-site rule", "C1 nido=3": "nest period 3 weeks",
             "C1 async": "asynchronous update", "C1 async blanda": "async., soft site capacity",
             "C1 CI edades 4-52": "initial ages 4–52 weeks", "C1 CI edades 4-80": "initial ages 4–80 weeks",
             "C1 s0=0.25": "pups born at $s=0.25$", "C1 mu_bg=0.003": "background mortality 0.003"}
    order = list(names)
    y = np.arange(len(order))[::-1]
    for j, (cand, col, mk) in enumerate((("C1", C_SP, "o"), ("C1p", C_C1P, "s"))):
        dd = Rb[Rb.candidato == cand].copy()
        dd["key"] = dd.brazo.str.replace("C1p ", "C1 ", regex=False)
        dd = dd.set_index("key").reindex(order)
        yy2 = y + (0.16 if j == 0 else -0.16)
        d.errorbar(dd.O17p, yy2, xerr=xe(dd.O17p, dd.lo, dd.hi), fmt=mk, ms=3.4, color=col, elinewidth=0.8,
                   capsize=0, label="C1" if j == 0 else "C1$'$")
    d.axvline(0.8, color=MUTED, lw=0.6, ls=":")
    d.set_yticks(y); d.set_yticklabels([names[k] for k in order], fontsize=7)
    d.set_xlim(-0.03, 1.03); d.set_xlabel("share of Calhoun-like runs"); d.grid(axis="y", visible=False)
    d.legend(loc="upper left", fontsize=7)
    panel(d, "d", "implementation variants ($n=200$)")
    # (e) Sobol
    e = fig.add_subplot(gs[1, 1])
    so = pd.read_csv(RES2 / "f2b_sobol" / "analisis" / "sobol.csv")
    dso = so[so.metric == "f_PRIMARIO"].set_index("name").sort_values("ST")
    yy = np.arange(len(dso))
    rel = dso["relevante"].astype(str) == "True"
    e.barh(yy, dso.ST, height=0.62, color=[C_SP if r_ else "#b7d3f6" for r_ in rel], label="total $S_T$")
    e.errorbar(dso.ST, yy, xerr=dso.ST_conf, fmt="none", ecolor=INK2, elinewidth=0.7)
    e.plot(dso.S1.clip(lower=0), yy, "o", ms=3.4, mfc="white", mec=INK, mew=0.8, ls="none", label="first order $S_1$")
    e.plot(dso.ST_ruido, yy, "|", color=CAT[7], ms=8, mew=1.4, ls="none", label="noise floor")
    e.set_yticks(yy); e.set_yticklabels([FR.FNAME[n] for n in dso.index])
    e.set_xlim(0, 0.72); e.set_xlabel("Sobol' index"); e.grid(axis="y", visible=False)
    e.legend(loc="lower right", fontsize=7)
    panel(e, "e", "sensitivity (11 factors)")
    # (f) región factible
    f_ = fig.add_subplot(gs[1, 2])
    keys = ["all", "cR<=2", "+mc<=10", "+fRcR>=0.3", "+pi>=100"]
    for j, (pref, lab, col, hatch) in enumerate((("primary", "primary evaluator", C_SP, None),
                                                 ("v1_", "preregistered", CAT[3], "//"))):
        dec = NUM[f"lhs_decomp_{pref}"]
        fv = np.array([dec[k]["f"] for k in keys]); lo = np.array([dec[k]["lo"] for k in keys])
        hi = np.array([dec[k]["hi"] for k in keys])
        x = np.arange(len(keys)) + (j - 0.5) * 0.34
        f_.bar(x, fv, width=0.32, color=col, label=lab, hatch=hatch, edgecolor="white", lw=0.4)
        f_.errorbar(x, fv, yerr=xe(fv, lo, hi), fmt="none", ecolor=INK2, elinewidth=0.7)
    ns = [NUM["lhs_decomp_primary"][k]["n"] for k in keys]
    f_.set_xticks(range(5))
    f_.set_xticklabels([f"{t}\n{n}" for t, n in zip(["all", "$c_R\\leq2$", "$+m_c$\n$\\leq10$", "$+f_Rc_R$\n$\\geq0.3$",
                                                     "$+\\pi$\n$\\geq100$"], ns)], fontsize=7)
    f_.set_ylabel("share of parameter points"); f_.set_ylim(0, 0.65)
    f_.legend(loc="upper left", fontsize=7); f_.grid(axis="x", visible=False)
    panel(f_, "f", "Calhoun-like region")
    NUMS["robustness"]["size"] = NUM["size"]["tabla"]
    save(fig, "fig_em_robustness")


# ============================================================================= Fase 9a: recuperación adulta (A3)
def _shade(col, f):
    """Mezcla col con blanco (f = 0: col; f = 1: blanco)."""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(col))
    return tuple(c + (1 - c) * f)


L_STYLE = {15: (0.55, "o"), 30: (0.25, "s"), 60: (0.0, "^")}


def fig_recovery():
    """A3 (cuerpo): P(extinción a 1000 semanas) frente a eps_ad, espacial y bien mezclado, L = 15, 30, 60
    (results_em/adult_recovery; 20 / 20 / 10 colonias por punto), y el umbral de rescate eps^c(L)
    (transicion.csv: punto medio del ajuste logístico)."""
    E = pd.read_csv(E1)
    E = E[E.model == "C1"]
    Tr = pd.read_csv(EM / "adult_recovery" / "transicion.csv")
    fig = plt.figure(figsize=(W2, 2.3))
    gs = fig.add_gridspec(1, 3, wspace=0.36, width_ratios=[1, 1, 0.82])
    NUMS["recovery"] = {}
    xt = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    def curves(ax, sel, col, ls, lab):
        for L, (sh, mk) in L_STYLE.items():
            g = E[sel & (E.L == L)].sort_values("out")
            if g.empty:
                continue
            x, f = g.out.to_numpy(float), g.P_ext_1000.to_numpy(float)
            c_ = _shade(col, sh)
            ax.plot(x, f, color=c_, ls=ls, marker=mk, ms=2.9, lw=1.0, mfc=c_ if ls == "-" else "white", mew=0.8)
            mid = (f > 0) & (f < 1)
            ax.errorbar(x[mid], f[mid], yerr=xe(f[mid], g.P_ext_1000_lo.to_numpy()[mid],
                                                   g.P_ext_1000_hi.to_numpy()[mid]),
                        fmt="none", ecolor=c_, elinewidth=0.6, alpha=0.8)
            NUMS["recovery"][f"{lab}_L{L}"] = {f"{a:g}": float(b) for a, b in zip(x, f)}
            NUMS["recovery"][f"{lab}_L{L}_n"] = int(g.n.iloc[0])

    def deco(ax):
        ax.set_xscale("symlog", linthresh=0.02, linscale=0.35)
        ax.set_xlim(0, 1.05); ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(xt); ax.set_xticklabels(["0", ".02", ".05", ".1", ".2", ".5", "1"])
        ax.axvspan(-0.001, 0.02, color=PH_COL["B"], lw=0, zorder=0)
        ax.set_xlabel("adult recovery factor $\\varepsilon_{\\rm ad}$")
    a = fig.add_subplot(gs[0])
    curves(a, (E.space == "sp") & (E.lam == 0.0), C_SP, "-", "sp_lam0")
    curves(a, (E.space == "sp") & (E.lam == 1.0), C_SP, (0, (3, 1.5)), "sp_lam1")
    deco(a); a.set_ylabel("P(extinct by week 1000)")
    panel(a, "a", "space: solid $\\lambda=0$, dotted 0.9, dashed 1")
    b = fig.add_subplot(gs[1])
    curves(b, (E.space == "wm") & (E.variant == "cell") & (E.lam == 0.0), C_WM, "-", "wm_cell_lam0")
    curves(b, (E.space == "wm") & (E.variant == "global") & (E.lam == 0.0), CAT[2], (0, (3, 1.5)),
           "wm_global_lam0")
    deco(b)
    hs = [Line2D([], [], color=_shade(INK2, sh), marker=mk, ms=3, lw=1.0) for L, (sh, mk) in L_STYLE.items()]
    lg = b.legend(hs, [f"$L={L}$" for L in L_STYLE], loc="lower left", bbox_to_anchor=(-0.01, 0.30), fontsize=7,
                  handlelength=1.6)
    b.add_artist(lg)
    b.legend([Line2D([], [], color=C_WM, lw=1.0), Line2D([], [], color=CAT[2], lw=1.0, ls=(0, (3, 1.5)))],
             ["mating in the cell", "global mating"], loc="lower left", fontsize=7, handlelength=2.0,
             bbox_to_anchor=(-0.01, 0.02))
    panel(b, "b", "w.m., $\\lambda=0$ ($\\lambda\\geq0.9$: all but one extinct)")
    c = fig.add_subplot(gs[2])
    lines = [("sp", "cell", 0.0, C_SP, "-", "o", "space, $\\lambda=0$"),
             ("sp", "cell", 0.9, C_SP, (0, (1, 1)), "D", "space, $\\lambda=0.9$"),
             ("sp", "cell", 1.0, C_SP, (0, (3, 1.5)), "s", "space, $\\lambda=1$"),
             ("wm", "cell", 0.0, C_WM, "-", "o", "w.m. local, $\\lambda=0$"),
             ("wm", "global", 0.0, CAT[2], (0, (3, 1.5)), "s", "w.m. global, $\\lambda=0$")]
    NUMS["recovery"]["threshold"] = {}
    for sp_, var, lam, col, ls, mk, lab in lines:
        g = Tr[(Tr.space == sp_) & (Tr.variant == var) & (Tr.lam == lam)].sort_values("L")
        c.plot(g.L, g.out_c, color=col, ls=ls, marker=mk, ms=3.2, lw=1.0, label=lab, mfc="white", mew=0.9)
        NUMS["recovery"]["threshold"][lab] = {int(L): float(v) for L, v in zip(g.L, g.out_c)}
    c.axhline(0.585, color=C_WM, lw=0.6, ls=":")
    c.text(16, 0.6, "kinetic limit 0.585", fontsize=7, color=C_WM, va="bottom")
    c.set_xscale("log"); c.set_xticks([15, 30, 60]); c.set_xticklabels(["15", "30", "60"]); c.minorticks_off()
    c.set_xlim(13, 68); c.set_ylim(0, 1.05)
    c.set_xlabel("lattice side $L$"); c.set_ylabel("rescue threshold $\\varepsilon^c_{\\rm ad}$")
    fig.legend(*c.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7,
               ncol=5, handlelength=2.0, columnspacing=1.0)
    panel(c, "c", "threshold falls with $L$")
    save(fig, "fig_em_recovery")


if __name__ == "__main__":
    which = sys.argv[1:] or ["mechanism", "trajectory", "wellmixed", "behaviour", "patterns", "robustness",
                             "conditions", "recovery", "ablations", "snapshots"]
    for w in which:
        globals()[f"fig_{w}"]()
    old = json.loads(NUMS_OUT.read_text()) if NUMS_OUT.exists() else {}
    sizes = {**old.get("_sizes_mm", {}), **NUMS.get("_sizes_mm", {})}
    old.update(NUMS)
    old["_sizes_mm"] = sizes
    NUMS_OUT.write_text(json.dumps(old, indent=1, default=float))
    print("nums", NUMS_OUT)
