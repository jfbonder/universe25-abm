"""Figuras de la versión reducida para Physical Review E (manuscript/pre/), a partir de los mismos datos que
figs_results.py.

    # desde la raíz del repositorio
    python3 scripts/figures/figs_pre.py          # o: make figs

Importa figs_results.py por sus utilidades y su carga de datos (results_v2/f2b_*/: summary.csv, series.parquet,
analisis/*.csv) sin ejecutar sus figuras, y escribe
  manuscript/pre/figs/fig_pre_*.pdf     cinco figuras: anchos 3.39 in (\\columnwidth, 8.6 cm) o 7.0 in
                                        (\\textwidth, 17.8 cm) de REVTeX a dos columnas, fuente base 8 pt.
                                        Texto principal (sections/results.tex): trajectories (figure*), rules,
                                        transplant, robustness (figure*). Suplemento (supplement.tex, Fig. S1): global.
  manuscript/pre/tables/tab_res_*.tex   copias de las tablas de manuscript/extended/tables/ que \\input-ea el suplemento
No corre simulaciones ni reescribe results_numbers.json. Las cifras de robustez (robmu, rob, size), de la región
factible (lhs_decomp_*) y la paleta se toman de figs_results.py / results_numbers.json, para que coincidan con las
del texto; la supervivencia, las trayectorias, las ablaciones, los transplantes y los índices de Sobol se leen de
las mismas salidas de results_v2/ que usa figs_results.py. Identidad nunca sólo por color (estilo de línea o
marcador como codificación secundaria); paleta categórica validada (dataviz).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "scripts" / "production"), str(ROOT / "scripts" / "analysis"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402

import figs_results as FR               # noqa: E402  (sólo utilidades y carga de datos; no ejecuta figuras)

PRE = ROOT / "manuscript" / "pre"
FIGS = PRE / "figs"
TABS = PRE / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)
NUM = json.loads(FR.NUMS.read_text())

CAT, INK, INK2, MUTED, GRID, LS, MK = FR.CAT, FR.INK, FR.INK2, FR.MUTED, FR.GRID, FR.LS, FR.MK
COL = 3.39     # \columnwidth de REVTeX (8.6 cm)
FULL = 7.0     # \textwidth (17.8 cm)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8, "lines.linewidth": 1.3, "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("fig", FIGS / name)


def panel(ax, letter):
    FR.panel(ax, letter)


# ============================================================================= 1. trayectorias (figure*)
def fig_trajectories():
    """Como fig_res_trajectories: N/K_gamma, nacimientos, estado social adulto y edad media del ensamble base."""
    ser = pd.read_parquet(FR.RES / "f2b_base" / "series.parquet",
                          columns=["point", "rep", "t", "N", "births", "S_adult", "mean_age"])
    pts = [(FR.pidx("base", FR.LAB["C1"]), "C1", CAT[0], "-"),
           (FR.pidx("base", FR.LAB["C1p"]), "C1$'$", CAT[3], (0, (5, 1, 1, 1))),
           (FR.pidx("base", FR.LAB["C0"]), "C0", CAT[1], "--"),
           (FR.pidx("base", FR.LAB["a0"]), "C1, $\\alpha=0$", CAT[2], "-.")]
    K = 8 * 900
    tmax = 260
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.0))
    for pt, name, col, ls in pts:
        g = ser[(ser.point == pt) & (ser.t <= tmax)]
        q = g.groupby("t").agg(N50=("N", "median"), N10=("N", lambda x: x.quantile(0.1)),
                               N90=("N", lambda x: x.quantile(0.9)), B50=("births", "median"))
        nrep = g.rep.nunique()
        alive = g[g.N > 0].groupby("t")["rep"].nunique().reindex(q.index).fillna(0) / nrep
        gl = g[g.N > 0].groupby("t").agg(S50=("S_adult", "median"), A50=("mean_age", "median")).reindex(q.index)
        q["S50"], q["A50"] = gl.S50, gl.A50
        q.loc[alive < 0.5, ["S50", "A50"]] = np.nan
        t = q.index.to_numpy()
        axs[0].plot(t, q.N50 / K, color=col, ls=ls, label=name)
        if name == "C1":
            axs[0].fill_between(t, q.N10 / K, q.N90 / K, color=col, alpha=0.18, lw=0)
            for r in sorted(g.rep.unique())[:12]:
                gg = g[g.rep == r]
                axs[0].plot(gg.t, gg.N / K, color=col, lw=0.3, alpha=0.35)
        axs[1].plot(t, q.B50, color=col, ls=ls)
        axs[2].plot(t, q.S50, color=col, ls=ls)
        axs[3].plot(t, q.A50, color=col, ls=ls)
    axs[0].set_ylabel("$N/K_\\gamma$"); axs[0].set_title("population", loc="left")
    axs[1].set_ylabel("births per week"); axs[1].set_title("births (median)", loc="left"); axs[1].set_xlim(0, 80)
    f2c = ROOT / "results" / "fisico_v3" / "fig2c_series.csv"   # medias sin dilución (analiza_gamma.py)
    S2 = pd.read_csv(f2c)
    S2 = S2[S2.t <= tmax]
    axs[2].plot(S2.t, S2.s_aw_med, color=CAT[0], ls=(0, (1, 1)), lw=1.1, label="C1, age $\\geq a_w$")
    axs[2].plot(S2.t, S2.s_f0_med, color=CAT[0], ls="--", lw=0.9, label="C1, founders")
    axs[2].legend(loc="upper right", fontsize=6.2, handlelength=2.2)
    axs[2].set_ylim(0, 1.02); axs[2].set_yticks([0, 0.2, 0.5, 1.0])
    axs[2].set_ylabel("$\\bar s$ (adults)"); axs[2].set_title("social state", loc="left")
    axs[2].axhline(0.5, color=MUTED, lw=0.6, ls=":"); axs[2].axhline(0.2, color=MUTED, lw=0.6, ls=":")
    axs[2].set_xlim(0, 80)
    axs[3].set_ylabel("mean age (weeks)"); axs[3].set_title("mean age", loc="left")
    tt = np.array([40, 200]); axs[3].plot(tt, tt - 10, color=MUTED, lw=0.8, ls=":")
    axs[3].text(150, 115, "slope 1", color=INK2, fontsize=6.8)
    for i, ax in enumerate(axs):
        ax.set_xlabel("week"); panel(ax, "abcd"[i])
        if i not in (1, 2):
            ax.set_xlim(0, tmax)
    axs[0].legend(loc="center right", bbox_to_anchor=(1.02, 0.58), handlelength=2.2, fontsize=6.6)
    fig.tight_layout(w_pad=0.7)
    save(fig, "fig_pre_trajectories.pdf")


# ============================================================================= 2. supervivencia y ablaciones (columna)
ABL_ROWS = [("abl1", "C1", "C1"), ("abl1", "C1 -R1", "$-$R1 (window)"),
            ("abl1", "C1 -R2", "$-$R2 (learning)"), ("abl1", "C1 -AV", "$-$AV (aversion)"),
            ("abl1", "C1 -R7", "$-$R7 (refuges)"), ("abl2", "C1 -R1 -R2", "$-$R1 $-$R2"),
            ("abl2", "C1 -R1 -R2 -AV -R7", "no rules"), ("abl2", "C1 gamma=0 (sin hacinamiento)", "$\\gamma=0$"),
            ("abl1", "C1 w=0 rec=0.15", "$w=0$, $r=0.15$"), ("abl1", "C1 -pref (refugios sin preferencia)", "$\\pi=0$"),
            ("abl1", "C1 -hard (capacidad blanda)", "soft capacity")]
ESS = ["O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p"]


def fig_rules():
    """(a) Kaplan--Meier de los candidatos y controles; (b) O17p de las ablaciones con ambos evaluadores y el
    criterio esencial que falla."""
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.9), gridspec_kw=dict(height_ratios=[1.0, 1.15]))
    a = axs[0]
    groups = [("C1", FR.arm("base", "C1")), ("C1$'$", FR.arm("base", "C1p")), ("C0", FR.arm("base", "C0")),
              ("C2", FR.arm("c1c2", "C2")), ("short-lived", FR.arm("ref", "notebook")),
              ("long-lived", FR.arm("ref", "longlived"))]
    for i, (lab, g) in enumerate(groups):
        km, t, ev = FR.km_of(g)
        a.step(km.t, km.S, where="post", color=CAT[i], ls=LS[i], label=lab)
        if i in (0, 4):
            a.fill_between(km.t, km.lo, km.hi, step="post", color=CAT[i], alpha=0.15, lw=0)
    a.set_xlim(0, 1000); a.set_ylim(-0.02, 1.02); a.set_xlabel("week"); a.set_ylabel("$P$(colony alive)")
    a.set_title("survival of colonies", loc="left")
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.97), fontsize=6.6, handlelength=2.4, ncol=2, columnspacing=1.0)
    panel(a, "a")
    a = axs[1]
    T = {"abl1": FR.tab("abl1"), "abl2": FR.tab("abl2")}
    Tv = {"abl1": FR.ptab("abl1", "v1_"), "abl2": FR.ptab("abl2", "v1_")}
    rows = []
    for st, lab, name in ABL_ROWS:
        r = T[st][T[st].label == lab].iloc[0]
        rv = Tv[st][Tv[st].label == lab].iloc[0]
        fe = {c: float(r[f"f_{c}"]) for c in ESS}
        fe_ok = {c: (0.0 if not np.isfinite(v) else v) for c, v in fe.items()}   # NA cuenta como fallo
        worst = min(fe_ok, key=fe_ok.get)
        rows.append(dict(name=name, n=int(r["n"]), f=float(r.f_O17p), lo=float(r.lo_O17p), hi=float(r.hi_O17p),
                         fv1=float(rv.f_O17p), worst=worst, fworst=fe_ok[worst]))
    D = pd.DataFrame(rows)
    y = np.arange(len(D))[::-1]
    a.errorbar(D.f, y, xerr=FR.xe(D.f, D.lo, D.hi), fmt="o", ms=4, color=CAT[0], ecolor=CAT[0], elinewidth=1,
               capsize=0, label="primary evaluator")
    a.plot(D.fv1, y, "D", ms=3.6, mfc="white", mec=CAT[1], mew=1.0, ls="none", label="preregistered evaluator")
    a.axvline(0.8, color=MUTED, lw=0.7, ls=":")
    for yy, (_, r) in zip(y, D.iterrows()):
        if r.f < 0.95:
            a.text(1.04, yy, f"{r.worst} {r.fworst:.2f}", va="center", ha="left", fontsize=6.4, color=INK2)
    a.text(1.04, y[0], "fails", va="center", ha="left", fontsize=6.4, color=INK2, style="italic")
    a.set_yticks(y); a.set_yticklabels(list(D.name), fontsize=7)
    a.set_xlim(-0.03, 1.03); a.set_xlabel("O17p, fraction of runs (95% Wilson CI)")
    a.axhline(y[4] - 0.5, color=MUTED, lw=0.5); a.axhline(y[7] - 0.5, color=MUTED, lw=0.5)
    a.legend(loc="lower right", fontsize=6.4, handletextpad=0.3, bbox_to_anchor=(0.97, 0.0))
    a.grid(axis="y", visible=False)
    a.set_title("rule ablations of C1", loc="left")
    panel(a, "b")
    fig.tight_layout(h_pad=1.2)
    save(fig, "fig_pre_rules.pdf")


# ============================================================================= 3. transplantes (columna)
def fig_transplant():
    """(a) animales de fase D a tres tiempos de muestreo y de fase B; (b) animales de estado intermedio."""
    A = FR.trans_long()
    arms = FR.TR_ARMS
    fig, axs = plt.subplots(2, 1, figsize=(COL, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    y0 = np.arange(len(arms))[::-1]

    def series(ph):
        f = np.array([A[(A.brazo == l) & (A.fase == ph)].P_T.iloc[0] for l, _ in arms], float)
        lo = np.array([A[(A.brazo == l) & (A.fase == ph)].lo.iloc[0] for l, _ in arms], float)
        hi = np.array([A[(A.brazo == l) & (A.fase == ph)].hi.iloc[0] for l, _ in arms], float)
        return f, lo, hi

    a = axs[0]
    phases = [("B", "phase B (growth)"), ("D", "phase D, $1.75\\,t'_{\\rm pk}$"),
              ("Darg", "phase D, $1.75\\,t_{\\rm pk}$ (arg max)"), ("Dfix", "week 95 (all arms)")]
    for j, (ph, lab) in enumerate(phases):
        f, lo, hi = series(ph)
        a.errorbar(f, y0 + (1.5 - j) * 0.2, xerr=FR.xe(f, lo, hi), fmt=MK[j], ms=3.8, color=CAT[j], elinewidth=0.9,
                   capsize=0, label=lab)
    a.set_yticks(y0); a.set_yticklabels([n for _, n in arms])
    a.axvline(0.1, color=MUTED, lw=0.6, ls=":"); a.axvline(0.5, color=MUTED, lw=0.6, ls=":")
    a.set_title("phase-D animals ($s\\approx0$) and phase-B controls", loc="left")
    a.tick_params(labelbottom=True)
    a.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, handletextpad=0.3, columnspacing=1.0)
    a.grid(axis="y", visible=False)
    panel(a, "a")
    a = axs[1]
    phases = [("BAND", "band animals only"), ("BAND_MIX_M", "band males + healthy females"),
              ("BAND_MIX_F", "band females + healthy males")]
    for j, (ph, lab) in enumerate(phases):
        f, lo, hi = series(ph)
        a.errorbar(f, y0 + (1 - j) * 0.24, xerr=FR.xe(f, lo, hi), fmt=MK[j], ms=3.8, color=CAT[j + 4], elinewidth=0.9,
                   capsize=0, label=lab)
    a.set_yticks(y0); a.set_yticklabels([n for _, n in arms])
    a.set_xlim(-0.02, 1.02); a.set_xlabel("$P_T$: probability of founding a colony (95% Wilson CI; 60 trials)")
    a.set_title("adults of intermediate state, $0.2\\leq s\\leq0.5$", loc="left")
    a.legend(fontsize=6.4, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, handletextpad=0.3, columnspacing=1.0)
    a.grid(axis="y", visible=False)
    panel(a, "b")
    fig.tight_layout(h_pad=2.6)
    save(fig, "fig_pre_transplant.pdf")


# ============================================================================= 4. robustez (figure*)
def fig_robustness():
    """Como fig_res_rob: (a) O17p contra mu_bg; (b) variantes y condición inicial; (c) tamaño del sistema.
    Cifras de results_numbers.json (robmu, rob, size), generadas por figs_results.py de results_v2/."""
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.55), gridspec_kw=dict(width_ratios=[1.0, 1.4, 1.0]))
    M = pd.DataFrame(NUM["robmu"])
    a = axs[0]
    for j, (cand, name) in enumerate((("C1", "C1"), ("C1p", "C1$'$"))):
        d = M[M.candidato == cand].sort_values("mu_bg")
        x = d.mu_bg.to_numpy(float) * 1e3
        a.errorbar(x, d.O17p, yerr=FR.xe(d.O17p, d.lo, d.hi), fmt=MK[j] + "-", ms=3.8, color=CAT[j], lw=1.1,
                   elinewidth=0.7, capsize=0, label=f"{name}, primary")
        a.plot(x, d.O17p_v1, MK[j], ls="--", mfc="white", mec=CAT[j], color=CAT[j], ms=3.5, lw=0.9,
               label=f"{name}, preregistered evaluator")
    a.axhline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("background mortality $\\mu_{\\rm bg}$ ($10^{-3}$ per week)"); a.set_ylabel("O17p")
    a.set_ylim(-0.03, 1.03); a.set_title("background mortality", loc="left")
    a.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=2.0)
    panel(a, "a")
    a = axs[1]
    R = pd.DataFrame(NUM["rob"])
    names = {"C1 R7 legacy (v1)": "single-pass refuge admission", "C1 mu_bg=0.003": "$\\mu_{\\rm bg}=0.003$",
             "C1 mu_bg=0.008": "$\\mu_{\\rm bg}=0.008$", "C1 nido=3": "nest period 3 wk", "C1 async": "asynchronous update",
             "C1 async blanda": "async., soft capacity", "C1 s0=0.25": "pups born at $s=0.25$",
             "C1 s0=0.5": "pups born at $s=0.5$", "C1 CI edades 4-52": "initial ages U{4..52}",
             "C1 CI edades 4-80": "initial ages U{4..80}"}
    order = list(names)
    y = np.arange(len(order))[::-1]
    for j, cand in enumerate(("C1", "C1p")):
        d = R[R.candidato == cand].copy()
        d["key"] = d.brazo.str.replace("C1p ", "C1 ", regex=False)
        d = d.set_index("key").reindex(order)
        yy = y + (0.18 if j == 0 else -0.18)
        a.errorbar(d.O17p, yy, xerr=FR.xe(d.O17p, d.lo, d.hi), fmt=MK[j], ms=3.8, color=CAT[j], elinewidth=0.9,
                   capsize=0, label=("C1" if j == 0 else "C1$'$") + ", primary")
        a.plot(d.O17p_v1, yy, MK[j], mfc="white", mec=CAT[j], ms=3.5, ls="none",
               label=("C1" if j == 0 else "C1$'$") + ", preregistered evaluator")
    for j, cand in enumerate(("C1", "C1p")):
        r0 = FR.fr(FR.arm("base", cand), "O17p")["f"]
        a.axvline(r0, color=CAT[j], lw=0.7, ls=LS[1 + j], alpha=0.8)
    a.axvline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_yticks(y); a.set_yticklabels([names[k] for k in order], fontsize=7); a.set_xlim(-0.03, 1.03)
    a.set_xlabel("O17p (95% Wilson CI; $n=200$)"); a.set_title("variants and initial condition", loc="left")
    a.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2); a.grid(axis="y", visible=False)
    panel(a, "b")
    a = axs[2]
    S = pd.DataFrame(NUM["size"]["tabla"])
    a.errorbar(S.L, S.spread, yerr=FR.xe(S.spread, S.spread_lo, S.spread_hi), fmt="o-", color=CAT[0], ms=3.8, lw=1.0,
               elinewidth=0.7, capsize=0, label="KM spread of $T_{\\rm ext}$")
    a.plot(S.L, S.t_last_iqr / S.t_last_med, "s--", color=CAT[1], ms=3.5, lw=0.9, label="IQR/median of $t_{\\rm last}$")
    a.plot(S.L, S.O17p, "^:", color=CAT[2], ms=3.8, lw=0.9, label="O17p")
    a.set_xticks([15, 20, 30, 45, 60]); a.set_xlim(10, 65)
    a.set_xlabel("lattice side $L$ (density $N_0/L^2=0.44$)"); a.set_ylim(0, 1.05)
    a.set_title("system size", loc="left")
    a.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    panel(a, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_pre_robustness.pdf")


# ============================================================================= 5. región factible y Sobol (columna)
def fig_global():
    """(a) índices de Sobol totales de O17p (11 factores) con S1 y piso de ruido; (b) fracción Calhoun-like de los
    hipercubos en subconjuntos anidados (post hoc), tres evaluadores."""
    so = pd.read_csv(FR.RES / "f2b_sobol" / "analisis" / "sobol.csv")
    fig, axs = plt.subplots(2, 1, figsize=(COL, 3.9), gridspec_kw=dict(height_ratios=[1.1, 1.0]))
    a = axs[0]
    d = so[so.metric == "f_PRIMARIO"].set_index("name").sort_values("ST")
    yy = np.arange(len(d))
    rel = d["relevante"].astype(str) == "True"
    a.barh(yy, d.ST, height=0.62, color=[CAT[0] if r else "#b7d3f6" for r in rel], label="$S_T$ (total)")
    a.errorbar(d.ST, yy, xerr=d.ST_conf, fmt="none", ecolor=INK2, elinewidth=0.7)
    a.plot(d.S1.clip(lower=0), yy, "o", ms=3.6, mfc="white", mec=INK, mew=0.8, ls="none", label="$S_1$ (first order)")
    a.plot(d.ST_ruido, yy, "|", color=CAT[7], ms=9, mew=1.6, ls="none", label="noise floor")
    a.set_yticks(yy); a.set_yticklabels([FR.FNAME[n] for n in d.index])
    a.set_xlim(0, 0.72); a.set_xlabel("Sobol' index of O17p")
    a.set_title("variance decomposition (11 factors)", loc="left"); a.grid(axis="y", visible=False)
    a.legend(loc="lower right", fontsize=6.4)
    panel(a, "a")
    a = axs[1]
    keys = ["all", "cR<=2", "+mc<=10", "+fRcR>=0.3", "+pi>=100"]
    for j, (pref, lab) in enumerate((("primary", "primary evaluator"), ("v1_", "preregistered evaluator"),
                                     ("mc_", "crowded class at $m>m_c$"))):
        dec = NUM[f"lhs_decomp_{pref}"]
        f = np.array([dec[k]["f"] for k in keys]); lo = np.array([dec[k]["lo"] for k in keys])
        hi = np.array([dec[k]["hi"] for k in keys])
        x = np.arange(len(keys)) + (j - 1) * 0.24
        a.bar(x, f, width=0.22, color=CAT[j], label=lab, hatch=[None, "//", ".."][j], edgecolor="white", lw=0.4)
        a.errorbar(x, f, yerr=FR.xe(f, lo, hi), fmt="none", ecolor=INK2, elinewidth=0.7)
    a.set_xticks(range(5))
    ns = [NUM["lhs_decomp_primary"][k]["n"] for k in keys]
    a.set_xticklabels([f"{t}\n($n$={n})" for t, n in zip(
        ["all", "$c_R\\leq2$", "$+\\,m_c\\leq10$", "$+\\,f_Rc_R\\geq0.3$", "$+\\,\\pi\\geq100$"], ns)], fontsize=6.8)
    a.set_ylabel("fraction of points\nwith $k/n\\geq0.8$"); a.set_title("Calhoun-like region of the hypercubes", loc="left")
    a.legend(fontsize=6.4, loc="upper left"); a.grid(axis="x", visible=False)
    a.set_ylim(0, 0.72)
    panel(a, "b")
    fig.tight_layout(h_pad=1.4)
    save(fig, "fig_pre_global.pdf")


# ============================================================================= tablas del suplemento
SUPP_TABLES = ["tab_res_production_full", "tab_res_production_si", "tab_res_null_full", "tab_res_null_v1",
               "tab_res_Qcmp", "tab_res_rescoring", "tab_res_rob", "tab_res_size", "tab_res_trans", "tab_res_ensembles"]


def copy_tables():
    """Copia a manuscript/pre/tables/ las tablas generadas por figs_results.py que el suplemento \\input-ea."""
    for t in SUPP_TABLES:
        src = ROOT / "manuscript" / "extended" / "tables" / f"{t}.tex"
        shutil.copyfile(src, TABS / f"{t}.tex")
        print("tab", TABS / f"{t}.tex")


if __name__ == "__main__":
    for fn in (fig_trajectories, fig_rules, fig_transplant, fig_robustness, fig_global, copy_tables):
        fn()
