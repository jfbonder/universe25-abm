"""Figuras y tablas de la sección Results de la versión extendida a partir de results_v2/f2b_*/.

    # desde la raíz del repositorio (o: make figs)
    python3 scripts/figures/figs_results.py

Lee sólo salidas ya analizadas (summary.csv, series.parquet, analisis/*.csv, f2b_final/*.csv y f2b_final/v2/*.csv);
no corre simulaciones. Escribe:
  manuscript/extended/figs/fig_res_*.pdf     figuras vectoriales
  manuscript/extended/tables/tab_res_*.tex   tablas que \\input-ean sections/results.tex y appendix_results.tex
  results_v2/results_numbers.json      números citados en el texto, cada uno con su ensamble;
                                           incluye O11 a la edad a_w (results_v2/o11_aw/) y Γ (results/fisico_v3/)

Ensamble de referencia único: `base` (C1, C1', C0, alpha=0; n = 200, T = 600). `c1c2` es
la réplica independiente de C1 y el ensamble de C2. Los brazos se buscan por etiqueta (labels.json), nunca por índice.
Variantes del evaluador (f2b_model_v2): obj_<pref>_* con pref en v1, ta, te, mc, l5, o4a; se aplican con
f2b.apply_variant sobre una copia del summary.
Paleta: categórica de referencia (orden fijo) y rampa secuencial azul del skill de visualización; identidad nunca
sólo por color (estilo de línea o marcador como codificación secundaria).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]       # raíz del repositorio
for _p in (ROOT / "scripts" / "production", ROOT / "scripts" / "analysis", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import survival, f2b                   # noqa: E402
from pipeline import v2 as v2mod                     # noqa: E402

RES = Path(os.environ.get("F2B_RES", "results_v2"))
RES = RES if RES.is_absolute() else ROOT / RES
FIGS = ROOT / "manuscript" / "extended" / "figs"
TABS = ROOT / "manuscript" / "extended" / "tables"
NUMS = ROOT / "results_v2" / "results_numbers.json"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- estilo
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + SEQ)
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W2 = 6.9   # ancho de página (pulgadas)

NUM: dict = {"_fuente": str(RES.relative_to(ROOT)) if RES.is_relative_to(ROOT) else str(RES)}

# etiquetas del ensamble de referencia y de los demás brazos (labels.json)
LAB = {"C1": "C1", "C1p": "C1p", "C0": "C0 (control)", "a0": "C1 alpha=0 (control O14)", "C2": "C2",
       "notebook": "notebook alpha=1.4", "longlived": "longevo",
       "null_g0": "C1 gamma=0 (sin hacinamiento)", "null_R2": "C1 -R2 (w=1, sin aprendizaje social)",
       "null_long": "longevo (notebook a_max=120, CI de C1)", "null_none": "C1 sin reglas (-R1 -R2 -AV -R7)"}
VARS = [("primary", ""), ("v1", "v1_"), ("argmax", "ta_"), ("plateau end", "te_"), ("$m>m_c$", "mc_"),
        ("O5 on $t_{\\rm last}$", "l5_"), ("O4 v1", "o4a_")]


# ----------------------------------------------------------------------------- utilidades
def save(fig, name):
    fig.savefig(FIGS / name)
    plt.close(fig)
    print("fig", FIGS / name)


def write_tab(name, rows):
    (TABS / name).write_text("\n".join(rows) + "\n\\bottomrule\n")
    print("tab", TABS / name)


def panel(ax, letter):
    t = ax.get_title(loc="left")
    ax.set_title(f"({letter}) {t}", loc="left", color=INK)


def labels(stage):
    return json.load(open(RES / f"f2b_{stage}" / "labels.json"))


def pidx(stage, label):
    return labels(stage).index(label)


def tab(stage):
    return pd.read_csv(RES / f"f2b_{stage}" / "analisis" / "tabla_puntos.csv")


_RUNS: dict = {}


def run(stage):
    if stage not in _RUNS:
        _RUNS[stage] = f2b.load(RES / f"f2b_{stage}")
    return _RUNS[stage]


def summ(stage):
    return run(stage)["summary"]


def summ_var(stage, pref):
    """Copia del summary con la variante del evaluador `pref` ('' = primario) aplicada sobre obj_*."""
    s = summ(stage).copy()
    pref = pref.rstrip("_")
    return f2b.apply_variant(s, pref) if pref else s


def ptab(stage, pref="", primary="O17p"):
    """Tabla por punto con la variante `pref` del evaluador."""
    r = dict(run(stage))
    r["summary"] = summ_var(stage, pref)
    return f2b.point_table(r, primary)


def arm(stage, key, pref=""):
    """Corridas de un brazo (clave de LAB o etiqueta literal), con la variante `pref`."""
    s = summ_var(stage, pref)
    return s[s.point == pidx(stage, LAB.get(key, key))]


def km_of(g):
    ev = g["extinct"].to_numpy(bool)
    t = np.where(ev, g["t_ext"], g["t_end"]).astype(float)
    km = survival.kaplan_meier(t, ev)
    if t.max() > km.t.max():          # censuras después del último evento: el escalón sigue hasta T
        last = km.iloc[-1].copy(); last["t"] = t.max()
        km = pd.concat([km, last.to_frame().T], ignore_index=True)
    return km, t, ev


def wilson(k, n):
    return survival.wilson(int(k), int(n))


def fr(g, code):
    """Fracción PASS (todas las corridas para O17p/O17_run; evaluables para los demás) con Wilson."""
    return f2b.frac(g[f"obj_{code}"], code)


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run_max = 0.0
    for r, i in enumerate(o):
        run_max = max(run_max, (m - r) * p[i])
        adj[i] = min(1.0, run_max)
    return adj


def p_tex(p, level=0.05):
    """p con dos cifras significativas; si está a menos de 0,01 del nivel, con cuatro decimales."""
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-16:
        return "$<10^{-16}$"
    if p < 1e-3:
        e = int(math.floor(math.log10(p))); m = p / 10 ** e
        return f"${m:.1f}\\times10^{{{e}}}$"
    if abs(p - level) < 0.01:
        return f"{p:.4f}"
    if p < 0.01:
        return f"{p:.2g}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def xe(f, lo, hi):
    """Barras de error no negativas (Wilson devuelve cotas 1e-18 para k = 0)."""
    f, lo, hi = (np.asarray(x, float) for x in (f, lo, hi))
    return [np.clip(f - lo, 0, None), np.clip(hi - f, 0, None)]


def fci(f, lo, hi, nd=2):
    if f is None or not np.isfinite(f):
        return "--"
    return f"{f:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def heat(ax, P, title, fmt="{:.2f}", vmin=0, vmax=1, xlabel="", ylabel="", xt=None, yt=None, cmap=CMAP):
    A = P.to_numpy(float)
    im = ax.imshow(A, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(xt or [f"{v:g}" for v in P.columns])
    ax.set_yticks(range(A.shape[0])); ax.set_yticklabels(yt or [f"{v:g}" for v in P.index])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    for (i, j), v in np.ndenumerate(A):
        if np.isfinite(v):
            rel = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.6,
                    color="white" if rel > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def spread_km(g, B=500):
    """(Q75 − Q25)/Q50 de T_ext con Kaplan–Meier e IC bootstrap."""
    ev = g["extinct"].to_numpy(bool)
    t = np.where(ev, g["t_ext"], g["t_end"]).astype(float)
    est, lo, hi, _ = v2mod.km_spread(t, ev, B=B)
    te = t[ev]
    return dict(iqr_med=float(est), lo=float(lo), hi=float(hi), cv=float(np.std(te, ddof=1) / np.mean(te)) if te.size > 1 else np.nan,
                n_ext=int(ev.sum()), max=float(te.max()) if te.size else np.nan)


# ============================================================================= 1. ensambles de producción
def fig_trajectories():
    ser = pd.read_parquet(RES / "f2b_base" / "series.parquet",
                          columns=["point", "rep", "t", "N", "births", "S_mean", "S_adult", "mean_age"])
    pts = [(pidx("base", LAB["C1"]), "C1", CAT[0], "-"), (pidx("base", LAB["C1p"]), "C1$'$", CAT[3], (0, (5, 1, 1, 1))),
           (pidx("base", LAB["C0"]), "C0", CAT[1], "--"), (pidx("base", LAB["a0"]), "C1, $\\alpha=0$", CAT[2], "-.")]
    K = 8 * 900
    tmax = 260
    fig, axs = plt.subplots(1, 4, figsize=(W2, 1.85))
    for pt, name, col, ls in pts:
        g = ser[(ser.point == pt) & (ser.t <= tmax)]
        q = g.groupby("t").agg(N50=("N", "median"), N10=("N", lambda x: x.quantile(0.1)), N90=("N", lambda x: x.quantile(0.9)),
                               B50=("births", "median"))
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
    # Escala lineal, niveles 0.2 y 0.5 marcados, y las medias sin dilución de C1 (fundadores y animales con
    # edad >= a_w) junto a la media adulta (edad >= 6). Serie de scripts/analysis/analiza_gamma.py.
    f2c = ROOT / "results" / "fisico_v3" / "fig2c_series.csv"
    if not f2c.exists():   # insumo versionado: si falta, la figura falla (y results_numbers.json no se reescribe)
        raise FileNotFoundError(f"{f2c.relative_to(ROOT)} (de scripts/analysis/analiza_gamma.py)")
    S2 = pd.read_csv(f2c)
    S2 = S2[S2.t <= tmax]
    axs[2].plot(S2.t, S2.s_aw_med, color=CAT[0], ls=(0, (1, 1)), lw=1.1, label="C1, age $\\geq a_w$")
    axs[2].plot(S2.t, S2.s_f0_med, color=CAT[0], ls="--", lw=0.9, label="C1, founders")
    axs[2].legend(loc="upper right", fontsize=5.4, handlelength=2.2)
    axs[2].set_ylim(0, 1.02); axs[2].set_yticks([0, 0.2, 0.5, 1.0])
    axs[2].set_ylabel("$\\bar s$"); axs[2].set_title("social state", loc="left")
    axs[2].axhline(0.5, color=MUTED, lw=0.6, ls=":"); axs[2].axhline(0.2, color=MUTED, lw=0.6, ls=":")
    axs[3].set_ylabel("mean age (weeks)"); axs[3].set_title("mean age", loc="left")
    tt = np.array([40, 200]); axs[3].plot(tt, tt - 10, color=MUTED, lw=0.8, ls=":")
    axs[3].text(150, 120, "slope 1", color=INK2, fontsize=6.3)
    axs[2].set_xlim(0, 80)      # la caída de s ocurre en las primeras 40 semanas; misma ventana que el panel (b)
    for i, ax in enumerate(axs):
        ax.set_xlabel("week"); panel(ax, "abcd"[i])
        if i not in (1, 2):
            ax.set_xlim(0, tmax)
    axs[0].legend(loc="upper right", handlelength=2.2, fontsize=5.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_res_trajectories.pdf")
    g = ser[ser.point == pidx("base", LAB["C1"])]
    lastb = g[g.births > 0].groupby("rep").t.max()
    NUM["C1_t_lastbirth_median"] = dict(valor=float(lastb.median()), ensamble="base:C1")
    sa = g[g.S_adult < 0.5].groupby("rep").t.min()
    NUM["C1_t_Sadult_below_05_median"] = dict(valor=float(sa.median()), q25=float(sa.quantile(.25)), q75=float(sa.quantile(.75)))


def trans_long():
    d = RES / "f2b_trans" / "analisis" / "tabla_brazos_fases.csv"
    return pd.read_csv(d)


def table_production():
    """tab_res_production.tex (texto): esenciales, O17p con IC, transplantes, antipatrones y magnitudes;
    tab_res_production_si.tex (apéndice): deseables y el resto. Todo del ensamble de referencia `base`
    (C1, C1', C0, alpha=0), de `c1c2` (C2) y de `ref` (notebook, longevo)."""
    B, C, R = tab("base"), tab("c1c2"), tab("ref")
    cols = [("C1", "base", LAB["C1"]), ("C1$'$", "base", LAB["C1p"]), ("C0", "base", LAB["C0"]),
            ("C1, $\\alpha{=}0$", "base", LAB["a0"]), ("C2", "c1c2", LAB["C2"]), ("notebook", "ref", LAB["notebook"]),
            ("long-lived", "ref", LAB["longlived"])]
    T = {"base": B, "c1c2": C, "ref": R}
    rowsT = {nm: T[st][T[st].point == pidx(st, lab)].iloc[0] for nm, st, lab in cols}
    groups = {nm: arm(st, lab) for nm, st, lab in cols}
    TR = trans_long()

    def cell(r, code, g=None):
        if f"n_{code}" not in r and g is not None and f"obj_{code}" in g:
            d = fr(g, code)
            return "--" if not np.isfinite(d["f"]) else f"{d['f']:.2f}"
        if f"n_{code}" not in r or r[f"n_{code}"] == 0 or not np.isfinite(r.get(f"f_{code}", np.nan)):
            return "--"
        return f"{r[f'f_{code}']:.2f}"

    def rowp(code, desc):
        return f"{ {'O17_run': 'O17'}.get(code, code)} & {desc} & " + " & ".join(cell(rowsT[nm], code, groups[nm]) for nm, _, _ in cols) + " \\\\"

    ess = [("O4", "peak below capacity, free space"), ("O5", "natality ends at high $N$"), ("O6", "no rebound"),
           ("O7", "extinction"), ("O10", "adult deterioration before peak"), ("O11", "late cohorts fail"),
           ("O13", "two deteriorated classes"), ("O13p", "\\quad persistent isolated class"), ("O17_run", "Calhoun-like run (O17)")]
    out = [rowp(*x) for x in ess]
    out.append("\\midrule")
    out.append(rowp("O17p", "\\textbf{primary outcome}"))
    out.append(" & \\quad 95\\% Wilson interval & " + " & ".join(f"{rowsT[nm]['lo_O17p']:.2f}--{rowsT[nm]['hi_O17p']:.2f}" for nm, _, _ in cols) + " \\\\")
    out.append("\\midrule")
    main_rows = list(out)
    trarm = {"C1": "C1", "C1$'$": "C1p", "C0": "C0 (control)", "C2": "C2 (contraste)"}

    def ptr(nm, ph):
        if nm not in trarm:
            return "--"
        r = TR[(TR.brazo == trarm[nm]) & (TR.fase == ph)]
        return "--" if not len(r) else f"{int(r.k.iloc[0])}/{int(r.n.iloc[0])}"

    def pmix(nm, phs):
        if nm not in trarm:
            return "--"
        r = TR[(TR.brazo == trarm[nm]) & (TR.fase.isin(phs))]
        return "--" if not len(r) else f"{int(r.k.sum())}/{int(r.n.sum())}"
    out.append("O12 & transplants found: phase B & " + " & ".join(ptr(nm, "B") for nm, _, _ in cols) + " \\\\")
    out.append(" & \\quad phase D ($1.75\\,t'_{\\rm pk}$) & " + " & ".join(ptr(nm, "D") for nm, _, _ in cols) + " \\\\")
    out.append(" & \\quad week 95, all arms & " + " & ".join(ptr(nm, "Dfix") for nm, _, _ in cols) + " \\\\")
    out.append("O12b$^\\dagger$ & \\quad D + healthy partners & " + " & ".join(pmix(nm, ["MIX_M", "MIX_F"]) for nm, _, _ in cols) + " \\\\")

    def trig(r):
        tr = [a for a in ("A1", "A2", "A3", "A4") if r.get(f"trig_{a}", 0) > 0]
        return "none" if not tr else ", ".join(f"{a}: {r[f'trig_{a}']:.2f}" for a in tr)
    out.append("A1--A4 & anti-patterns triggered & " + " & ".join(trig(rowsT[nm]) for nm, _, _ in cols) + " \\\\")
    out.append("\\midrule")

    def mag(col, nd=2):
        return " & ".join(f"{rowsT[nm][col]:.{nd}f}" if np.isfinite(rowsT[nm][col]) else "--" for nm, _, _ in cols)
    SP = {nm: spread_km(groups[nm]) for nm, _, _ in cols}
    out.append("& $\\rho_{\\rm pk}=N_{\\rm pk}/K_\\gamma$ & " + mag("x_rho_pk_mean") + " \\\\")
    out.append("& empty cells at peak, $f_0$ & " + mag("x_f0_pk_mean") + " \\\\")
    out.append("& persistent isolated, $\\Phi^{\\rm pers}_{\\rm BO}$ & " + mag("x_BO4_mean") + " \\\\")
    out.append("& median $T_{\\rm ext}$ (weeks, KM) & " + " & ".join("--" if not np.isfinite(rowsT[nm]["t_ext_med"]) else f"{rowsT[nm]['t_ext_med']:.0f}" for nm, _, _ in cols) + " \\\\")
    out.append("& spread of $T_{\\rm ext}$ (KM IQR/median) & " + " & ".join("--" if not np.isfinite(SP[nm]["iqr_med"]) else f"{SP[nm]['iqr_med']:.2f}" for nm, _, _ in cols) + " \\\\")
    (TABS / "tab_res_production_full.tex").write_text("\n".join(out) + "\n\\midrule\n")
    # versión del texto principal: sin transplantes (tab:res-trans) ni antipatrones (texto); magnitudes al final
    i_mag = next(i for i, r in enumerate(out) if r.startswith("& $\\rho_{\\rm pk}"))
    write_tab("tab_res_production.tex", main_rows + out[i_mag:])

    des = [("O8", "senescent decline"), ("O8b", "slow initial decline"), ("O9", "old, female-biased remnant"),
           ("O10b", "functional growth phase"),
           ("O11b", "few late-born females conceive"), ("O13b", "isolated class later-born"),
           ("O14", "aggregation with free space"), ("O15", "infant mortality $\\to$ 1"),
           ("O15b", "conceptions without viable young")]
    out = [rowp(*x) for x in des if f"f_{x[0]}" in B or x[0] == "O10b"]

    def pm(g):
        v = g["obj_O13b"].astype(str)
        k, n = int(v.isin(["PASS", "MARG"]).sum()), int(v.isin(["PASS", "MARG", "FAIL"]).sum())
        return f"{k / n:.2f}" if n else "--"
    out.append("O13b & \\quad PASS or MARGINAL & " + " & ".join(pm(groups[nm]) for nm, _, _ in cols) + " \\\\")
    out.append("\\midrule")
    out.append("& $t'_{\\rm pk}$ (weeks, median) & " + " & ".join(f"{pd.to_numeric(groups[nm].x_tpk2, errors='coerce').median():.0f}" for nm, _, _ in cols) + " \\\\")
    out.append("& $t_{\\rm pk}=\\arg\\max N$ (median) & " + " & ".join(f"{pd.to_numeric(groups[nm].t_peak, errors='coerce').median():.0f}" for nm, _, _ in cols) + " \\\\")
    out.append("& plateau length (median) & " + " & ".join(f"{pd.to_numeric(groups[nm].x_plateau, errors='coerce').median():.0f}" for nm, _, _ in cols) + " \\\\")
    out.append("& $t_{B99}$ (median) & " + " & ".join(f"{pd.to_numeric(groups[nm].x_tB99, errors='coerce').median():.0f}" for nm, _, _ in cols) + " \\\\")

    def kmci(r):
        if not np.isfinite(r["t_ext_med"]):
            return "--"
        hi = f"{r['t_ext_med_hi']:.0f}" if np.isfinite(r["t_ext_med_hi"]) else "$\\infty$"
        return f"{r['t_ext_med_lo']:.0f}--{hi}"
    out.append("& \\quad 95\\% CI of the KM median & " + " & ".join(kmci(rowsT[nm]) for nm, _, _ in cols) + " \\\\")
    out.append("& $P_{\\rm ext}(T)$ & " + mag("P_ext") + " \\\\")
    out.append("& CV of $T_{\\rm ext}$ (extinct runs) & " + " & ".join("--" if not np.isfinite(SP[nm]["cv"]) else f"{SP[nm]['cv']:.2f}" for nm, _, _ in cols) + " \\\\")
    out.append("& $(T_{\\rm ext}-t_{\\rm pk})/t_{\\rm pk}$ (extinct) & " + mag("x_rho8_mean") + " \\\\")
    write_tab("tab_res_production_si.tex", out)
    for nm, st, lab in cols:
        r = rowsT[nm]
        d = {k: (float(r[k]) if isinstance(r[k], (int, float, np.floating, np.integer)) else str(r[k]))
             for k in r.index if k.startswith(("f_", "lo_", "hi_", "k_", "n_", "x_", "t_", "P_ext", "trig_"))}
        d["spread_KM"] = SP[nm]
        d["ensamble"] = f"{st}:{lab}"
        g = groups[nm]
        for c in ("x_tpk2", "t_peak", "x_plateau", "x_tB99", "x_tB50", "x_tauB", "x_GB", "x_t_s05ad", "x_tleave"):
            if c in g:
                v = pd.to_numeric(g[c], errors="coerce")
                d[f"{c}_med"] = float(v.median()) if v.notna().any() else None
                d[f"{c}_q25"] = float(v.quantile(.25)) if v.notna().any() else None
                d[f"{c}_q75"] = float(v.quantile(.75)) if v.notna().any() else None
        d["f_O10b"] = fr(g, "O10b")["f"] if "obj_O10b" in g else None
        NUM[f"prod_{nm}"] = d
    # O13b PASS o MARG y O10b de C1
    NUM["prod_C1"]["O13b_pass_or_marg"] = pm(groups["C1"])
    NUM["prod_C2"]["O13b_pass_or_marg"] = pm(groups["C2"])


def table_c1p():
    """C1' = C1 sin AV: ensamble de referencia contra C1, brazo AV del factorial, y ablaciones simples de C1'
    (= ablaciones dobles con AV de abl2, n = 40). tab_res_c1p.tex."""
    from scipy import stats as st
    B, A1, A2 = tab("base"), tab("abl1"), tab("abl2")
    rows = [("C1", B[B.point == pidx("base", LAB["C1"])].iloc[0], 600),
            ("C1$'$ $=$ C1 $-$ AV", B[B.point == pidx("base", LAB["C1p"])].iloc[0], 600),
            ("C1 (factorial)", A1[A1.label == "C1"].iloc[0], 600),
            ("C1 $-$ AV (factorial)", A1[A1.label == "C1 -AV"].iloc[0], 600)]
    nm = {"C1 -R1 -AV": "C1$'$ $-$ R1", "C1 -R2 -AV": "C1$'$ $-$ R2", "C1 -AV -R7": "C1$'$ $-$ R7"}
    for lab, name in nm.items():
        rows.append((name, A2[A2.label == lab].iloc[0], 600))

    def fisher(a, b, alt):
        return float(st.fisher_exact([[a.k_O17p, a.n_O17p - a.k_O17p], [b.k_O17p, b.n_O17p - b.k_O17p]], alternative=alt).pvalue)
    pv = {1: fisher(rows[0][1], rows[1][1], "two-sided"), 3: fisher(rows[2][1], rows[3][1], "greater")}
    base = rows[3][1]
    for i in range(4, len(rows)):
        pv[i] = fisher(base, rows[i][1], "greater")
    out = []
    for i, (lab, r, T) in enumerate(rows):
        def g(c, nd=2):
            v = r.get(f"f_{c}", np.nan)
            return "--" if not np.isfinite(v) else f"{v:.{nd}f}"
        o17 = f"{r.f_O17p:.2f} [{r.lo_O17p:.2f}, {r.hi_O17p:.2f}]"
        out.append(f"{lab} & {int(r.n)} & {T} & {o17} & {p_tex(pv[i]) if i in pv else ''} & {g('O5')} & {g('O11')} & {g('O13p')} & "
                   f"{g('O8')} & {r.x_f0_pk_mean:.2f} \\\\")
        if i in (1, 3):
            out.append("\\midrule")
    write_tab("tab_res_c1p.tex", out)
    NUM["c1p_table"] = [dict(label=lab, n=int(r.n), T=T, f_O17p=float(r.f_O17p), lo=float(r.lo_O17p), hi=float(r.hi_O17p),
                             p=pv.get(i), f_O11=float(r.f_O11), f_O13p=float(r.f_O13p), f_O8=float(r.get("f_O8", np.nan)),
                             f_O5=float(r.f_O5), f0=float(r.x_f0_pk_mean)) for i, (lab, r, T) in enumerate(rows)]
    # diferencia C1 - C1' (Newcombe) en el factorial (n = 120) y en el ensamble de referencia (n = 200)
    for nm_, a, b in (("factorial", rows[2][1], rows[3][1]), ("base", rows[0][1], rows[1][1])):
        lo, hi = v2mod.newcombe(int(a.k_O17p), int(a.n_O17p), int(b.k_O17p), int(b.n_O17p))
        NUM[f"AV_diff_{nm_}"] = dict(d=float(a.f_O17p - b.f_O17p), lo=lo, hi=hi, n=int(a.n_O17p))


# ============================================================================= 2. ablaciones
ABL_ORDER = [("abl1", "C1"), ("abl1", "C1 -R1"), ("abl1", "C1 -R2"), ("abl1", "C1 -AV"), ("abl1", "C1 -R7"),
             ("abl2", "C1 -R1 -R2"), ("abl2", "C1 -R1 -AV"), ("abl2", "C1 -R1 -R7"), ("abl2", "C1 -R2 -AV"),
             ("abl2", "C1 -R2 -R7"), ("abl2", "C1 -AV -R7"), ("abl2", "C1 -R1 -R2 -AV"), ("abl2", "C1 -R1 -R2 -R7"),
             ("abl2", "C1 -R1 -AV -R7"), ("abl2", "C1 -R2 -AV -R7"), ("abl2", "C1 -R1 -R2 -AV -R7"),
             ("abl1", "C1 -R1 alpha=6"), ("abl1", "C1 w=0 rec=0.15"), ("abl1", "C1 -pref (refugios sin preferencia)"),
             ("abl1", "C1 -hard (capacidad blanda)"), ("abl2", "C1 gamma=0 (sin hacinamiento)")]
ABL_NAME = {"C1": "C1", "C1 -R1 alpha=6": "$-$R1, $\\alpha=6$", "C1 w=0 rec=0.15": "$w=0$, $r=0.15$",
            "C1 -pref (refugios sin preferencia)": "$-$preference ($\\pi=0$)",
            "C1 -hard (capacidad blanda)": "soft capacity", "C1 gamma=0 (sin hacinamiento)": "$\\gamma=0$"}


def abl_name(l):
    return ABL_NAME.get(l, l.replace("C1 ", "").replace("-", "$-$"))


def fig_ablations():
    T = {"abl1": tab("abl1"), "abl2": tab("abl2")}
    Tv = {"abl1": ptab("abl1", "v1_"), "abl2": ptab("abl2", "v1_")}
    rows = []
    for st, lab in ABL_ORDER:
        r = T[st][T[st].label == lab].iloc[0]
        d = {c: r.get(c, np.nan) for c in r.index}
        d.update(label=lab, name=abl_name(lab), n=int(r["n"]), f_O17p_v1=float(Tv[st][Tv[st].label == lab].iloc[0].f_O17p))
        rows.append(d)
    D = pd.DataFrame(rows)
    pats = ["O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p", "O17_run", "O17p", "O8", "O9", "O14", "O15b"]
    pats_num = [p for p in pats if p not in ("O17p", "O17_run")]
    patn = ["O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p", "O17", "O17p", "O8", "O9", "O14", "O15b"]
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.05)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax)
    y = np.arange(len(D))[::-1]
    f, lo, hi = D.f_O17p.to_numpy(float), D.lo_O17p.to_numpy(float), D.hi_O17p.to_numpy(float)
    ax.errorbar(f, y, xerr=xe(f, lo, hi), fmt="o", ms=3.5, color=CAT[0], ecolor=CAT[0], elinewidth=1, capsize=0,
                label="O17p (primary evaluator)")
    ax.plot(D.f_O17p_v1.to_numpy(float), y, "D", ms=3.2, mfc="white", mec=CAT[1], mew=1.0, label="O17p, preregistered evaluator", ls="none")
    ax.axvline(0.8, color=MUTED, lw=0.7, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([f"{nm} ($n$={n})" for nm, n in zip(D.name, D.n)])
    ax.set_xlim(-0.03, 1.03); ax.set_xlabel("fraction of runs (95% Wilson CI)")
    for yy in (y[4] - 0.5, y[15] - 0.5):
        ax.axhline(yy, color=MUTED, lw=0.5)
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.3)
    panel(ax, "a")
    M = D[[f"f_{p}" for p in pats]].to_numpy(float)
    ax2.imshow(M, aspect="auto", cmap=CMAP, vmin=0, vmax=1, extent=(-0.5, len(pats) - 0.5, -0.5, len(D) - 0.5),
               origin="upper", interpolation="nearest")
    ax2.grid(False)
    for i in range(len(D)):
        for j in range(len(pats)):
            v = M[i, j]
            yy = len(D) - 1 - i
            ax2.text(j, yy, "NA" if not np.isfinite(v) else f"{v:.2f}".replace("0.", ".").replace("1.00", "1"),
                     ha="center", va="center", fontsize=5.4, color="white" if (np.isfinite(v) and v > 0.55) else INK)
    ax2.set_xticks(range(len(pats))); ax2.set_xticklabels(patn, rotation=0)
    ax2.xaxis.tick_top()
    plt.setp(ax2.get_yticklabels(), visible=False)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    for yy in (y[4] - 0.5, y[15] - 0.5):
        ax2.axhline(yy, color="white", lw=1.5)
    panel(ax2, "b")
    save(fig, "fig_res_ablations.pdf")
    NUM["ablations"] = D[["label", "n", "f_O17p", "lo_O17p", "hi_O17p", "f_O17p_v1", "f_O17_run", "P_ext"] +
                         [f"f_{p}" for p in pats_num] + [f"trig_{a}" for a in ("A1", "A2", "A3", "A4") if f"trig_{a}" in D]].to_dict("records")


# ============================================================================= 3. tablas Q y QC
Q_DESC = {
    "Q1": ("Structural irreversibility (paired cross-transplant, exact McNemar)",
           "$P_T(\\mathrm{D}\\mid\\textsf{C1})\\le0.05$, $P_T(\\mathrm{D}\\mid X)\\ge0.25$"),
    "Q2": ("Each rule necessary (IUT over single ablations, Fisher)",
           "$f(\\textsf{C1})\\ge0.9$, $f(\\textsf{C1}-r)\\le0.5\\ \\forall r$"),
    "Q3": ("Near-deterministic collapse vs.\\ short-lived control (Fisher)",
           "$P_{\\rm ext}\\ge0.99$, CV $\\le0.10$; short-lived $\\le0.7$"),
    "Q4": ("Phase-D law: slope of $T_{\\rm ext}-t_{\\rm last}$ on $a_{\\max}$ (TOST)", "slope $\\in[0.8,1.25]$, $|\\bar r|\\le10$ wk"),
    "Q5": ("O13 band centred at $n^*\\approx m_c$ (Firth, $b_2<0$)", "$b_2<0$, CI of $n^*_{\\rm opt}\\subset[4,12]$"),
    "Q6": ("$a_w$ weighs less than $r$ (profile LR, $\\theta<1$)", "$0<\\theta<1$"),
    "Q7": ("Endogenous collapse at $\\gamma=0$ (Fisher)", "corner $\\ge0.8$, \\textsf{C1} $\\le0.2$"),
}


def parse_Q(Q):
    """Estimaciones legibles y números a partir de la columna `efecto` de Q.csv (fuente única: pipeline.f2b)."""
    q = Q.set_index("Q")
    est, num = {}, {}
    e = str(q.loc["Q1", "efecto"])
    m = re.search(r"candidato (\d+)/(\d+) .*? vs cruzado (\d+)/(\d+) = ([\d.]+) \[([\d.]+), ([\d.]+)\]; pares discordantes 0→1 = (\d+), 1→0 = (\d+)", e)
    if m:
        k1, n1, k2, n2, f2, lo, hi, n01, n10 = m.groups()
        est["Q1"] = f"{k1}/{n1} vs.\\ {k2}/{n2} ($P_T={float(f2):.2f}$ [{lo}, {hi}]); discordant pairs {n01}:{n10}"
        num["Q1"] = dict(k_C1=int(k1), n_C1=int(n1), k_X=int(k2), n_X=int(n2), P_X=float(f2), lo=float(lo), hi=float(hi), n01=int(n01), n10=int(n10))
    e = str(q.loc["Q2", "efecto"])
    m = re.search(r"C1: ([\d.]+); C1 -R1: ([\d.]+) \(p=([\d.e+-]+)\); C1 -R2: ([\d.]+) \(p=([\d.e+-]+)\); C1 -AV: ([\d.]+) \(p=([\d.e+-]+)\); C1 -R7: ([\d.]+) \(p=([\d.e+-]+)\)", e)
    if m:
        c, r1, p1, r2, p2, av, pav, r7, p7 = m.groups()
        est["Q2"] = f"{c} vs.\\ $-$R1 {r1}, $-$R2 {r2}, $-$AV {av}, $-$R7 {r7}"
        num["Q2"] = dict(C1=float(c), R1=float(r1), R2=float(r2), AV=float(av), R7=float(r7), p_R1=float(p1), p_R2=float(p2), p_AV=float(pav), p_R7=float(p7))
    e = str(q.loc["Q3", "efecto"])
    m = re.search(r"= (\d+)/(\d+); notebook \(T=\d+\) = (\d+)/(\d+); CV\(T_ext\) C1 = ([\d.]+) \[([\d.]+), ([\d.]+)\]", e)
    if m:
        k1, n1, k2, n2, cv, lo, hi = m.groups()
        est["Q3"] = f"{k1}/{n1} vs.\\ {k2}/{n2}; CV {cv} [{lo}, {hi}]"
        num["Q3"] = dict(k_C1=int(k1), n_C1=int(n1), k_nb=int(k2), n_nb=int(n2), cv=float(cv), cv_lo=float(lo), cv_hi=float(hi))
    e = str(q.loc["Q4", "efecto"])
    m = re.search(r"pendiente ([\d.]+) ± ([\d.]+) \(HC3\), intercepto (-?[\d.]+); residuo .*? = (-?[\d.]+) ± ([\d.]+) sem \(sd ([\d.]+)\)", e)
    if m:
        b, se, b0, r, rse, sd = m.groups()
        est["Q4"] = f"slope ${b}\\pm{se}$; $\\bar r={r}\\pm{rse}$ wk"
        num["Q4"] = dict(slope=float(b), se=float(se), intercept=float(b0), resid=float(r), resid_se=float(rse), resid_sd=float(sd))
    e = str(q.loc["Q5", "efecto"])
    m = re.search(r"b2 = (-?[\d.]+) ± ([\d.]+)", e)
    if m:
        b2, se = m.groups()
        est["Q5"] = f"$b_2={float(b2):+.1f}\\pm{float(se):.1f}$ ({'convex' if float(b2) > 0 else 'concave'}); no interior maximum" if float(b2) > 0 else f"$b_2={float(b2):+.1f}\\pm{float(se):.1f}$"
        num["Q5"] = dict(b2=float(b2), se=float(se), nota=str(q.loc["Q5", "nota"]))
    e = str(q.loc["Q6", "efecto"])
    m = re.search(r"θ̂ = (-?[\d.]+) \[IC perfil 95 % (-?[\d.]+), (-?[\d.]+)\]; LR\(θ=1\) = ([\d.]+); cuasi-binomial φ = ([\d.]+): IC \[(-?[\d.]+), (-?[\d.]+)\], p = ([\d.e+-]+)", e)
    if m:
        th, lo, hi, lr, phi, qlo, qhi, pq = m.groups()
        est["Q6"] = f"$\\hat\\theta={float(th):.2f}$ [{float(qlo):.2f}, {float(qhi):.2f}] (quasi-binomial, $\\phi={float(phi):.1f}$)"
        num["Q6"] = dict(theta=float(th), lo=float(lo), hi=float(hi), LR=float(lr), phi=float(phi), qlo=float(qlo), qhi=float(qhi), p_quasi=float(pq))
    e = str(q.loc["Q7", "efecto"])
    m = re.search(r"esquina subcrítica (\d+)/(\d+) vs candidato (\d+)/(\d+) \(explota: ([\d.]+)\)", e)
    if m:
        k1, n1, k2, n2, ex = m.groups()
        est["Q7"] = f"{k1}/{n1} vs.\\ {k2}/{n2} ({int(round(float(ex) * int(n2)))}/{n2} explode)"
        num["Q7"] = dict(k_corner=int(k1), n_corner=int(n1), k_C1=int(k2), n_C1=int(n2), f_exploded=float(ex))
    return est, num


def q_table_rows(Q, est):
    """Filas (p, p_Holm con el p cuasi-binomial de Q6, rechazo, predicción)."""
    q = Q.set_index("Q").copy()
    p = q["p"].to_numpy(float).copy()
    i6 = list(q.index).index("Q6")
    e6 = str(q.loc["Q6", "efecto"])
    m = re.search(r"p = ([\d.e+-]+)$", e6)
    if m:
        p[i6] = float(m.group(1))        # Q6: el p cuasi-binomial es el que entra en la secuencia de Holm
    ph = holm(p)
    rows = {}
    for j, k in enumerate(q.index):
        met = q.loc[k, "cumple_prediccion"]
        rows[k] = dict(p=float(p[j]), p_holm=float(ph[j]), rej=bool(ph[j] <= 0.05),
                       met=("yes" if str(met) == "True" else ("no" if str(met) == "False" else "--")), est=est.get(k, "--"))
    return rows


def table_Q():
    Q = pd.read_csv(RES / "f2b_final" / "Q.csv")
    Qv = pd.read_csv(RES / "f2b_final" / "v1" / "Q.csv")
    est, num = parse_Q(Q)
    estv, numv = parse_Q(Qv)
    R, Rv = q_table_rows(Q, est), q_table_rows(Qv, estv)
    out = []
    for k in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]:
        r = R[k]
        out.append(f"{k} & {Q_DESC[k][1]} & {r['est']} & {p_tex(r['p'])} & {p_tex(r['p_holm'])} & "
                   f"{'yes' if r['rej'] else 'no'} & {r['met']} \\\\")
    write_tab("tab_res_Q.tex", out)
    # comparación primario frente al evaluador v1 (réplica de lo preregistrado)
    out = []
    for k in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]:
        r, rv = R[k], Rv[k]
        out.append(f"{k} & {r['est']} & {p_tex(r['p_holm'])} & {'yes' if r['rej'] else 'no'} & {r['met']} & "
                   f"{rv['est']} & {p_tex(rv['p_holm'])} & {'yes' if rv['rej'] else 'no'} & {rv['met']} \\\\")
    write_tab("tab_res_Qcmp.tex", out)
    NUM["Q"] = {k: dict(**R[k], **num.get(k, {})) for k in R}
    NUM["Q_v1"] = {k: dict(**Rv[k], **numv.get(k, {})) for k in Rv}
    NUM["Q"]["_ensambles"] = "trans (Q1), abl1 (Q2), base+ref (Q3), amax (Q4), p2 (Q5), p1aw (Q6), p1 (Q7)"


def table_QC():
    C = pd.read_csv(RES / "f2b_final" / "contraste_c1c2.csv")
    Cv = pd.read_csv(RES / "f2b_final" / "v1" / "contraste_c1c2.csv")
    NUM["QC"] = C.to_dict("records")
    NUM["QC_v1"] = Cv.to_dict("records")
    name = {"O17_run": "O17", "O13b (PASS o MARG)": "O13b (PASS or MARG.)", "x_rho_pk": "$\\rho_{\\rm pk}$ (median)",
            "x_f0_pk": "$f_0$ (median)", "x_BO4": "$\\Phi^{\\rm pers}_{\\rm BO}$ (median)",
            "x_rho8 (extintas)": "$(T_{\\rm ext}-t_{\\rm pk})/t_{\\rm pk}$ (extinct)",
            "x_D_last (extintas)": "$T_{\\rm ext}-t_{\\rm last}$ (extinct)",
            "t_ext (mediana KM; log-rank con censura)": "$T_{\\rm ext}$ (KM median; log-rank)"}
    skip = {"O13", "O10", "O13b", "A1", "A3", "A4", "O15", "O14", "O8b", "O11b"}
    out = []
    cv = Cv.set_index("objetivo")
    for _, r in C.iterrows():
        o = r["objetivo"]
        if o in skip:
            continue
        cont = not o.startswith(("O", "A"))
        fa = f"{r.f_A:.3g}" if cont else f"{r.f_A:.3f}"
        fb = f"{r.f_B:.3g}" if cont else f"{r.f_B:.3f}"
        sig = "\\textbf{yes}" if r["difiere_holm_0.05"] else "no"
        v1 = ""
        if o in cv.index and not cont:
            v1 = f"{cv.loc[o, 'f_A']:.2f} / {cv.loc[o, 'f_B']:.2f}" + (" yes" if cv.loc[o, "difiere_holm_0.05"] else " no")
        elif o in cv.index:
            v1 = "yes" if cv.loc[o, "difiere_holm_0.05"] else "no"
        out.append(f"{name.get(o, o)} & {fa} ({int(r.n_A)}) & {fb} ({int(r.n_B)}) & {p_tex(r.p)} & {p_tex(r.p_holm)} & {sig} & {v1} \\\\")
    write_tab("tab_res_QC.tex", out)
    NUM["QC_family_size"] = int(len(C))
    NUM["QC_rejected"] = int(C["difiere_holm_0.05"].sum())
    NUM["QC_rejected_v1"] = int(Cv["difiere_holm_0.05"].sum())


# ============================================================================= 4. poder discriminante
NULL_COLS = [("C1", "C1"), ("C1′", "C1$'$"), ("γ=0", "$\\gamma=0$"), ("α=0", "$\\alpha=0$"), ("C0 (−R7)", "C0"),
             ("−R2", "$-$R2$^*$"), ("longevo", "long-lived"), ("sin reglas", "no rules"), ("notebook", "short-lived")]
NULL_ROWS = ["O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p", "O17p", "O10b", "O8", "O8b", "O9", "O11b", "O14"]


def table_null():
    """tab_res_null.tex: fracción PASS de cada criterio (NA cuenta como no pasa) en C1, C1', los nulos verdaderos y
    las ablaciones parciales, con el evaluador primario; columna final: genérico / discrimina. tab_res_null_v1.tex:
    lo mismo con el evaluador v1 para O4, O5, O10, O11, O13p y O17p (antes y después de la redefinición)."""
    res, _ = v2mod.discriminant_table(RES)
    W, S, D = res["ancho"], res["resumen"].set_index("criterio"), res["tabla"]
    old = f2b.VARIANT
    f2b.VARIANT = "v1"
    try:
        resv, _ = v2mod.discriminant_table(RES)
    finally:
        f2b.VARIANT = old
    Wv, Sv = resv["ancho"], resv["resumen"].set_index("criterio")
    disc_name = {"γ=0": "$\\gamma{=}0$", "α=0": "$\\alpha{=}0$", "C0 (−R7)": "C0", "−R2": "$-$R2", "longevo": "long",
                 "sin reglas": "none", "notebook": "short", "C1′": "C1$'$"}

    def row(W_, S_, code):
        cells = []
        for c, _ in NULL_COLS:
            v = W_.loc[code, c] if (code in W_.index and c in W_.columns) else np.nan
            cells.append("--" if not np.isfinite(v) else f"{v:.2f}")
        gen = "yes" if bool(S_.loc[code, "generico"]) else "no"
        disc = str(S_.loc[code, "discrimina_frente_a"])
        disc = ", ".join(disc_name.get(x.strip(), x.strip()) for x in disc.split(",")) if disc != "—" else "none"
        return f"{code} & " + " & ".join(cells) + f" & {gen} & {disc} \\\\"
    out = []
    for code in NULL_ROWS:
        if code in W.index:
            out.append(row(W, S, code))
        if code == "O17p":
            out.append("\\midrule")
    write_tab("tab_res_null_full.tex", out)
    write_tab("tab_res_null.tex", [row(W, S, c) for c in ("O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p", "O17p")] +
              ["\\midrule", row(W, S, "O10b")])
    out = []
    for code in ["O4", "O5", "O10", "O11", "O13p", "O17p"]:
        if code in Wv.index:
            out.append(row(Wv, Sv, code))
    write_tab("tab_res_null_v1.tex", out)
    NUM["null"] = dict(ancho=W.reset_index().to_dict("records"), resumen=S.reset_index().to_dict("records"),
                       ancho_v1=Wv.reset_index().to_dict("records"), resumen_v1=Sv.reset_index().to_dict("records"),
                       n_na=D.groupby("criterio").n_na.sum().to_dict(),
                       ensambles="base (C1, C1', α=0, C0), nulos (γ=0, −R2, longevo CI de C1, sin reglas), ref (notebook)")


# ============================================================================= 5. re-puntuaciones
RESC_ROWS = [("base", "C1", "C1"), ("base", "C1p", "C1$'$"), ("c1c2", "C2", "C2"), ("abl1", "C1 -AV", "C1 $-$ AV"),
             ("abl1", "C1 -R1", "C1 $-$ R1"), ("abl1", "C1 -R2", "C1 $-$ R2"), ("abl1", "C1 -R7", "C1 $-$ R7"),
             ("base", "C0 (control)", "C0"), ("base", "C1 alpha=0 (control O14)", "C1, $\\alpha=0$"),
             ("nulos", "C1 gamma=0 (sin hacinamiento)", "$\\gamma=0$"), ("nulos", "C1 -R2 (w=1, sin aprendizaje social)", "$-$R2$^*$ ($w=1$)"),
             ("nulos", "longevo (notebook a_max=120, CI de C1)", "long-lived"), ("nulos", "C1 sin reglas (-R1 -R2 -AV -R7)", "no rules"),
             ("rob", "C1 R7 legacy (v1)", "C1, single-pass R7"), ("rob", "C1 mu_bg=0.003", "C1, $\\mu_{\\rm bg}=0.003$"),
             ("rob", "C1 mu_bg=0.008", "C1, $\\mu_{\\rm bg}=0.008$"), ("rob", "C1 nido=3", "C1, nest 3 wk"),
             ("rob", "C1 async", "C1, asynchronous"), ("rob", "C1 async blanda", "C1, async., soft cap."),
             ("rob", "C1 s0=0.25", "C1, $s_{\\rm base}=0.25$"), ("rob", "C1 s0=0.5", "C1, $s_{\\rm base}=0.5$"),
             ("ci", "C1 CI edades 4-52", "C1, ages U\\{4..52\\}"), ("ci", "C1 CI edades 4-80", "C1, ages U\\{4..80\\}"),
             ("rob", "C1p mu_bg=0.003", "C1$'$, $\\mu_{\\rm bg}=0.003$"), ("rob", "C1p nido=3", "C1$'$, nest 3 wk"),
             ("rob", "C1p async", "C1$'$, asynchronous"), ("rob", "C1p s0=0.25", "C1$'$, $s_{\\rm base}=0.25$"),
             ("ci", "C1p CI edades 4-52", "C1$'$, ages U\\{4..52\\}"), ("ci", "C1p CI edades 4-80", "C1$'$, ages U\\{4..80\\}")]


def table_rescoring():
    R = pd.read_csv(RES / "f2b_final" / "v2" / "repuntuacion.csv")
    P = R[R.criterio == "O17p"].pivot_table(index=["etapa", "brazo"], columns="variante", values="f", aggfunc="first")
    vmap = {"primario v2": "primary", "evaluador v1": "v1", "argmax": "argmax", "fin meseta": "plateau end",
            "m > m_c": "$m>m_c$", "O5 con t_last": "O5 on $t_{\\rm last}$", "O4 v1": "O4 v1"}
    order = ["primario v2", "evaluador v1", "argmax", "fin meseta", "mc", "l5", "o4a"]
    cols = [c for c in ("primario v2", "evaluador v1", "argmax", "fin meseta", "m > m_c", "O5 con t_last", "O4 v1") if c in P.columns]
    out = []
    recs = []
    for st, lab, name in RESC_ROWS:
        if (st, lab) not in P.index:
            continue
        r = P.loc[(st, lab)]
        out.append(f"{name} & " + " & ".join("--" if not np.isfinite(r[c]) else f"{r[c]:.2f}" for c in cols) + " \\\\")
        recs.append(dict(etapa=st, brazo=lab, **{vmap[c]: float(r[c]) for c in cols}))
        if lab in ("C1 -R7", "C1 sin reglas (-R1 -R2 -AV -R7)", "C1 CI edades 4-80") and st != "ci" or (st, lab) == ("ci", "C1 CI edades 4-80"):
            out.append("\\midrule")
    write_tab("tab_res_rescoring.tex", out)
    NUM["rescoring"] = recs
    # qué criterio explica cada caída (primario frente a v1) en los brazos de rob y ci
    Rc = R[R.criterio.isin(["O4", "O5", "O10", "O11", "O13p"]) & R.variante.isin(["primario v2", "evaluador v1"])]
    NUM["rescoring_criteria"] = Rc.pivot_table(index=["etapa", "brazo", "criterio"], columns="variante", values="f").reset_index().to_dict("records")


# ============================================================================= 6. robustez
ROB_ROWS = [("rob", "C1 R7 legacy (v1)", "single-pass R7 admission"), ("rob", "C1 mu_bg=0.003", "$\\mu_{\\rm bg}=0.003$"),
            ("rob", "C1 mu_bg=0.008", "$\\mu_{\\rm bg}=0.008$"), ("rob", "C1 nido=3", "nest period 3 wk"),
            ("rob", "C1 async", "asynchronous, hard"), ("rob", "C1 async blanda", "asynchronous, soft"),
            ("rob", "C1 s0=0.25", "$s_{\\rm base}=0.25$"), ("rob", "C1 s0=0.5", "$s_{\\rm base}=0.5$"),
            ("ci", "C1 CI edades 4-52", "ages U\\{4..52\\}"), ("ci", "C1 CI edades 4-80", "ages U\\{4..80\\}")]


def _fail_crit(g, codes=("O4", "O5", "O6", "O7", "O10", "O11", "O13", "O13p")):
    fs = {c: fr(g, c)["f"] for c in codes if f"obj_{c}" in g}
    # NA cuenta como falla para el diagnóstico
    for c in codes:
        if f"obj_{c}" in g:
            v = g[f"obj_{c}"].astype(str)
            fs[c] = float((v == "PASS").sum() / len(v))
    worst = min(fs, key=fs.get)
    return worst, fs[worst]


def _disc(g):
    """O17p restringido a los criterios que discriminan frente a los nulos
    (O11, O13, O13p) y a los anti-patrones; deja fuera los criterios genéricos (O4, O5, O6, O7, O10)."""
    ok = np.ones(len(g), bool)
    for c in ("O11", "O13", "O13p"):
        ok &= (g[f"obj_{c}"].astype(str) == "PASS").to_numpy()
    for a in ("A1", "A2", "A3", "A4"):
        if f"obj_{a}" in g:
            ok &= (g[f"obj_{a}"].astype(str) != "TRIGGERED").to_numpy()
    return float(ok.mean())


def table_rob():
    rows, recs = [], []
    for cand, suf in (("C1", "C1"), ("C1p", "C1p")):
        ref = arm("base", cand)
        ref_v1 = arm("base", cand, "v1_")
        r0 = fr(ref, "O17p")
        rv0 = fr(ref_v1, "O17p")
        cname = "C1" if cand == "C1" else "C1$'$"
        rta0 = fr(arm("base", cand, "ta_"), "O17p")
        rho0 = pd.to_numeric(ref.x_rho_pk, errors="coerce").median(); plat0 = pd.to_numeric(ref.x_plateau, errors="coerce").median()
        tmed0 = survival.km_median(km_of(ref)[0])["median"]
        rows.append(f"\\emph{{{cname}}}, reference & {fci(r0['f'], r0['lo'], r0['hi'])} & -- & {rv0['f']:.2f} & {rta0['f']:.2f} & {_disc(ref):.2f} & -- & "
                    f"{rho0:.2f} & {plat0:.0f} & {tmed0:.0f} \\\\")
        for st, lab, name in ROB_ROWS:
            lab2 = lab if cand == "C1" else lab.replace("C1 ", "C1p ")
            if lab2 not in labels(st):
                continue
            g = arm(st, lab2)
            gv = arm(st, lab2, "v1_")
            d = fr(g, "O17p"); dv = fr(gv, "O17p"); dta = fr(arm(st, lab2, "ta_"), "O17p")
            lo, hi = v2mod.newcombe(d["k"], d["n"], r0["k"], r0["n"])
            worst, fw = _fail_crit(g)
            rho = pd.to_numeric(g.x_rho_pk, errors="coerce").median()
            f0 = pd.to_numeric(g.x_f0_pk, errors="coerce").median()
            plat = pd.to_numeric(g.x_plateau, errors="coerce").median()
            km, t, ev = km_of(g)
            tmed = survival.km_median(km)["median"]
            fail = f"{worst} {fw:.2f}" if d["f"] < 0.95 else "--"
            o5 = g["obj_O5"].astype(str).value_counts()
            rows.append(f"{name} & {fci(d['f'], d['lo'], d['hi'])} & {d['f'] - r0['f']:+.2f} [{lo:+.2f}, {hi:+.2f}] & {dv['f']:.2f} & {dta['f']:.2f} & "
                        f"{_disc(g):.2f} & {fail} & {rho:.2f} & {plat:.0f} & {tmed:.0f} \\\\")
            recs.append(dict(candidato=cand, etapa=st, brazo=lab2, n=int(d["n"]), O17p=d["f"], lo=d["lo"], hi=d["hi"],
                             d=d["f"] - r0["f"], d_lo=lo, d_hi=hi, O17p_v1=dv["f"], O17p_argmax=dta["f"], falla=worst, f_falla=fw,
                             rho_pk_med=float(rho), f0_med=float(f0), plateau_med=float(plat), t_ext_med=float(tmed),
                             preserva=bool(d["f"] >= 0.8), O8=fr(g, "O8")["f"], O11=fr(g, "O11")["f"], O5=fr(g, "O5")["f"],
                             disc_O11_O13_O13p=_disc(g),
                             O5_PASS=int(o5.get("PASS", 0)), O5_MARG=int(o5.get("MARG", 0)), O5_FAIL=int(o5.get("FAIL", 0)),
                             t_pk_prim_med=float(pd.to_numeric(g.get("x_t_pk_prim"), errors="coerce").median()),
                             t_B99_prim_med=float(pd.to_numeric(g.get("x_t_B99_prim"), errors="coerce").median())))
        if cand == "C1":
            rows.append("\\midrule")
    write_tab("tab_res_rob.tex", rows)
    NUM["rob"] = recs
    # curva de mortalidad de fondo (robmu + rob + base)
    mu_rows = []
    for cand in ("C1", "C1p"):
        for st in ("base", "rob", "robmu"):
            for lab in labels(st):
                if st == "base" and lab != LAB[cand]:
                    continue
                if st != "base" and not (lab.startswith(cand + " mu_bg=")):
                    continue
                mu = 0.0 if st == "base" else float(lab.split("mu_bg=")[1])
                g = arm(st, lab)
                d = fr(g, "O17p"); dv = fr(arm(st, lab, "v1_"), "O17p"); dta = fr(arm(st, lab, "ta_"), "O17p")
                mu_rows.append(dict(candidato=cand, etapa=st, mu_bg=mu, n=int(d["n"]), O17p=d["f"], lo=d["lo"], hi=d["hi"],
                                    O17p_v1=dv["f"], O17p_argmax=dta["f"], O5=fr(g, "O5")["f"], O8=fr(g, "O8")["f"],
                                    plateau_med=float(pd.to_numeric(g.x_plateau, errors="coerce").median()),
                                    rho_pk_med=float(pd.to_numeric(g.x_rho_pk, errors="coerce").median()),
                                    t_ext_med=float(survival.km_median(km_of(g)[0])["median"])))
    NUM["robmu"] = mu_rows


def table_size():
    S = pd.read_csv(RES / "f2b_size" / "analisis" / "tabla_tamano.csv")
    s = summ("size")
    out, recs = [], []
    for _, r in S.iterrows():
        g = s[s.point == r.point]
        d = fr(g, "O17p")
        N0 = int(str(r.label).split("N0=")[1])
        out.append(f"{int(r.L)} & {N0} & {int(r.n)} & {fci(d['f'], d['lo'], d['hi'])} & {r.rho_pk_med:.2f} & {r.f0_med:.2f} & {r.BO4_med:.2f} & "
                   f"{r.t_last_med:.0f} & {r.t_last_iqr:.0f} & {r.D_last_med:.0f} & {r.t_ext_med:.0f} & {r.iqr_med_KM:.3f} [{r.iqr_lo:.3f}, {r.iqr_hi:.3f}] \\\\")
        recs.append(dict(L=int(r.L), N0=N0, n=int(r.n), O17p=d["f"], lo=d["lo"], hi=d["hi"], rho_pk=float(r.rho_pk_med), f0=float(r.f0_med),
                         BO4=float(r.BO4_med), t_last_med=float(r.t_last_med), t_last_iqr=float(r.t_last_iqr), D_last=float(r.D_last_med),
                         t_ext_med=float(r.t_ext_med), spread=float(r.iqr_med_KM), spread_lo=float(r.iqr_lo), spread_hi=float(r.iqr_hi)))
    write_tab("tab_res_size.tex", out)
    from scipy import stats as st
    kw = {}
    for c in ("x_rho_pk", "x_f0_pk", "x_BO4", "x_t_lastbirth", "x_D_last"):
        grp = [pd.to_numeric(s[s.point == pt][c], errors="coerce").dropna() for pt in sorted(s.point.unique())]
        kw[c] = dict(H=float(st.kruskal(*grp).statistic), p=float(st.kruskal(*grp).pvalue))
    # mediana de T_ext contra log n: pendiente
    Nn = np.log(S.L.to_numpy(float) ** 2 * 0.44 * 6.5)   # proporcional al número de animales (N_pk ≈ 6,5 N0)
    b = np.polyfit(np.log(S.L.to_numpy(float) ** 2), S.t_ext_med.to_numpy(float), 1)
    NUM["size"] = dict(tabla=recs, kruskal=kw, slope_t_ext_vs_logL2=float(b[0]), ensamble="size")


def fig_rob():
    """(a) O17p contra la mortalidad de fondo (primario, v1, argmax); (b) variantes de robustez y CI;
    (c) tamaño: dispersión KM y O17p contra L."""
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.35, 1.0]))
    M = pd.DataFrame(NUM["robmu"])
    a = axs[0]
    for j, (cand, name) in enumerate((("C1", "C1"), ("C1p", "C1$'$"))):
        d = M[M.candidato == cand].sort_values("mu_bg")
        x = d.mu_bg.to_numpy(float) * 1e3
        a.errorbar(x, d.O17p, yerr=xe(d.O17p, d.lo, d.hi), fmt=MK[j] + "-", ms=3.5, color=CAT[j], lw=1.1, elinewidth=0.7, capsize=0,
                   label=f"{name}, primary")
        a.plot(x, d.O17p_v1, MK[j], ls="--", mfc="white", mec=CAT[j], color=CAT[j], ms=3.2, lw=0.9, label=f"{name}, preregistered evaluator")
        a.plot(x, d.O17p_argmax, "x", ls=":", color=CAT[j], ms=3.5, lw=0.8, label=f"{name}, argmax $t_{{\\rm pk}}$")
    a.axhline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("background mortality $\\mu_{\\rm bg}$ ($10^{-3}$ per week)"); a.set_ylabel("O17p"); a.set_ylim(-0.03, 1.03)
    a.set_title("background mortality", loc="left")
    a.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, handlelength=2.0)
    panel(a, "a")
    a = axs[1]
    R = pd.DataFrame(NUM["rob"])
    names = {"C1 R7 legacy (v1)": "single-pass R7", "C1 mu_bg=0.003": "$\\mu_{\\rm bg}=0.003$", "C1 mu_bg=0.008": "$\\mu_{\\rm bg}=0.008$",
             "C1 nido=3": "nest 3 wk", "C1 async": "asynchronous", "C1 async blanda": "async., soft cap.",
             "C1 s0=0.25": "$s_{\\rm base}=0.25$", "C1 s0=0.5": "$s_{\\rm base}=0.5$", "C1 CI edades 4-52": "ages U{4..52}",
             "C1 CI edades 4-80": "ages U{4..80}"}
    order = [k for k in names]
    y = np.arange(len(order))[::-1]
    for j, cand in enumerate(("C1", "C1p")):
        d = R[R.candidato == cand].copy()
        d["key"] = d.brazo.str.replace("C1p ", "C1 ", regex=False)
        d = d.set_index("key").reindex(order)
        yy = y + (0.18 if j == 0 else -0.18)
        a.errorbar(d.O17p, yy, xerr=xe(d.O17p, d.lo, d.hi), fmt=MK[j], ms=3.5, color=CAT[j], elinewidth=0.9, capsize=0,
                   label=("C1" if j == 0 else "C1$'$") + ", primary")
        a.plot(d.O17p_v1, yy, MK[j], mfc="white", mec=CAT[j], ms=3.2, ls="none", label=("C1" if j == 0 else "C1$'$") + ", preregistered evaluator")
    for j, cand in enumerate(("C1", "C1p")):
        r0 = fr(arm("base", cand), "O17p")["f"]
        a.axvline(r0, color=CAT[j], lw=0.7, ls=LS[1 + j], alpha=0.8)
    a.axvline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_yticks(y); a.set_yticklabels([names[k] for k in order]); a.set_xlim(-0.03, 1.03)
    a.set_xlabel("O17p (95% Wilson CI; $n=200$)"); a.set_title("variants and initial condition", loc="left")
    a.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); a.grid(axis="y", visible=False)
    panel(a, "b")
    a = axs[2]
    S = pd.DataFrame(NUM["size"]["tabla"])
    a.errorbar(S.L, S.spread, yerr=xe(S.spread, S.spread_lo, S.spread_hi), fmt="o-", color=CAT[0], ms=3.5, lw=1.0, elinewidth=0.7,
               capsize=0, label="KM spread of $T_{\\rm ext}$")
    a.plot(S.L, S.t_last_iqr / S.t_last_med, "s--", color=CAT[1], ms=3.2, lw=0.9, label="IQR/median of $t_{\\rm last}$")
    a.plot(S.L, S.O17p, "^:", color=CAT[2], ms=3.5, lw=0.9, label="O17p")
    a.set_xticks([15, 20, 30, 45, 60]); a.set_xlim(10, 65)
    a.set_xlabel("lattice side $L$ (density $N_0/L^2=0.44$)"); a.set_ylim(0, 1.05)
    a.set_title("system size", loc="left"); a.legend(fontsize=5.0, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    panel(a, "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_res_rob.pdf")


# ============================================================================= 7. transplantes
TR_ARMS = [("C1", "C1"), ("C1p", "C1$'$"), ("C1 -R1", "C1 $-$ R1"), ("C1 -R2", "C1 $-$ R2"), ("C1 -R1 -R2", "C1 $-$ R1 $-$ R2"),
           ("X: madre C1, colonia -R1 -R2 rec=0.1", "$X$ (C1 animals)"), ("C0 (control)", "C0"), ("C2 (contraste)", "C2")]


def table_trans():
    A = trans_long()
    tr = pd.read_csv(RES / "f2b_trans" / "transplant.csv")
    E = pd.read_csv(RES / "f2b_trans" / "analisis" / "contraste_tiempo_igual.csv")

    def cell(lab, ph):
        r = A[(A.brazo == lab) & (A.fase == ph)]
        return "--" if not len(r) else f"{int(r.k.iloc[0])}/{int(r.n.iloc[0])}"

    def pooled(lab, phs):
        r = A[(A.brazo == lab) & (A.fase.isin(phs))]
        return "--" if not len(r) else f"{int(r.k.sum())}/{int(r.n.sum())}"
    out = []
    for lab, name in TR_ARMS:
        tD = A[(A.brazo == lab) & (A.fase == "D")].t_snap_mediana.iloc[0]
        tB = A[(A.brazo == lab) & (A.fase == "BAND")].t_snap_mediana.iloc[0]
        out.append(f"{name} & {cell(lab, 'B')} & {cell(lab, 'D')} ({tD:.0f}) & {cell(lab, 'Darg')} & {cell(lab, 'Dfix')} & "
                   f"{pooled(lab, ['MIXfix_M', 'MIXfix_F'])} & {cell(lab, 'BAND')} ({tB:.0f}) & {cell(lab, 'BAND_MIX_M')} & {cell(lab, 'BAND_MIX_F')} \\\\")
    write_tab("tab_res_trans.tex", out)
    # distribución de s y edad de los trasladados, por brazo: D, Dfix, BAND (medianas sobre ensayos de la mediana por ensayo)
    out = []
    for lab, name in TR_ARMS:
        cells = [name]
        for ph in ("D", "Dfix", "BAND"):
            g = tr[(tr.point == pidx("trans", lab)) & (tr.phase == ph)]
            smed = pd.to_numeric(g.s_med, errors="coerce"); smax = pd.to_numeric(g.s_max, errors="coerce")
            amin = pd.to_numeric(g.age_min, errors="coerce"); amax = pd.to_numeric(g.age_max, errors="coerce")
            cells.append(f"{smed.median():.3f} / {smax.max():.2f}")
            cells.append(f"{amin.median():.0f}--{amax.median():.0f}")
        g = tr[(tr.point == pidx("trans", lab)) & (tr.phase.isin(["BAND", "BAND_MIX_M", "BAND_MIX_F"]))]
        sd = pd.to_numeric(g.s_desc_ad, errors="coerce")
        nd = pd.to_numeric(g.n_desc_ad, errors="coerce")
        w = sd.notna() & (nd > 0)
        sdm = float(np.average(sd[w], weights=nd[w])) if w.any() else np.nan
        cells.append("--" if not np.isfinite(sdm) else f"{sdm:.2f} ({int(nd[w].sum())})")
        out.append(" & ".join(cells) + " \\\\")
    write_tab("tab_res_trans_s.tex", out)
    NUM["transplant"] = A.to_dict("records")
    NUM["transplant_equal_time"] = E.to_dict("records")
    # descendientes de la banda por brazo y fase (media ponderada por n_desc_ad)
    desc = []
    for lab, _ in TR_ARMS:
        for ph in ("BAND", "BAND_MIX_M", "BAND_MIX_F"):
            g = tr[(tr.point == pidx("trans", lab)) & (tr.phase == ph)]
            sd = pd.to_numeric(g.s_desc_ad, errors="coerce"); nd = pd.to_numeric(g.n_desc_ad, errors="coerce")
            sm = pd.to_numeric(g.s_desc_mat, errors="coerce")
            w = sd.notna() & (nd > 0)
            desc.append(dict(brazo=lab, fase=ph, k=int(pd.to_numeric(g.success, errors="coerce").sum()), n=int(g.success.notna().sum()),
                             s_desc_ad=float(np.average(sd[w], weights=nd[w])) if w.any() else None,
                             s_desc_mat=float(np.nanmedian(sm)) if sm.notna().any() else None, n_desc_ad=int(nd[w].sum()),
                             s_founders=float(pd.to_numeric(g.s_mean, errors="coerce").median())))
    NUM["transplant_band_desc"] = desc
    # McNemar C1 vs X por fase
    mc = {}
    a, b = pidx("trans", "C1"), pidx("trans", "X: madre C1, colonia -R1 -R2 rec=0.1")
    from scipy import stats as st
    for ph in ("D", "Darg", "Dfix", "BAND", "BAND_MIX_M", "BAND_MIX_F"):
        pa = tr[(tr.point == a) & (tr.phase == ph)].set_index("rep").success.dropna().astype(int)
        pb = tr[(tr.point == b) & (tr.phase == ph)].set_index("rep").success.dropna().astype(int)
        j = pa.index.intersection(pb.index)
        n01 = int(((pa[j] == 0) & (pb[j] == 1)).sum()); n10 = int(((pa[j] == 1) & (pb[j] == 0)).sum())
        p = float(st.binomtest(n01, n01 + n10, 0.5, alternative="greater").pvalue) if n01 + n10 else 1.0
        mc[ph] = dict(k_C1=int(pa.sum()), k_X=int(pb.sum()), n=int(len(j)), n01=n01, n10=n10, p=p)
    NUM["transplant_mcnemar"] = mc


def fig_transplant():
    A = trans_long()
    tr = pd.read_csv(RES / "f2b_trans" / "transplant.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1.0]))
    y0 = np.arange(len(TR_ARMS))[::-1]
    a = axs[0]
    phases = [("B", "phase B (growth)"), ("D", "D: $1.75\\,t'_{\\rm pk}$"), ("Darg", "D: $1.75\\,t_{\\rm pk}$ (argmax)"), ("Dfix", "week 95 (all arms)")]
    for j, (ph, lab) in enumerate(phases):
        f = np.array([A[(A.brazo == l) & (A.fase == ph)].P_T.iloc[0] for l, _ in TR_ARMS], float)
        lo = np.array([A[(A.brazo == l) & (A.fase == ph)].lo.iloc[0] for l, _ in TR_ARMS], float)
        hi = np.array([A[(A.brazo == l) & (A.fase == ph)].hi.iloc[0] for l, _ in TR_ARMS], float)
        yy = y0 + (1.5 - j) * 0.19
        a.errorbar(f, yy, xerr=xe(f, lo, hi), fmt=MK[j], ms=3.3, color=CAT[j], elinewidth=0.9, capsize=0, label=lab)
    a.set_yticks(y0); a.set_yticklabels([n for _, n in TR_ARMS])
    a.set_xlim(-0.02, 1.02); a.set_xlabel("$P_T$ (95% Wilson CI; 60 trials per arm)")
    a.axvline(0.1, color=MUTED, lw=0.6, ls=":"); a.axvline(0.5, color=MUTED, lw=0.6, ls=":")
    a.set_title("phase-D animals", loc="left")
    a.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    a.grid(axis="y", visible=False)
    panel(a, "a")
    a = axs[1]
    phases = [("BAND", "band animals only"), ("BAND_MIX_M", "band males + healthy females"), ("BAND_MIX_F", "band females + healthy males")]
    for j, (ph, lab) in enumerate(phases):
        f = np.array([A[(A.brazo == l) & (A.fase == ph)].P_T.iloc[0] for l, _ in TR_ARMS], float)
        lo = np.array([A[(A.brazo == l) & (A.fase == ph)].lo.iloc[0] for l, _ in TR_ARMS], float)
        hi = np.array([A[(A.brazo == l) & (A.fase == ph)].hi.iloc[0] for l, _ in TR_ARMS], float)
        yy = y0 + (1 - j) * 0.22
        a.errorbar(f, yy, xerr=xe(f, lo, hi), fmt=MK[j], ms=3.3, color=CAT[j + 4], elinewidth=0.9, capsize=0, label=lab)
    a.set_yticks(y0); plt.setp(a.get_yticklabels(), visible=False)
    a.set_xlim(-0.02, 1.02); a.set_xlabel("$P_T$, adults with $0.2\\leq s\\leq0.5$")
    a.set_title("band animals (weeks 8--11)", loc="left")
    a.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1)
    a.grid(axis="y", visible=False)
    panel(a, "b")
    a = axs[2]
    rng = np.random.default_rng(0)
    for i, (lab, _) in enumerate(TR_ARMS):
        g = tr[(tr.point == pidx("trans", lab)) & (tr.phase == "Dfix")]
        s = pd.to_numeric(g.s_mean, errors="coerce").to_numpy(float)
        yy = y0[i] + rng.uniform(-0.25, 0.25, s.size)
        a.plot(np.clip(s, 1e-4, 1), yy, ".", ms=2.6, color=CAT[0], alpha=0.5, mew=0)
        g2 = tr[(tr.point == pidx("trans", lab)) & (tr.phase == "BAND")]
        s2 = pd.to_numeric(g2.s_mean, errors="coerce").to_numpy(float)
        a.plot(np.clip(s2, 1e-4, 1), y0[i] + rng.uniform(-0.25, 0.25, s2.size), ".", ms=2.6, color=CAT[4], alpha=0.5, mew=0)
    a.plot([], [], ".", color=CAT[0], label="week 95"); a.plot([], [], ".", color=CAT[4], label="band")
    a.set_xscale("log"); a.set_xlim(8e-5, 1.3); a.set_xlabel("mean $s$ of the eight transplanted")
    a.set_yticks(y0); plt.setp(a.get_yticklabels(), visible=False)
    a.set_title("$s$ of the transplanted", loc="left")
    a.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    a.grid(axis="y", visible=False)
    panel(a, "c")
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_res_transplant.pdf")


# ============================================================================= 8. Q7 con a_w variable
def q7aw():
    T = pd.read_csv(RES / "f2b_q7aw" / "analisis" / "tabla_puntos.csv")
    th = pd.read_csv(RES / "f2b_q7aw" / "analisis" / "q7aw_theta.csv", index_col=0)
    J = json.load(open(RES / "f2b_q7aw" / "analisis" / "q7aw.json"))
    T["Lam"] = T.rec.astype(float) * T.plast_age_max.astype(float)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    for i, w in enumerate(sorted(T.inherit_weight.unique())):
        a = axs[i]
        for j, aw in enumerate(sorted(T.plast_age_max.unique())):
            d = T[(np.isclose(T.inherit_weight, w)) & (T.plast_age_max == aw)].sort_values("Lam")
            lo = np.array([wilson(k, n)[0] for k, n in zip(d.k_ext, d.n)]); hi = np.array([wilson(k, n)[1] for k, n in zip(d.k_ext, d.n)])
            a.errorbar(d.Lam, d.P_ext, yerr=xe(d.P_ext, lo, hi), fmt=MK[j], ls=LS[j], color=CAT[j], ms=3.5, lw=1.0,
                       elinewidth=0.7, capsize=0, label=f"$a_w={aw:g}$")
        a.set_xscale("log"); a.set_xlabel("$\\Lambda=r\\,a_w$"); a.set_title(f"$w={w:g}$", loc="left")
        a.set_xticks([0.24, 0.36, 0.48, 0.6, 0.84, 1.2]); a.set_xticklabels(["0.24", "0.36", "0.48", "0.6", "0.84", "1.2"])
        a.set_xticks([], minor=True)
        panel(a, "abc"[i])
    axs[0].set_ylabel("$P_{\\rm ext}$ at $\\gamma=0$"); axs[0].legend(fontsize=5.8, loc="lower left"); axs[0].set_ylim(-0.03, 1.03)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_res_q7aw.pdf")
    # 51 de los 54 puntos tienen P_ext in {0, 1}, así que la logística está separada y el
    # perfil de theta mide la geometría de la grilla, no el error de muestreo. La tabla del texto es ahora el resumen
    # no paramétrico por ventana: horquilla de grilla de Lambda_1/2, interpolación log-lineal y theta local entre
    # ventanas (acotado por la grilla). El ajuste separado se conserva en NUM, rotulado, y no se imprime como inferencia.
    T["Lam"] = T.Lam.round(6)                       # los duplicados en coma flotante inflaban la razón de grilla (1.157)
    grid = np.sort(T.Lam.unique())
    steps = grid[1:] / grid[:-1]
    deg = int(((T.P_ext <= 0) | (T.P_ext >= 1)).sum())

    def bracket(d):
        """Último punto con P_ext >= 1/2 y primero con P_ext < 1/2 (ordenados en Lam), e interpolación log-lineal."""
        d = d.sort_values("Lam")
        x, y = np.log(d.Lam.to_numpy(float)), d.P_ext.to_numpy(float)
        idx = np.where((y[:-1] >= 0.5) & (y[1:] < 0.5))[0]
        if not idx.size:
            return None
        i = idx[0]
        half = float(np.exp(x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))
        return dict(lo=float(d.Lam.iloc[i]), hi=float(d.Lam.iloc[i + 1]), k_lo=int(d.k_ext.iloc[i]), k_hi=int(d.k_ext.iloc[i + 1]),
                    n=int(d.n.iloc[i]), half=half)

    br = {}
    for w in sorted(T.inherit_weight.unique()):
        for aw in sorted(T.plast_age_max.unique()):
            br[(float(w), float(aw))] = bracket(T[np.isclose(T.inherit_weight, w) & (T.plast_age_max == aw)])

    def theta_local(b1, b2, a1, a2):
        """theta en log r + theta log a_w implicado por Lambda_1/2(a1) y Lambda_1/2(a2): theta = 1 - log(L2/L1)/log(a2/a1).
        Devuelve (acotado por la grilla: min, max) y el valor interpolado."""
        la = math.log(a2 / a1)
        lo = 1 - math.log(b2["hi"] / b1["lo"]) / la
        hi = 1 - math.log(b2["lo"] / b1["hi"]) / la
        return lo, hi, 1 - math.log(b2["half"] / b1["half"]) / la

    aws = sorted(T.plast_age_max.unique())
    out, recs, thl = [], [], []
    for iw, w in enumerate(sorted(T.inherit_weight.unique())):
        if iw:
            out.append("\\midrule")
        for j, aw in enumerate(aws):
            b = br[(float(w), float(aw))]
            if b is None:
                out.append(f"{w:g} & {aw:g} & -- & -- & -- & -- \\\\"); continue
            recs.append(dict(w=float(w), a_w=float(aw), **b))
            tl = "--"
            if j + 1 < len(aws) and br[(float(w), float(aws[j + 1]))] is not None:
                lo, hi, mid = theta_local(b, br[(float(w), float(aws[j + 1]))], aw, aws[j + 1])
                thl.append(dict(w=float(w), a_w_from=float(aw), a_w_to=float(aws[j + 1]), theta_lo=lo, theta_hi=hi, theta_interp=mid))
                tl = f"({lo:.2f}, {hi:.2f}); {mid:.2f}"
            out.append(f"{w:g} & {aw:g} & {b['lo']:g} ({b['k_lo']}/{b['n']}) & {b['hi']:g} ({b['k_hi']}/{b['n']}) & {b['half']:.2f} & {tl} \\\\")
    write_tab("tab_res_q7aw.tex", out)
    NUM["q7aw"] = dict(ensamble="q7aw", n_points=int(len(T)), n_runs=int(T.n.sum()), n_points_P_ext_0_or_1=deg,
                       grid_Lam=[float(x) for x in grid], grid_step_ratios=[float(x) for x in steps],
                       grid_ratio_Lam=float(np.exp(np.mean(np.log(steps)))),
                       bracket_by_window=recs, theta_local=thl,
                       # ajuste logístico separado: se conserva para trazabilidad; no es inferencia
                       separated_logistic_fit=dict(nota="51/54 puntos con P_ext in {0,1}: IC de perfil y LR miden la grilla, no el muestreo",
                                                   theta=th.reset_index().to_dict("records"), sigma=J["sigma"], phi=J["phi"],
                                                   sigma_ic=J["sigma_ic"], phi_ic=J["phi_ic"]))
    NUM["q7aw"]["Lambda_half_w0"] = {f"{aw:g}": (br[(0.0, float(aw))] or {}).get("half") for aw in aws}


# ============================================================================= 9. región factible
def region():
    """Tres estimadores (ingenuo, volumen, empírico-Bayes) con el primario, el evaluador v1 y el umbral m > m_c,
    sin y con las réplicas extra de lhsx; descomposición por subconjuntos (post hoc) con el primario y el v1."""
    names, scales = (json.load(open(RES / "f2b_lhs" / "problem.json"))["problem"][k] for k in ("names", "scale"))
    rx = run("lhsx")
    out, recs = [], []
    post_all = {}
    for vname, pref in (("primary", ""), ("v1 evaluator", "v1_"), ("$m>m_c$", "mc_")):
        tabs = []
        for st in ("lhs", "lhs2"):
            T = ptab(st, pref)
            T["_src"] = st
            tabs.append(T)
        T = pd.concat(tabs, ignore_index=True)
        rxx = dict(rx); rxx["summary"] = summ_var("lhsx", pref)
        Tx = f2b.point_table(rxx, "O17p")
        extra = {}
        for _, r in Tx.iterrows():
            src, pt = str(r["label"]).split(":")
            src = src.replace("f2b_", "")
            j = T.index[(T._src == src) & (T.point == int(pt))]
            if len(j):
                extra[int(j[0])] = (float(r["k_PRIMARIO"]), float(r["n_PRIMARIO"]))
        for lab, ex in (("10 per point", None), ("+40 at 18 points", extra)):
            rr = v2mod.feasible_region(T, names, scales, extra=ex, B=300)
            out.append(f"{ {'v1 evaluator': 'preregistered'}.get(vname, vname)} & {lab} & {rr['ingenuo']:.3f} [{rr['ingenuo_ic'][0]:.3f}, {rr['ingenuo_ic'][1]:.3f}] & "
                       f"{rr['volumen']:.3f} [{rr['volumen_ic'][0]:.3f}, {rr['volumen_ic'][1]:.3f}] & "
                       f"{rr['eb']:.3f} [{rr['eb_ic'][0]:.3f}, {rr['eb_ic'][1]:.3f}] & {rr['phi']:.2f} & {rr['n_inciertos']} \\\\")
            recs.append(dict(evaluador=vname, lhsx=ex is not None, **{k: v for k, v in rr.items() if k != "post"}))
            if ex is not None:
                post_all[vname] = (T, rr["post"], extra)
        # descomposición por subconjuntos (post hoc; con las réplicas extra)
        k = T["k_PRIMARIO"].to_numpy(float).copy(); n = T["n_PRIMARIO"].to_numpy(float).copy()
        for i, (kk, nn) in extra.items():
            k[i] += kk; n[i] += nn
        f = k / n
        feas = f >= 0.8
        cap = T.refuge_frac * T.refuge_capacity
        subsets = [("all", np.ones(len(T), bool)), ("cR<=2", (T.refuge_capacity <= 2).to_numpy()),
                   ("+mc<=10", ((T.refuge_capacity <= 2) & (T.m_c <= 10)).to_numpy()),
                   ("+fRcR>=0.3", ((T.refuge_capacity <= 2) & (T.m_c <= 10) & (cap >= 0.3)).to_numpy()),
                   ("+pi>=100", ((T.refuge_capacity <= 2) & (T.m_c <= 10) & (cap >= 0.3) & (T.refuge_pref >= 100)).to_numpy())]
        rng = np.random.default_rng(0)
        dec = {}
        for nm_, m in subsets:
            x = feas[m].astype(float)
            bs = [rng.choice(x, x.size).mean() for _ in range(2000)]
            dec[nm_] = dict(n=int(m.sum()), k=int(x.sum()), f=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                            mean_O17p=float(f[m].mean()))
        sub = subsets[-1][1]
        nf = ~feas[sub]
        dec["fail_in_last_subset"] = dict(n_fail=int(nf.sum()), O11_lt_08=int((nf & (T.f_O11.to_numpy()[sub] < 0.8)).sum()),
                                           O13p_lt_08=int((nf & (T.f_O13p.to_numpy()[sub] < 0.8)).sum()),
                                           O5_lt_08=int((nf & (T.f_O5.to_numpy()[sub] < 0.8)).sum()),
                                           O4_lt_08=int((nf & (T.f_O4.to_numpy()[sub] < 0.8)).sum()))
        dec["cR3"] = dict(n=int((T.refuge_capacity == 3).sum()), feas=int(feas[(T.refuge_capacity == 3).to_numpy()].sum()))
        NUM[f"lhs_decomp_{pref or 'primary'}"] = dec
    write_tab("tab_res_region.tex", out)
    NUM["region"] = recs
    # fracción ingenua y volumen con cada opción del evaluador por separado (con lhsx): qué opción explica el cambio
    byv = {}
    for vname, pref in VARS:
        tabs = []
        for st in ("lhs", "lhs2"):
            T_ = ptab(st, pref); T_["_src"] = st; tabs.append(T_)
        T_ = pd.concat(tabs, ignore_index=True)
        rxx = dict(rx); rxx["summary"] = summ_var("lhsx", pref)
        Tx_ = f2b.point_table(rxx, "O17p")
        k_ = T_["k_PRIMARIO"].to_numpy(float).copy(); n_ = T_["n_PRIMARIO"].to_numpy(float).copy()
        for _, r in Tx_.iterrows():
            src, pt = str(r["label"]).split(":")
            j = T_.index[(T_._src == src.replace("f2b_", "")) & (T_.point == int(pt))]
            if len(j):
                k_[j[0]] += float(r["k_PRIMARIO"]); n_[j[0]] += float(r["n_PRIMARIO"])
        byv[vname] = dict(ingenuo=float((k_ / n_ >= 0.8).mean()), volumen=float((k_ / n_).mean()), k_ge08=int((k_ / n_ >= 0.8).sum()))
    NUM["region_by_variant"] = byv
    NUM["lhsx"] = dict(n_points=int(len(Tx)), reps=int(Tx.n.iloc[0]) if len(Tx) else 0)
    # figura: k/n contra la probabilidad a posteriori de p >= 0,8 (primario, con lhsx), y descomposición
    T, post, extra = post_all["primary"]
    k = T["k_PRIMARIO"].to_numpy(float).copy(); n = T["n_PRIMARIO"].to_numpy(float).copy()
    isx = np.zeros(len(T), bool)
    for i, (kk, nn) in extra.items():
        k[i] += kk; n[i] += nn; isx[i] = True
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.1), gridspec_kw=dict(width_ratios=[1, 1.2]))
    a = axs[0]
    rng = np.random.default_rng(1)
    a.plot(k[~isx] / n[~isx] + rng.uniform(-0.012, 0.012, (~isx).sum()), post[~isx], "o", ms=2.6, color=CAT[0], alpha=0.55, mew=0, label="10 runs")
    a.plot(k[isx] / n[isx], post[isx], "D", ms=3.6, color=CAT[1], mew=0, label="50 runs (boundary)")
    a.axhline(0.5, color=MUTED, lw=0.6, ls=":"); a.axvline(0.8, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("observed O17p of the point, $k/n$"); a.set_ylabel("$P(p_i\\geq0.8\\mid\\mathrm{data})$")
    a.set_title("per-point classification (400 points)", loc="left"); a.legend(fontsize=5.8, loc="center left")
    panel(a, "a")
    a = axs[1]
    for j, (pref, lab) in enumerate((("primary", "primary evaluator"), ("v1_", "preregistered evaluator"), ("mc_", "$m>m_c$"))):
        dec = NUM[f"lhs_decomp_{pref}"]
        keys = ["all", "cR<=2", "+mc<=10", "+fRcR>=0.3", "+pi>=100"]
        f = np.array([dec[kk]["f"] for kk in keys]); lo = np.array([dec[kk]["lo"] for kk in keys]); hi = np.array([dec[kk]["hi"] for kk in keys])
        x = np.arange(len(keys)) + (j - 1) * 0.22
        a.bar(x, f, width=0.2, color=CAT[j], label=lab)
        a.errorbar(x, f, yerr=xe(f, lo, hi), fmt="none", ecolor=INK2, elinewidth=0.7)
    a.set_xticks(range(5)); a.set_xticklabels(["all", "$c_R\\leq2$", "$+\\,m_c\\leq10$", "$+\\,f_Rc_R\\geq0.3$", "$+\\,\\pi\\geq100$"], fontsize=6)
    a.set_ylabel("fraction of points with $k/n\\geq0.8$"); a.set_title("nested subsets (post hoc)", loc="left")
    a.legend(fontsize=5.8, loc="upper left"); a.grid(axis="x", visible=False)
    for j, kk in enumerate(["all", "cR<=2", "+mc<=10", "+fRcR>=0.3", "+pi>=100"]):
        a.text(j, -0.06, f"$n$={NUM['lhs_decomp_primary'][kk]['n']}", ha="center", fontsize=5.4, color=INK2, transform=a.get_xaxis_transform())
    panel(a, "b")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_res_region.pdf")
    # efectos del LHS (logística cuasi-binomial) con el primario
    ef = []
    for st in ("lhs", "lhs2"):
        p = RES / f"f2b_{st}" / "analisis" / "lhs_efectos.csv"
        if p.exists():
            e = pd.read_csv(p)
            lg = e[e.familia == "LHS_logit"][["factor", "efecto", "p", "p_bh_familia"]].copy()
            lg["lhs"] = st
            ef.append(lg)
    if ef:
        NUM["lhs_logit"] = pd.concat(ef).round(4).to_dict("records")


# ============================================================================= 10. umbral de clase θ y τ
def fig_theta():
    C = pd.read_csv(RES / "f2b_final" / "v2" / "theta.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.1), sharey=True)
    a = axs[0]
    sel = [("base", "C1", "C1"), ("base", "C1p", "C1$'$"), ("c1c2", "C2", "C2"), ("rob", "C1 async", "C1, asynchronous"),
           ("rob", "C1 nido=3", "C1, nest 3 wk")]
    for j, (st, lab, name) in enumerate(sel):
        d = C[(C.etapa == st) & (C.brazo == lab)].sort_values("theta")
        if len(d):
            a.plot(d.theta, d.O17p, MK[j], ls=LS[j], color=CAT[j], ms=3.5, lw=1.1, label=name)
    a.axhline(0.8, color=MUTED, lw=0.6, ls=":"); a.axvline(0.10, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("class threshold $\\theta$"); a.set_ylabel("O17p re-scored"); a.set_title("candidates", loc="left")
    a.legend(fontsize=5.6, loc="lower left"); a.set_ylim(-0.03, 1.03)
    panel(a, "a")
    a = axs[1]
    sel = [("abl1", "C1", "C1 ($n=120$)"), ("abl1", "C1 -AV", "$-$AV"), ("abl1", "C1 -R1", "$-$R1"), ("abl1", "C1 -R2", "$-$R2"), ("abl1", "C1 -R7", "$-$R7")]
    for j, (st, lab, name) in enumerate(sel):
        d = C[(C.etapa == st) & (C.brazo == lab)].sort_values("theta")
        a.plot(d.theta, d.O17p, MK[j], ls=LS[j], color=CAT[j], ms=3.5, lw=1.1, label=name)
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        a.axhline(tau, color=GRID, lw=0.8)
    a.text(0.205, 0.5, "$\\tau$", fontsize=6, color=INK2, va="center")
    a.axvline(0.10, color=MUTED, lw=0.6, ls=":")
    a.set_xlabel("class threshold $\\theta$"); a.set_title("single ablations: necessity on $(\\theta,\\tau)$", loc="left")
    a.legend(fontsize=5.6, loc="center right")
    panel(a, "b")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_res_theta.pdf")
    NUM["theta"] = C.to_dict("records")
    # necesidad de AV en la grilla: tau mínimo para el que -AV queda por debajo, por θ
    d = C[(C.etapa == "abl1") & (C.brazo == "C1 -AV")].set_index("theta").O17p
    NUM["theta_AV"] = {f"{th:g}": float(v) for th, v in d.items()}


# ============================================================================= 11. sensibilidad
FNAME = {"alpha": "$\\alpha$", "crowd_aversion": "$\\kappa$", "crowd_power": "$p$", "rec": "$r$", "inherit_weight": "$w$",
         "plast_age_max": "$a_w$", "gamma": "$\\gamma$", "a_max": "$a_{\\max}$", "p0": "$p_0$", "delta": "$\\delta$",
         "m_c": "$m_c$", "eta": "$\\eta$", "refuge_frac": "$f_R$", "refuge_pref": "$\\pi$", "refuge_capacity": "$c_R$"}


def fig_sensitivity():
    mo = pd.read_csv(RES / "f2b_morris" / "analisis" / "morris.csv")
    so = pd.read_csv(RES / "f2b_sobol" / "analisis" / "sobol.csv")
    stg = json.load(open(RES / "f2b_sobol" / "stage.json"))
    nrep = int(stg.get("reps_run", stg.get("reps", 0)))
    fig = plt.figure(figsize=(W2, 2.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    m = mo[mo.metric == "f_PRIMARIO"].set_index("name")
    nz = mo[mo.metric == "ruido_PRIMARIO"].set_index("name")["mu_star"]
    order = m.sort_values("mu_star").index
    y = np.arange(len(order))
    ax.barh(y, m.loc[order, "mu_star"], height=0.6, color=CAT[0], label="$\\mu^*$ (O17p)")
    ax.errorbar(m.loc[order, "mu_star"], y, xerr=m.loc[order, "mu_star_conf"], fmt="none", ecolor=INK2, elinewidth=0.7)
    ax.plot(nz.loc[order], y, "|", color=CAT[7], ms=7, mew=1.6, label="noise $\\mu^*$")
    ax.set_yticks(y); ax.set_yticklabels([FNAME[n] for n in order]); ax.set_xlabel("Morris $\\mu^*$")
    ax.set_title("Morris, O17p", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=5.8, ncol=1)
    ax.grid(axis="y", visible=False)
    panel(ax, "a")
    mets = [("f_PRIMARIO", "Sobol, O17p"), ("f_O13p", "persistent O13"), ("x_rho_pk_mean", "$\\rho_{\\rm pk}$"),
            ("x_f0_pk_mean", "$f_0$ at peak")]
    order2 = so[so.metric == "f_PRIMARIO"].set_index("name").sort_values("ST").index
    for i, (mt, ttl) in enumerate(mets):
        a = fig.add_subplot(gs[i + 1])
        d = so[so.metric == mt].set_index("name").reindex(order2)
        yy = np.arange(len(d))
        rel = d["relevante"].astype(str) == "True"
        a.barh(yy, d.ST, height=0.6, color=[CAT[0] if r else "#b7d3f6" for r in rel])
        a.errorbar(d.ST, yy, xerr=d.ST_conf, fmt="none", ecolor=INK2, elinewidth=0.7)
        a.plot(d.S1.clip(lower=0), yy, "o", ms=3, mfc="white", mec=INK, mew=0.8, label="$S_1$")
        a.plot(d.ST_ruido, yy, "|", color=CAT[7], ms=7, mew=1.6, label="noise floor")
        a.set_yticks(yy); a.set_yticklabels([FNAME[n] for n in d.index] if i == 0 else [])
        a.set_xlim(0, max(0.85, float(np.nanmax((d.ST + d.ST_conf).to_numpy(float))) * 1.05)); a.set_xlabel("$S_T$")
        a.set_title(ttl, loc="left"); a.grid(axis="y", visible=False)
        if i == 0:
            a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=5.8, ncol=2)
        panel(a, "bcde"[i])
    save(fig, "fig_res_sensitivity.pdf")
    NUM["morris_f_PRIMARIO"] = m[["mu_star", "mu_star_conf", "sigma", "mu"]].assign(noise=nz).round(4).reset_index().to_dict("records")
    NUM["morris_selection"] = json.load(open(RES / "f2b_morris" / "analisis" / "seleccion_sobol.json"))
    NUM["sobol"] = so[so.metric.isin([x for x, _ in mets] + ["f_O13", "x_BO4_mean", "t_ext_med", "f_O11"])][
        ["metric", "name", "S1", "S1_conf", "ST", "ST_conf", "ST_ruido", "relevante"]].round(4).to_dict("records")
    NUM["sobol_meta"] = dict(factores=stg.get("factors"), N=int(stg.get("n_points", 0)) // (len(stg.get("factors", [])) + 2) if stg.get("factors") else None,
                             n_points=int(stg.get("n_points", 0)), reps=nrep, n_runs=int(len(summ("sobol"))))
    dg = pd.read_csv(RES / "f2b_sobol" / "analisis" / "sobol_diagnostico.csv")
    NUM["sobol_diag"] = dg.round(4).to_dict("records")


# ============================================================================= 12. supervivencia
def fig_survival():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.15), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
    groups1 = [("C1", arm("base", "C1")), ("C1$'$", arm("base", "C1p")), ("C0", arm("base", "C0")), ("C2", arm("c1c2", "C2")),
               ("notebook ($\\alpha=1.4$)", arm("ref", "notebook")), ("long-lived only", arm("ref", "longlived"))]
    out = {}
    for i, (lab, g) in enumerate(groups1):
        km, t, ev = km_of(g)
        disp = {"notebook ($\\alpha=1.4$)": "short-lived control", "long-lived only": "long-lived control"}.get(lab, lab)
        axs[0].step(km.t, km.S, where="post", color=CAT[i], ls=LS[i], label=f"{disp} ($n$={len(g)})")
        if i in (0, 4):
            axs[0].fill_between(km.t, km.lo, km.hi, step="post", color=CAT[i], alpha=0.15, lw=0)
        out[lab] = dict(n=len(g), events=int(ev.sum()), median=survival.km_median(km))
    axs[0].set_xlim(0, 1000); axs[0].set_xlabel("week"); axs[0].set_ylabel("$P$(colony alive)")
    axs[0].set_title("candidates and controls", loc="left"); axs[0].legend(loc="upper right", fontsize=5.6)
    panel(axs[0], "a")
    sA = summ("abl1"); lA = labels("abl1")
    groups2 = [("C1", "C1"), ("$-$R1", "C1 -R1"), ("$-$R2", "C1 -R2"), ("$-$AV", "C1 -AV"), ("$-$R7", "C1 -R7")]
    for i, (lab, l) in enumerate(groups2):
        g = sA[sA.point == lA.index(l)]
        km, t, ev = km_of(g)
        axs[1].step(km.t, km.S, where="post", color=CAT[[0, 4, 5, 6, 7][i]], ls=LS[i], label=f"{lab} ($n$={len(g)})")
        out["abl " + l] = dict(n=len(g), events=int(ev.sum()), median=survival.km_median(km))
    axs[1].set_xlim(0, 600); axs[1].set_xlabel("week"); axs[1].set_title("single-rule ablations of C1", loc="left")
    axs[1].legend(loc="upper right", fontsize=5.8); panel(axs[1], "b")
    s = summ("amax")
    s = s[s.extinct & s["x_D_last"].notna()]
    rng = np.random.default_rng(1)
    x = s.a_max.to_numpy(float); yv = s.x_D_last.to_numpy(float)
    axs[2].plot(x + rng.uniform(-2.5, 2.5, len(x)), yv, "o", ms=2.2, color=CAT[0], alpha=0.45, mew=0, label="runs")
    beta, se, res = f2b.ols(np.column_stack([np.ones_like(x), x]), yv)
    xx = np.linspace(85, 155, 50)
    axs[2].plot(xx, beta[0] + beta[1] * xx, color=CAT[0], lw=1.2, label=f"OLS slope {beta[1]:.3f}")
    nl = np.median(s.x_n_last.fillna(1).to_numpy(float))
    th = [f2b._amax_expected(a, 0.15, nl) for a in xx]
    axs[2].plot(xx, th, color=CAT[1], ls="--", lw=1.2, label=f"$E[\\max]$ of {nl:.0f} lifetimes")
    axs[2].set_xlabel("$a_{\\max}$ (weeks)"); axs[2].set_ylabel("$T_{\\rm ext}-t_{\\rm last}$ (weeks)")
    axs[2].set_title("phase-D law (Q4)", loc="left"); axs[2].legend(loc="upper left", fontsize=5.8)
    panel(axs[2], "c")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_res_survival.pdf")
    NUM["km"] = {k: dict(n=v["n"], events=v["events"], median=v["median"]) for k, v in out.items()}
    NUM["Q4_nlast_median"] = float(nl)
    sv = pd.read_csv(RES / "f2b_base" / "analisis" / "supervivencia.csv")
    NUM["weibull_base"] = sv.to_dict("records")
    # amax: O8 por a_max
    Ta = tab("amax")
    NUM["amax_O8"] = Ta[["label", "f_O8", "f_O17p", "t_ext_med"]].to_dict("records")
    # residuo de Q4 con n_last1 y n_last4
    res_ = {}
    for nc in ("x_n_last", "x_n_last1", "x_n_last4"):
        if nc in s:
            nn = pd.to_numeric(s[nc], errors="coerce").fillna(1).clip(lower=1).to_numpy(float)
            exp_ = np.array([f2b._amax_expected(a, 0.15, n_) for a, n_ in zip(x, nn)])
            r = yv - exp_
            res_[nc] = dict(mean=float(r.mean()), se=float(r.std(ddof=1) / np.sqrt(len(r))), sd=float(r.std(ddof=1)), n_med=float(np.median(nn)))
    NUM["Q4_residual_by_nlast"] = res_
    # dispersión KM de todos los ensambles
    disp = pd.read_csv(RES / "f2b_final" / "v2" / "dispersion.csv")
    NUM["dispersion"] = disp.to_dict("records")
    bl = [l for l in open(RES / "f2b_final" / "v2" / "informe_v2.md", encoding="utf-8").read().splitlines() if l.startswith("- C1′ − longevo")]
    NUM["dispersion_C1p_minus_longlived_bootstrap_median"] = bl[0] if bl else None   # mediana bootstrap (v2.py), no el estimador puntual
    # La diferencia C1′ − longevo se informa como diferencia de los estimadores puntuales,
    # con IC bootstrap percentil (remuestreo de corridas, independiente en cada brazo).
    gp, gl = arm("base", "C1p"), arm("nulos", "null_long")

    def _spread(g):
        km, t, ev = km_of(g)
        q = [survival.km_median(km, q_)["median"] for q_ in (0.75, 0.5, 0.25)]
        return (q[2] - q[0]) / q[1] if all(np.isfinite(q)) and q[1] > 0 else np.nan
    sp_p, sp_l = _spread(gp), _spread(gl)
    rng = np.random.default_rng(2024)
    bs = np.array([_spread(gp.iloc[rng.integers(0, len(gp), len(gp))]) - _spread(gl.iloc[rng.integers(0, len(gl), len(gl))])
                   for _ in range(2000)])
    NUM["dispersion_C1p_minus_longlived"] = dict(C1p=float(sp_p), longlived_CI_C1=float(sp_l), diff=float(sp_p - sp_l),
                                                 lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                                                 boot_median=float(np.nanmedian(bs)), B=2000, nota="estimador puntual; IC percentil")


# ============================================================================= 13. planos
def fig_planes_social():
    T1 = tab("p1"); Ta = tab("p1aw")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.45))
    P = T1[T1.gamma == 0.1].pivot_table(index="inherit_weight", columns="rec", values="f_PRIMARIO")
    heat(axs[0], P, "P1: O17p, $\\gamma=0.1$", xlabel="recovery rate $r$", ylabel="inheritance $w$")
    P0 = T1[T1.gamma == 0.0].pivot_table(index="inherit_weight", columns="rec", values="P_ext")
    heat(axs[1], P0, "P1: $P_{\\rm ext}$, $\\gamma=0$ (no crowding)", xlabel="recovery rate $r$")
    Q = pd.read_csv(RES / "f2b_final" / "exploratorio.csv")
    frn = Q[Q.familia == "frontera_gamma0"]
    phi = float(frn[frn.contraste.str.startswith("φ̂")].efecto.iloc[0])
    c0 = float(frn[frn.contraste.str.startswith("cte")].efecto.iloc[0])
    recs = P0.columns.to_numpy(float); ws = P0.index.to_numpy(float)
    rr = np.linspace(recs.min(), recs.max(), 200)
    wl = c0 - phi * 12 * rr
    xi = np.interp(rr, recs, np.arange(len(recs)))
    ok = (wl >= ws.min()) & (wl <= ws.max())
    yi = np.interp(wl[ok], ws, np.arange(len(ws)))
    axs[1].plot(xi[ok], yi, color=CAT[1], lw=1.6, ls="--")
    axs[1].text(3.6, 3.5, f"fitted boundary\n$w+{phi:.2f}\\,r a_w={c0:.2f}$", color=CAT[1], fontsize=6.3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CAT[1], lw=0.6))
    Ta2 = Ta.copy(); Ta2["aw"] = Ta2.plast_age_max.fillna(np.inf)
    Pa = Ta2.pivot_table(index="aw", columns="rec", values="f_PRIMARIO")
    heat(axs[2], Pa, "P1-$a_w$: O17p", xlabel="recovery rate $r$", ylabel="critical window $a_w$",
         yt=[("$\\infty$" if not np.isfinite(v) else f"{v:g}") for v in Pa.index])
    for i, a in enumerate(axs):
        panel(a, "abc"[i])
        plt.setp(a.get_xticklabels(), rotation=45)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_res_planes_social.pdf")
    NUM["gamma0_frontier"] = dict(phi=phi, c=c0, sigma=1 / c0, phi_norm=phi / c0)
    NUM["p1aw_inf_row"] = Ta[~np.isfinite(Ta.plast_age_max)][["rec", "f_PRIMARIO", "P_ext", "trig_A2"]].to_dict("records")
    # P1-a_w con el evaluador v1 (Q6 depende del evaluador)
    Tv = ptab("p1aw", "v1_")
    Tv2 = Tv.copy(); Tv2["aw"] = Tv2.plast_age_max.fillna(np.inf)
    NUM["p1aw_O17p_primary"] = Pa.round(3).to_dict()
    NUM["p1aw_O17p_v1"] = Tv2.pivot_table(index="aw", columns="rec", values="f_PRIMARIO").round(3).to_dict()
    NUM["p1aw_O11_primary"] = Ta2.pivot_table(index="aw", columns="rec", values="f_O11").round(3).to_dict()
    NUM["p1aw_O13p"] = Ta2.pivot_table(index="aw", columns="rec", values="f_O13p").round(3).to_dict()
    NUM["p1_summary"] = dict(n_points_g01=int((T1.gamma == 0.1).sum()), all_extinct_g01=bool((T1[T1.gamma == 0.1].P_ext == 1).all()),
                             O17p_ge09_w_le03_r_le015=float(T1[(T1.gamma == 0.1) & (T1.inherit_weight <= 0.3) & (T1.rec <= 0.15)].f_PRIMARIO.min()),
                             O17p_w1=T1[(T1.gamma == 0.1) & (T1.inherit_weight == 1)].f_PRIMARIO.round(2).tolist())
    fin = Ta[np.isfinite(Ta.plast_age_max)]
    NUM["p1aw_summary"] = dict(n_finite=int(len(fin)), n_Pext1=int((fin.P_ext == 1).sum()), Pext_min=float(fin.P_ext.min()))
    # Q6 con el modelo aditivo (f2b_final/v2/q6_modelos.csv)
    q6 = pd.read_csv(RES / "f2b_final" / "v2" / "q6_modelos.csv", index_col=0)
    NUM["q6_models"] = q6.reset_index().to_dict("records")


def fig_planes_space():
    T2 = tab("p2"); T5 = tab("p5")
    T2 = T2[np.isclose(T2.crowd_power, 4.0)]
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.5))
    axs = axs.ravel()
    P = T2.pivot_table(index="crowd_aversion", columns="alpha", values="f_PRIMARIO")
    heat(axs[0], P, "P2: O17p", xlabel="attraction $\\alpha$", ylabel="aversion $\\kappa$")
    P = T2.pivot_table(index="crowd_aversion", columns="alpha", values="f_O13")
    heat(axs[1], P, "P2: O13 (instantaneous)", xlabel="attraction $\\alpha$")
    P = T2.pivot_table(index="crowd_aversion", columns="alpha", values="x_f0_pk_mean")
    heat(axs[2], P, "P2: empty cells at peak $f_0$", xlabel="attraction $\\alpha$", vmax=0.6)
    al = P.columns.to_numpy(float); ka = P.index.to_numpy(float)
    for a in axs[:3]:
        for ns, lab in ((4, "4"), (8, "$m_c$")):
            aa = np.linspace(al.min(), al.max(), 200)
            kk = 1 / (ns + 1 / aa)
            ok = (kk >= ka[1]) & (kk <= ka.max())
            xi = np.interp(aa[ok], al, np.arange(len(al)))
            yi = np.interp(kk[ok], ka, np.arange(len(ka)))
            a.plot(xi, yi, color=CAT[1], lw=1.1, ls="--" if ns == 8 else ":")
    for j, cap in enumerate((1, 2)):
        P = T5[T5.refuge_capacity == cap].pivot_table(index="refuge_pref", columns="refuge_frac", values="f_O13p")
        heat(axs[3 + j], P, f"P5: persistent O13, $c_R={cap}$", xlabel="refuge fraction $f_R$",
             ylabel="preference $\\pi$" if j == 0 else "")
    a = axs[5]
    T5 = T5.copy()
    T5["phiR"] = T5.refuge_frac * T5.refuge_capacity * 900 / (T5.x_rho_pk_mean * 7200)
    for i, pref in enumerate(sorted(T5.refuge_pref.unique())):
        d = T5[T5.refuge_pref == pref]
        a.plot(d.phiR, d.x_BO4_mean, MK[i], ms=3.5, color=CAT[i], mew=0, ls="none", label=f"$\\pi$={pref:g}")
    xx = np.linspace(0, T5.phiR.max() * 1.02, 10)
    a.plot(xx, xx, color=MUTED, lw=0.8, ls="--")
    a.axhline(0.1, color=MUTED, lw=0.6, ls=":")
    a.text(0.02, 0.11, "O13 threshold", fontsize=5.6, color=INK2)
    a.set_xlabel("refuge capacity / $N_{\\rm pk}$, $\\phi_R$"); a.set_ylabel("$\\Phi^{\\rm pers}_{\\rm BO}$")
    a.set_title("capacity bound $\\Phi^{\\rm pers}_{\\rm BO}\\leq\\phi_R$", loc="left"); a.legend(fontsize=5.4, loc="upper left")
    for i, a in enumerate(axs):
        panel(a, "abcdef"[i])
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    save(fig, "fig_res_planes_space.pdf")
    NUM["p5_bound_violations"] = int((T5.x_BO4_mean > T5.phiR + 0.01).sum())
    c1 = T5[(T5.refuge_frac == 0.3) & (T5.refuge_capacity == 2) & (T5.refuge_pref == 100)]
    NUM["p5_phiR_C1"] = float(c1.phiR.iloc[0]) if len(c1) else None
    NUM["p5_summary"] = dict(n_points=int(len(T5)), n_O13p_ge08=int((T5.f_O13p >= 0.8).sum()), n_O17p_ge08=int((T5.f_PRIMARIO >= 0.8).sum()),
                             O8_min=float(T5.f_O8.min()) if "f_O8" in T5 else None)
    NUM["p2_summary"] = dict(n_points=int(len(T2)), n_O17p_ge08=int((T2.f_PRIMARIO >= 0.8).sum()),
                             n_alpha_ge2_ge08=int(((T2.alpha >= 2) & (T2.f_PRIMARIO >= 0.8)).sum()), n_alpha_ge2=int((T2.alpha >= 2).sum()),
                             fail_points=T2[T2.f_PRIMARIO < 0.8][["alpha", "crowd_aversion", "f_PRIMARIO", "f_O11", "f_O13", "f_O13p", "f_O5", "f_O4"]].round(2).to_dict("records"))


def fig_p4():
    T = tab("p4").copy()
    T["N0"] = T.n_males + T.n_females
    T["dens"] = T.N0 / T.grid_size ** 2
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0))
    for j, L in enumerate((15, 20, 30)):
        for moore in (0, 1):
            d = T[(T.grid_size == L) & (T.mate_radius == moore)]
            if not len(d):
                continue
            rnd = d[~d.founders_same_cell.astype(bool)]
            fnd = d[d.founders_same_cell.astype(bool)]
            col = CAT[j] if moore == 0 else CAT[4]
            lab = f"$L={L}$" if moore == 0 else f"$L={L}$, Moore mating"
            ls = "-" if moore == 0 else ":"
            for a, y in ((axs[0], "x_rho_pk_mean"), (axs[1], "x_f0_pk_mean")):
                a.plot(rnd.dens, rnd[y], MK[j] if moore == 0 else "v", ls=ls, color=col, ms=3.2, lw=0.9, label=lab)
                a.plot(fnd.dens, fnd[y], MK[j] if moore == 0 else "v", color=col, ms=4.5, mfc="white", mew=1.0, ls="none")
            f, lo, hi = rnd.f_PRIMARIO.to_numpy(float), rnd.lo_PRIMARIO.to_numpy(float), rnd.hi_PRIMARIO.to_numpy(float)
            axs[2].errorbar(rnd.dens * (1 + 0.04 * (j - 1)), f, yerr=xe(f, lo, hi), fmt=MK[j] if moore == 0 else "v", ls=ls,
                            color=col, ms=3.2, lw=0.9, elinewidth=0.7, capsize=0)
    for a in axs:
        a.set_xscale("log"); a.set_xlabel("seeding density $N_0/L^2$")
        a.axvline(400 / 900, color=MUTED, lw=0.6, ls="--")
    axs[0].set_ylabel("$\\rho_{\\rm pk}$"); axs[0].set_title("peak density", loc="left")
    axs[0].axhline(0.57, color=INK2, lw=0.6, ls=":"); axs[0].text(0.011, 0.6, "Universe 25", fontsize=5.8, color=INK2)
    axs[1].set_ylabel("$f_0$"); axs[1].set_title("empty cells at the peak", loc="left")
    axs[1].axhline(0.2, color=INK2, lw=0.6, ls=":")
    axs[2].set_ylabel("O17p"); axs[2].set_title("primary outcome", loc="left"); axs[2].set_ylim(-0.03, 1.03)
    axs[2].axhline(0.8, color=MUTED, lw=0.6, ls=":")
    axs[0].legend(loc="lower right", fontsize=5.4, handlelength=1.6)
    for i, a in enumerate(axs):
        panel(a, "abc"[i])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_res_p4.pdf")
    from scipy import stats as st
    mm = (~T.founders_same_cell.astype(bool)) & (T.mate_radius == 0)
    NUM["p4"] = T[["grid_size", "N0", "founders_same_cell", "mate_radius", "dens", "f_PRIMARIO", "lo_PRIMARIO", "hi_PRIMARIO",
                   "f_O13p", "f_O11", "f_O4", "P_ext", "x_rho_pk_mean", "x_f0_pk_mean", "x_BO4_mean"]].round(4).to_dict("records")
    NUM["p4_spearman_rho_dens"] = float(st.spearmanr(T[mm].dens, T[mm].x_rho_pk_mean).statistic)
    ok = T[mm & (T.f_PRIMARIO >= 0.8)]
    NUM["p4_summary"] = dict(dens_ok_min=float(ok.dens.min()), dens_ok_max=float(ok.dens.max()), rho_min=float(T[mm].x_rho_pk_mean.min()),
                             rho_max=float(T[mm].x_rho_pk_mean.max()),
                             founders_rho=T[T.founders_same_cell.astype(bool)][["grid_size", "mate_radius", "x_rho_pk_mean", "f_PRIMARIO"]].round(3).to_dict("records"))


def table_p6():
    s6 = run("p6")
    S6 = s6["summary"]
    T6 = tab("p6")
    r6 = []
    for pt, gg in S6.groupby("point"):
        ok = (gg.obj_O1 == "PASS") & (gg.obj_O2 == "PASS") & (gg.obj_O3 == "PASS")
        full = ok & (gg.obj_O17p == "PASS")
        lo, hi = wilson(int(full.sum()), len(gg))
        rr = T6[T6.point == pt].iloc[0]
        r6.append(dict(label=s6["labels"][pt], n=len(gg), f_O1=float(rr.f_O1), f_O2=float(rr.f_O2), f_O3=float(rr.f_O3), f_O123=float(ok.mean()),
                       k_full=int(full.sum()), f_full=float(full.mean()), lo_full=lo, hi_full=hi, f_O17p=float(rr.f_PRIMARIO),
                       lo_O17p=float(rr.lo_PRIMARIO), hi_O17p=float(rr.hi_PRIMARIO), f_O13p=float(rr.f_O13p), P_ext=float(rr.P_ext),
                       rho_pk=float(rr.x_rho_pk_mean), f0=float(rr.x_f0_pk_mean)))
    NUM["p6"] = r6
    out = []
    for r in r6:
        kv = dict(x.split("=") for x in r["label"].split()[1:])
        out.append(f"{float(kv['s0']):.1f} & {kv['k']} & {float(kv['a_max']):.0f} & {r['f_O1']:.2f} & {r['f_O123']:.2f} & "
                   f"{r['f_O17p']:.2f} [{r['lo_O17p']:.2f}, {r['hi_O17p']:.2f}] & {r['f_O13p']:.2f} & {r['P_ext']:.2f} & "
                   f"{r['rho_pk']:.2f} & {r['f0']:.2f} \\\\")
    write_tab("tab_res_p6.tex", out)


def fig_cohorts():
    s = summ("c1c2")
    iC1, iC2 = pidx("c1c2", "C1"), pidx("c1c2", "C2")
    ser = pd.read_parquet(RES / "f2b_base" / "series.parquet", columns=["point", "rep", "t", "births"])
    sb = summ("base")
    b0 = pidx("base", LAB["C1"])
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 2.3))
    a = axs[0]
    for j, (pt, lab) in enumerate(((iC1, "C1"), (iC2, "C2"))):
        g = s[s.point == pt]
        jit = np.random.default_rng(j).uniform(-0.3, 0.3, (2, len(g)))
        a.plot(g.o13b_med_bt_cr + jit[0], g.o13b_med_bt_iso + jit[1], MK[j], ms=2.6, color=CAT[j], alpha=0.6, mew=0,
               ls="none", label=f"{lab} ($n$={len(g)})")
    lim = [0, max(40, float(s[["o13b_med_bt_cr", "o13b_med_bt_iso"]].max().max()) + 2)]
    a.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel("median birth week, crowded class"); a.set_ylabel("median birth week, isolated class")
    a.set_title("birth time of the two classes", loc="left"); a.legend(loc="upper left", fontsize=6)
    panel(a, "a")
    a = axs[1]
    g = ser[ser.point == b0]
    tp = sb[sb.point == b0].set_index("rep").x_tpk2
    cur = []
    grid = np.linspace(0, 2.0, 201)
    for rep, h in g.groupby("rep"):
        h = h.sort_values("t")
        b = h.births.to_numpy(float)
        if b.sum() <= 0:
            continue
        c = np.cumsum(b) / b.sum()
        cur.append(np.interp(grid, h.t.to_numpy(float) / tp.loc[rep], c, left=0, right=1))
    cur = np.array(cur)
    a.plot(grid, np.median(cur, 0), color=CAT[0], lw=1.3, label="C1, median over runs")
    a.fill_between(grid, np.percentile(cur, 10, 0), np.percentile(cur, 90, 0), color=CAT[0], alpha=0.18, lw=0)
    a.axvline(0.8, color=CAT[1], lw=0.9, ls=":")
    a.text(0.82, 0.25, "O13b window\n($t>0.8\\,t'_{\\rm pk}$)", fontsize=5.8, color=CAT[1])
    a.set_xlabel("time / $t'_{\\rm pk}$ (start of plateau)"); a.set_ylabel("cumulative fraction of births")
    a.set_title("births are front-loaded", loc="left")
    panel(a, "b")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_res_cohorts.pdf")
    med = np.array([np.interp(0.5, c, grid) for c in cur])
    g1, g2 = s[s.point == iC1], s[s.point == iC2]
    NUM["cohorts"] = dict(C1_med_iso=float(g1.o13b_med_bt_iso.median()), C1_med_cr=float(g1.o13b_med_bt_cr.median()),
                          C2_med_iso=float(g2.o13b_med_bt_iso.median()), C2_med_cr=float(g2.o13b_med_bt_cr.median()),
                          C1_frac_iso_later=float((g1.o13b_med_bt_iso > g1.o13b_med_bt_cr).mean()),
                          C2_frac_iso_later=float((g2.o13b_med_bt_iso > g2.o13b_med_bt_cr).mean()),
                          C1_births_median_over_tpk2=float(np.median(med)),
                          C1_births_before_08tpk2=float(np.median(cur[:, np.searchsorted(grid, 0.8)])),
                          C1_births_before_1tpk2=float(np.median(cur[:, np.searchsorted(grid, 1.0)])))


# ============================================================================= 13b. tabla de ensambles
ENS_DESC = {"base": ("reference ensemble: C1, C1$'$, C0, $\\alpha=0$", "production"), "c1c2": ("C1 and C2", "QC"),
            "ref": ("short-lived and long-lived controls", "Q3, controls"), "nulos": ("four null models", "discrimination"),
            "abl1": ("single ablations and variants", "Q2"), "abl2": ("double to quadruple ablations, $\\gamma=0$", "ablations"),
            "trans": ("8 transplant arms, 11 phases", "Q1, O12"), "amax": ("$a_{\\max}\\in\\{90,\\dots,150\\}$", "Q4"),
            "rob": ("demographic and update variants", "robustness"), "ci": ("non-synchronous initial condition", "robustness"),
            "size": ("$L\\in\\{15,\\dots,60\\}$ at fixed density", "size"), "q7aw": ("$a_w$ varied at $\\gamma=0$", "Q7"),
            "p1": ("plane $(r,w)\\times\\gamma$", "Q7, P1"), "p1aw": ("plane $(r,a_w)$", "Q6"), "p2": ("plane $(\\alpha,\\kappa)$, $p$", "Q5, P2"),
            "p5": ("refuge plane", "P5"), "lhs": ("Latin hypercube 1", "global"), "lhs2": ("Latin hypercube 2", "global"),
            "lhsx": ("boundary points of the hypercubes", "global"), "morris": ("Morris screening", "global"),
            "sobol": ("Saltelli design, 11 factors", "global"), "robmu": ("$\\mu_{\\rm bg}$ curve", "robustness"),
            "p4": ("seeding density and lattice", "P4"), "trgrid": ("transplants on $(\\alpha,\\kappa)$", "O12"),
            "refine": ("40 boundary points of the planes", "planes"), "p6": ("founder protocol", "O1--O3")}
ENS_ORDER = ["base", "c1c2", "ref", "nulos", "abl1", "abl2", "trans", "amax", "rob", "ci", "size", "q7aw", "p1", "p1aw", "p2", "p5",
             "lhs", "lhs2", "lhsx", "morris", "sobol", "robmu", "p4", "trgrid", "refine", "p6"]


def table_ensembles():
    E = pd.read_csv(RES / "f2b_final" / "v2" / "ensambles.csv")
    out, recs = [], []
    for st in ENS_ORDER:
        d = RES / f"f2b_{st}"
        stj = json.load(open(d / "stage.json")) if (d / "stage.json").exists() else {}
        e = E[E.etapa == f"f2b_{st}"]
        if len(e):
            npts = len(e) if not str(e.brazo.iloc[0]).endswith("puntos") else int(str(e.brazo.iloc[0]).split()[0])
            nrun = int(e.n.sum()); T = int(e["T"].max()); seed = int(e.semilla_base.iloc[0])
            per = int(round(nrun / npts))
        else:   # transplantes: sin summary.csv
            tr = pd.read_csv(d / "transplant.csv")
            npts = int(tr.point.nunique()); per = int(tr.groupby("point").rep.nunique().max()); nrun = int(npts * per)
            T = int(stj.get("T", 0)); seed = int(stj.get("base_seed", 0))
        desc, use = ENS_DESC.get(st, (st, ""))
        out.append(f"\\texttt{{{st}}} & {desc} & {npts} & {per} & {nrun} & {T} & {seed} & {use} \\\\")
        recs.append(dict(stage=st, points=npts, per_point=per, runs=nrun, T=T, seed=seed))
    write_tab("tab_res_ensembles.tex", out)
    NUM["ensembles_table"] = recs


# ============================================================================= 14. números sueltos
def numbers_extra():
    """Esquina de Q7, meseta, refine, trgrid, ensambles, presupuesto. Todo desde salidas ya guardadas."""
    s = summ("p1")
    d = pd.read_csv(RES / "f2b_p1" / "design.csv")
    s = s.merge(d, on="point", suffixes=("", "_d"))
    g = s[(s.gamma == 0) & (s.inherit_weight <= 0.1) & (s.rec <= 0.03)]
    NUM["q7_corner"] = dict(n=int(len(g)), rho_pk_median_by_point=g.groupby(["inherit_weight", "rec"]).x_rho_pk.median().round(3).tolist(),
                            rho_pk_min=float(g.x_rho_pk.min()), rho_pk_max=float(g.x_rho_pk.max()),
                            N_max_median_by_point=g.groupby(["inherit_weight", "rec"]).N_max.median().tolist(),
                            t_pk_median_by_point=g.groupby(["inherit_weight", "rec"]).t_peak.median().tolist(),
                            O4_pass=float((g.obj_O4 == "PASS").mean()), O17p_pass=float((g.obj_O17p == "PASS").mean()))
    g0 = s[s.gamma == 0]
    NUM["p1_gamma0_boundary"] = g0.groupby(["inherit_weight", "rec"]).extinct.mean().round(2).reset_index().to_dict("records")
    # meseta del pico (C1 y C0 del ensamble de referencia), a partir de la serie completa
    ser = pd.read_parquet(RES / "f2b_base" / "series.parquet", columns=["point", "rep", "t", "N"])
    sb = summ("base").set_index(["point", "rep"])
    rows = []
    for key in ("C1", "C0"):
        pt = pidx("base", LAB[key])
        for (p_, rep), h in ser[ser.point == pt].groupby(["point", "rep"]):
            h = h.sort_values("t"); N = h.N.to_numpy(); tt = h.t.to_numpy(); i = int(N.argmax())
            a98 = tt[N >= 0.98 * N[i]]
            te = sb.loc[(pt, rep), "t_ext"] if bool(sb.loc[(pt, rep), "extinct"]) else np.nan
            rows.append(dict(cand=key, tpk=tt[i], tpk98=a98.min(), tend98=a98.max(), te=te, tB0=sb.loc[(pt, rep), "t_B0"],
                             tlast=sb.loc[(pt, rep), "x_t_lastbirth"], tB99=sb.loc[(pt, rep), "x_tB99"]))
    D = pd.DataFrame(rows)
    D["dshift"] = D.tpk - D.tpk98
    D["rho8"] = (D.te - D.tpk) / D.tpk
    D["rho8p"] = (D.te - D.tpk98) / D.tpk98
    q = lambda x: [float(np.nanpercentile(x, v)) for v in (25, 50, 75)]  # noqa: E731
    d0 = D[D.cand == "C1"]
    NUM["plateau_C1"] = dict(tpk=q(d0.tpk), tpk98=q(d0.tpk98), tend98=q(d0.tend98), width98=q(d0.tend98 - d0.tpk98),
                             shift=q(d0.dshift), shift_rel=q(d0.dshift / d0.tpk), rho8=q(d0.rho8), rho8p=q(d0.rho8p),
                             tpk_eq_tlast=float((d0.tpk == d0.tlast).mean()), tB99_over_tpk98=q(d0.tB99 / d0.tpk98),
                             tlast_over_tpk=q(d0.tlast / d0.tpk), ensamble="base:C1")
    # refine: 100 corridas en 40 puntos de borde frente a las 30 originales
    Rf = tab("refine"); P1 = tab("p1"); P2 = tab("p2"); Pa = tab("p1aw")
    rr = []
    for _, r in Rf.iterrows():
        src = str(r.label).split(":")[0]
        if src == "f2b_p1aw":
            aw = pd.to_numeric(Pa.plast_age_max, errors="coerce")
            m = Pa[np.isclose(Pa.rec, r.rec) & ((aw == r.plast_age_max) | (~np.isfinite(aw) & ~np.isfinite(r.plast_age_max)))]
        elif src == "f2b_p1":
            m = P1[np.isclose(P1.rec, r.rec) & np.isclose(P1.inherit_weight, r.inherit_weight) & np.isclose(P1.gamma, r.gamma)]
        else:
            m = P2[np.isclose(P2.alpha, r.alpha) & np.isclose(P2.crowd_aversion, r.crowd_aversion) & np.isclose(P2.crowd_power, r.crowd_power)]
        if not len(m):
            continue
        m = m.iloc[0]
        rr.append(dict(label=r.label, f30=float(m.f_PRIMARIO), f100=float(r.f_PRIMARIO), lo=float(r.lo_PRIMARIO), hi=float(r.hi_PRIMARIO)))
    RR = pd.DataFrame(rr)
    NUM["refine"] = dict(n_points=int(len(RR)), n_p1=int(RR.label.str.startswith("f2b_p1:").sum()), n_p1aw=int(RR.label.str.startswith("f2b_p1aw:").sum()),
                         n_p2=int(RR.label.str.startswith("f2b_p2:").sum()),
                         mean_abs_diff=float((RR.f100 - RR.f30).abs().mean()), mean_diff=float((RR.f100 - RR.f30).mean()),
                         max_abs_diff=float((RR.f100 - RR.f30).abs().max()),
                         class_changes=int(((RR.f30 >= 0.8) != (RR.f100 >= 0.8)).sum()),
                         class_changes_up=int(((RR.f30 < 0.8) & (RR.f100 >= 0.8)).sum()),
                         f30_in_ci=int(((RR.f30 >= RR.lo) & (RR.f30 <= RR.hi)).sum()),
                         corr=float(np.corrcoef(RR.f30, RR.f100)[0, 1]), points=RR.round(3).to_dict("records"))
    TG = pd.read_csv(RES / "f2b_trgrid" / "analisis" / "tabla_brazos.csv")
    NUM["trgrid"] = dict(k_D=int(TG.k_D.sum()), n_D=int(TG.n_D.sum()), k_mix=int((TG.k_MIX_M + TG.k_MIX_F).sum()),
                         n_mix=int((TG.n_MIX_M + TG.n_MIX_F).sum()), PB_min=float(TG.P_T_B.min()), PB_max=float(TG.P_T_B.max()),
                         O12_pass=int((TG.O12 == "PASS").sum()), n_points=int(len(TG)),
                         PB_by_alpha=TG.assign(a=TG.label.str.extract(r"a=(\d+)")[0].astype(int)).groupby("a").P_T_B.mean().round(3).to_dict())
    # tabla de ensambles
    E = pd.read_csv(RES / "f2b_final" / "v2" / "ensambles.csv")
    NUM["ensembles"] = E.to_dict("records")
    # presupuesto: corridas y tiempo de pared registrado
    bud = {}
    for f in sorted(RES.glob("f2b_*/progress.log")):
        stg = f.parent.name
        txt = f.read_text(errors="ignore")
        mins = sum(float(m) for m in re.findall(r"fin de la invocación: \d+/\d+ hechas, \d+ fallas, ([\d.]+) min", txt))
        if stg in ("f2b_trans", "f2b_trgrid"):
            mins = sum(int(m) for m in re.findall(r"\((\d+) s\)", txt)) / 60
        nrun = 0
        if (f.parent / "summary.csv").exists():
            nrun = int(len(pd.read_csv(f.parent / "summary.csv", usecols=["point", "rep"]).drop_duplicates()))
        if (f.parent / "transplant.csv").exists():
            tr = pd.read_csv(f.parent / "transplant.csv")
            nrun = int(tr.groupby("point").rep.nunique().sum())
        bud[stg] = dict(minutes=round(mins, 2), runs=nrun)
    NUM["budget"] = bud
    NUM["budget_total"] = dict(runs=int(sum(v["runs"] for k, v in bud.items() if k not in ("f2b_trans", "f2b_trgrid"))),
                               transplant_reps=int(sum(v["runs"] for k, v in bud.items() if k in ("f2b_trans", "f2b_trgrid"))),
                               minutes=round(sum(v["minutes"] for v in bud.values()), 1))
    # nulos: O17p y P_ext
    Tn = tab("nulos")
    NUM["nulos"] = Tn[["label", "n", "f_PRIMARIO", "f_O4", "f_O5", "f_O11", "f_O13p", "P_ext", "x_rho_pk_mean", "x_f0_pk_mean", "t_ext_med"]].round(3).to_dict("records")
    # O17p(θ) del LHS (umbral de clase), re-puntuado con las fracciones por instante (m > 8)
    thr = {}
    for st in ("lhs", "lhs2"):
        s_ = summ(st)
        for th in (0.05, 0.10, 0.15):
            vals = []
            for pt, g in s_.groupby("point"):
                v = v2mod.o17p_theta(g, th)
                vals.append(v[0] if v else np.nan)
            thr.setdefault(f"{th:.2f}", []).extend(vals)
    NUM["threshold_lhs"] = {k: float(np.nanmean(np.array(v) >= 0.8)) for k, v in thr.items()}
    # alpha=0 falla O4 v2 por f0
    a0 = arm("base", "a0")
    NUM["alpha0_f0_mean"] = float(pd.to_numeric(a0.x_f0_pk, errors="coerce").mean())


def numbers_v3():
    """Cifras que no salen de results_v2/f2b_*/: O11 a la edad a_w
    (results_v2/o11_aw/<etapa>/summary.csv, `make o11-aw`; O11 y O17p a la edad 6 del summary de la etapa de origen)
    y Γ sin dilución (results/fisico_v3/gamma_v3.parquet, scripts/analysis/gamma_v3.py; mismas definiciones que
    analiza_gamma.py: Γ = mediana(t_1/2 - t_on) / mediana(T_gen))."""
    o11 = ROOT / "results_v2" / "o11_aw"
    rows = []
    for st in ("f2b_base", "f2b_nulos", "f2b_rob"):
        new = pd.read_csv(o11 / st / "summary.csv", usecols=["point", "rep", "obj_O11", "obj_O17p"])
        old = pd.read_csv(ROOT / "results_v2" / st / "summary.csv", usecols=["point", "rep", "obj_O11", "obj_O17p"])
        lab = json.loads((o11 / st / "labels.json").read_text())
        m = new.merge(old, on=["point", "rep"], suffixes=("_aw", "_6"), validate="1:1")
        for pt, g in m.groupby("point"):
            r = dict(stage=st, point=int(pt), arm=lab[int(pt)], n=int(len(g)))
            for c in ("O11_6", "O11_aw", "O17p_6", "O17p_aw"):
                k = int((g[f"obj_{c}"] == "PASS").sum())
                lo, hi = wilson(k, len(g))
                r[c] = dict(k=k, f=round(k / len(g), 3), ci=[round(lo, 3), round(hi, 3)])
            r["O11_aw_NA"] = int((g.obj_O11_aw == "NA").sum())
            rows.append(r)
    NUM["o11_aw"] = dict(arms=rows, fuente="results_v2/o11_aw/<etapa>/summary.csv (make o11-aw); edad 6: "
                         "results_v2/<etapa>/summary.csv")
    gp = ROOT / "results" / "fisico_v3" / "gamma_v3.parquet"
    D = pd.read_parquet(gp)
    q = lambda x: [round(float(v), 3) for v in np.nanpercentile(x, [25, 50, 75])]  # noqa: E731
    G = []
    for lab_, g in D.groupby("label", sort=False):
        r = dict(label=lab_, block=g.block.iloc[0], n=int(len(g)))
        for c in ("t_on", "t_ad", "t_aw", "t_f0", "T_gen", "gB", "Rnet0", "Rnet1"):
            r[c] = round(float(g[c].median()), 3)
        for k, c in (("ad", "t_ad"), ("aw", "t_aw"), ("f0", "t_f0")):
            td = (g[c] - g.t_on).to_numpy(float)
            tg = g.T_gen.to_numpy(float)
            ok = np.isfinite(td) & np.isfinite(tg)
            r[f"Gamma_{k}"] = round(float(np.median(td[ok]) / np.median(tg[ok])), 3) if ok.sum() >= 5 else None
            r[f"Gamma_{k}_runs_q25_q50_q75"] = q(td[ok] / tg[ok]) if ok.sum() >= 5 else None
        for c in ("O17p", "O10", "O10_aw", "O10b", "O10b_aw", "O10b_f0"):
            r[f"{c}_pass"] = round(float((g[c] == "PASS").mean()), 3)
        G.append(r)
    NUM["gamma_v3"] = dict(rows=G, fuente=str(gp.relative_to(ROOT)) + " (scripts/analysis/gamma_v3.py)",
                           definicion="t_1/2: primera semana con s media < 0.5 en adultos (edad >= a_min), animales "
                                      "fuera de la ventana (a_w actualizaciones cumplidas) o fundadores; t_on: primera "
                                      "semana con f_over >= 0.05; T_gen = med_F2 - med_F1")


ALL_FN = None


if __name__ == "__main__":
    fallas = []
    only = [x for x in os.environ.get("FIGS_ONLY", "").split(",") if x]
    if only and NUMS.exists():
        NUM.update(json.loads(NUMS.read_text()))
    for fn in (fig_trajectories, table_production, table_c1p, fig_ablations, table_Q, table_QC, table_null, table_rescoring,
               table_rob, table_size, fig_rob, table_trans, fig_transplant, q7aw, region, fig_theta, fig_sensitivity,
               fig_survival, fig_planes_social, fig_planes_space, fig_p4, table_p6, fig_cohorts, table_ensembles, numbers_extra,
               numbers_v3):
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fallas.append(fn.__name__)
            print(f"AVISO: {fn.__name__} no se generó ({type(e).__name__}: {e})", file=sys.stderr)
            traceback.print_exc(limit=3, file=sys.stderr)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return str(o)
    if fallas:   # con fallas no se toca results_numbers.json (quedaría sin las claves de esas salidas)
        print(f"{len(fallas)} salidas no generadas: {', '.join(fallas)}; {NUMS.name} NO se reescribió", file=sys.stderr)
        sys.exit(1)
    NUMS.write_text(json.dumps(NUM, indent=1, default=conv, ensure_ascii=False))
    print("nums", NUMS)
