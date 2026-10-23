"""Tarea 2 (ítem 1, E2/E3): búsqueda de pareja en la variante bien mezclada.

Compara apareamiento en la celda (u25), en el bloque de Moore (mate_radius = 1) y global (padre uniforme en la arena),
para SL (control) y C1 (sin refugios), en el ABM y en la ecuación cinética.
Entradas: results_em/mate_search/summary.csv (SL), results_em/adult_recovery/summary.csv (C1 en
lambda = 1, out = 0), results_em/kinetic_grid/ (no se usa).
Salida: results_em/mate_search/tabla_mate.csv; results_em/extra_figs/fig_mate_search.pdf (+ png)
Uso: python3 scripts/em/f7_analyze_mate.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT
from kinetic import params_SL, params_C1
from kinetic_f7 import KineticF7
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MATE = os.path.join(ROOT, "results_em", "mate_search")
E1 = os.path.join(ROOT, "results_em", "adult_recovery")
FIG = os.path.join(ROOT, "results_em", "extra_figs")   # figuras de otro artículo (no usadas en el de EM)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
COL = {"cell": "#2a78d6", "moore": "#eb6834", "global": "#1baf7a"}
LAB = {"cell": "mate in the cell", "moore": "mate in the Moore block", "global": "global mating"}
SHORT = {"cell": "cell", "moore": "Moore block", "global": "global"}


def main():
    sl = pd.read_csv(os.path.join(MATE, "summary.csv"))
    sl["mate"] = sl.cfg.str.split("_").str[2]
    e1 = pd.read_csv(os.path.join(E1, "summary.csv"))
    c1 = e1[(e1.space == "wm") & (e1.lam == 1.0) & (e1.out == 0.0)].copy()
    c1["mate"] = c1.variant
    tab = pd.concat([sl.assign(model="SL"), c1.assign(model="C1")])
    keep = ["model", "mate", "L", "n", "n_ext", "P_ext_300", "P_ext_1000", "Text_med", "Text_med_lo", "Text_med_hi",
            "tlast_med", "late_k", "late_risk", "late_rate", "late_rate_lo", "late_rate_hi"]
    tab[[c for c in keep if c in tab]].sort_values(["model", "mate", "L"]).to_csv(os.path.join(MATE, "tabla_mate.csv"), index=False)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.55))
    # (a) cinética: nu(t) de SL con las tres reglas y de C1 con apareamiento global
    ax = axs[0]
    for mate, pair in (("cell", "shared"), ("moore", "moore"), ("global", "global")):
        df = KineticF7(params_SL(), M=100, mode="closure", pair=pair).run(600, stop_nu=1e-12)
        ax.semilogy(df.t, df.nu, color=COL[mate], lw=1.1, label=f"SL, {SHORT[mate]}")
    df = KineticF7(params_C1(refuge_frac=0.0, refuge_pref=0.0, refuge_hard=False), M=100, mode="closure", pair="global").run(600, stop_nu=1e-12)
    ax.semilogy(df.t, df.nu, color="#555", lw=1.1, ls="--", label="C1, global")
    ax.axhline(1 / 900, color="#999", lw=0.6, ls=":")
    ax.set_ylim(1e-6, 30); ax.set_xlabel("week"); ax.set_ylabel(r"density $\nu_t$ (per cell)")
    ax.set_title("(a) kinetic equation, well mixed")
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.5)
    # (b) SL ABM: P(extinct by week 300) y mediana de T_ext frente a |G|
    ax = axs[1]
    for j, (mate, mk) in enumerate((("cell", "o"), ("moore", "s"), ("global", "^"))):
        g = sl[sl.mate == mate].sort_values("L")
        xg = g.L ** 2 * (1 + 0.06 * (j - 1))
        ax.errorbar(xg, g.P_ext_300, yerr=[g.P_ext_300 - g.P_ext_300_lo, g.P_ext_300_hi - g.P_ext_300],
                    fmt=mk + "-", ms=3, lw=1, color=COL[mate], label=SHORT[mate], capsize=1.5)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("$|G|$ (cells)"); ax.set_ylabel("P(extinct by week 300)")
    ax.set_title("(b) well-mixed SL")
    ax.legend(frameon=False, fontsize=5.8, loc="center right", bbox_to_anchor=(1.0, 0.58), title="mating", title_fontsize=5.8)
    # (c) tasa de extinción tardía de SL con apareamiento global frente a |G|
    ax = axs[2]
    g = sl[sl.mate == "global"].sort_values("L")
    m = g[g.late_k > 0]; u = g[(g.late_k == 0) & (g.late_risk > 0)]
    ax.errorbar(m.L ** 2, m.late_rate, yerr=[m.late_rate - m.late_rate_lo, m.late_rate_hi - m.late_rate], fmt="^",
                ms=3.5, color=COL["global"], capsize=1.5, label="SL, global mating: rate")
    ax.plot(u.L ** 2, u.late_rate_hi, "v", mfc="none", ms=4, color=COL["global"], label="95% upper bound (no event)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$|G|$ (cells)"); ax.set_ylabel("late extinction rate (per week)")
    ax.set_title("(c) well-mixed SL, global mating")
    ax.set_ylim(1e-5, 2e-1)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left")
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_mate_search.{ext}"), dpi=300, bbox_inches="tight")
    print(tab[[c for c in keep if c in tab]].sort_values(["model", "mate", "L"]).round(4).to_string())


if __name__ == "__main__":
    main()
