"""Tarea 1 (E1) y 2 (búsqueda de pareja): análisis y figuras.

Entradas: results_em/adult_recovery/summary.csv (f7_aggregate.py), results_em/kinetic_grid/e1_*.jsonl (opcional; sólo para la figura),
          results_em/mate_search/summary.csv.
Salidas:
  results_em/adult_recovery/transicion.csv   umbral out_c(lambda, L) (P_ext(2000) = 1/2, ajuste logístico)
  results_em/adult_recovery/escaleo.csv      pendiente de T_ext con ln|G| y de la tasa tardía con |G|
  results_em/extra_figs/fig_adult_recovery.pdf (+ png)
  results_em/extra_figs/fig_mate_search.pdf (+ png)
Uso: python3 scripts/em/f7_analyze_e1.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

E1 = os.path.join(ROOT, "results_em", "adult_recovery")
KIN = os.path.join(ROOT, "results_em", "kinetic_grid")
MATE = os.path.join(ROOT, "results_em", "mate_search")
FIG = os.path.join(ROOT, "results_em", "extra_figs")   # figuras de otro artículo (no usadas en el de EM)
H = 1000
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COLS = {15: "#9ecae1", 30: "#4292c6", 60: "#08306b"}
LAMC = {1.0: "#b2182b", 0.95: "#ef8a62", 0.9: "#f4a582", 0.8: "#92c5de", 0.5: "#4393c3", 0.0: "#2166ac"}


def kin_table(pair):
    fn = os.path.join(KIN, f"e1_{pair}.jsonl")
    if not os.path.exists(fn):
        return None
    rows = []
    for line in open(fn):
        r = json.loads(line)
        p = r["point"]
        rows.append(dict(lam=p["social_rec_lambda"], out=p["plast_outside_rec"], cls=r["cls"],
                         Tfl30=r.get("Tfl_L30", -1), Tfl60=r.get("Tfl_L60", -1), tl30=r.get("tlast_fl_L30", -1),
                         valley=r.get("nu_min_late", np.nan), Text_fl=r.get("t_ext_fluid", np.nan)))
    return pd.DataFrame(rows)


def logistic_fit(x, k, n):
    """P(ext) = 1/(1+exp((x - x0)/w)) por máxima verosimilitud binomial; devuelve (x0, w)."""
    x, k, n = map(np.asarray, (x, k, n))
    if k.sum() == 0 or k.sum() == n.sum():
        return np.nan, np.nan

    def nll(th):
        x0, lw = th
        p = 1 / (1 + np.exp((x - x0) / np.exp(lw)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    best = None
    for x0 in np.linspace(x.min(), x.max(), 9):
        r = minimize(nll, [x0, np.log(0.05)], method="Nelder-Mead")
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1]))


def transition(df):
    rows = []
    for (sp, var, lam, L), g in df.groupby(["space", "variant", "lam", "L"]):
        g = g.sort_values("out")
        k = np.round(g[f"P_ext_{H}"] * g.n).astype(int)
        x0, w = logistic_fit(g.out.values, k.values, g.n.values)
        rows.append(dict(space=sp, variant=var, lam=lam, L=L, out_c=x0, width=w, n_points=len(g)))
    return pd.DataFrame(rows)


def scaling(df):
    rows = []
    for (sp, var, lam, out), g in df.groupby(["space", "variant", "lam", "out"]):
        g = g.sort_values("L")
        if g.L.nunique() < 2:
            continue
        G = g.L.values ** 2
        r = dict(space=sp, variant=var, lam=lam, out=out, P_ext=list(np.round(g[f"P_ext_{H}"].values, 2)),
                 Ls=list(g.L.values), Text_med=list(g.Text_med.values))
        ok = np.isfinite(g.Text_med.values) & (g[f"P_ext_{H}"].values >= 0.9)
        if ok.sum() >= 2:
            r["b_lnG"] = float(np.polyfit(np.log(G[ok]), g.Text_med.values[ok], 1)[0])
        lr = g.late_rate.values
        okr = np.isfinite(lr) & (lr > 0)
        if okr.sum() >= 2:
            r["c_late"] = float(-np.polyfit(G[okr], np.log(lr[okr]), 1)[0])
        # clase empírica
        p = g[f"P_ext_{H}"].values
        if np.all(p >= 0.95):
            r["clase"] = "colapso"
        elif p[-1] <= 0.05 and p[-1] <= p[0]:
            r["clase"] = "persistencia"
        elif p[-1] < p[0]:
            r["clase"] = "metaestable (P_ext baja con |G|)"
        else:
            r["clase"] = "mixta"
        rows.append(r)
    return pd.DataFrame(rows)


def fig_adult_recovery(df, kin):
    LAMS_SEQ = {0.0: "#9ecae1", 0.5: "#4292c6", 0.9: "#2171b5", 1.0: "#08306b"}
    LS = {15: ":", 30: "--", 60: "-"}
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.62)
    # (a) cinética
    ax = fig.add_subplot(gs[0, 0])
    if kin is not None:
        lams = sorted(kin.lam.unique()); outs = sorted(kin.out.unique())
        code = {"esteril": 0, "allee": 1, "ciclo": 2, "fijo": 3}
        Z = np.full((len(lams), len(outs)), np.nan)
        for _, r in kin.iterrows():
            Z[lams.index(r.lam), outs.index(r.out)] = code[r.cls]
        cm = ListedColormap(["#67000d", "#fb6a4a", "#9ecae1", "#08519c"])
        ax.pcolormesh(np.arange(len(outs) + 1), np.arange(len(lams) + 1), Z, cmap=cm, vmin=-0.5, vmax=3.5,
                      edgecolors="w", linewidth=0.3)
        ax.set_xticks(np.arange(len(outs))[::2] + 0.5); ax.set_xticklabels([f"{o:g}" for o in outs][::2], rotation=90)
        ax.set_yticks(np.arange(len(lams)) + 0.5); ax.set_yticklabels([f"{l:g}" for l in lams])
        ax.set_xlabel(r"adult recovery factor $\epsilon_{\rm ad}$")
        ax.set_ylabel(r"social fraction of learning $\lambda$")
        ax.set_title("(a) fluid limit (well mixed)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cm.colors[:3], ["non-breeding", "Allee", "cycle"])],
                  loc="lower left", bbox_to_anchor=(0.0, 1.13), ncol=3, frameon=False, fontsize=5.6,
                  handlelength=1.0, columnspacing=0.8)
    # (b) espacial por lambda
    ax = fig.add_subplot(gs[0, 1])
    g0 = df[(df.space == "sp") & (df.variant == "cell")]
    for lam in (1.0, 0.9, 0.5, 0.0):
        for L in (15, 30, 60):
            g = g0[(g0.lam == lam) & (g0.L == L)].sort_values("out")
            if g.empty or f"P_ext_{H}" not in g:
                continue
            ax.plot(g.out, g[f"P_ext_{H}"], LS[L], marker="o", ms=2, lw=1, color=LAMS_SEQ[lam],
                    label=(rf"$\lambda$={lam:g}" if L == 30 else None))
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(-0.001, 1.05); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\epsilon_{\rm ad}$"); ax.set_ylabel(f"P(extinct by week {H})")
    ax.set_title("(b) spatial ABM")
    ax.legend(frameon=False, loc="lower left", fontsize=5.8)
    # (c) b.m.: tres reglas de apareamiento, lambda = 0
    ax = fig.add_subplot(gs[0, 2])
    for var, col in (("cell", "#2a78d6"), ("moore", "#eb6834"), ("global", "#1baf7a")):
        for L in (15, 30, 60):
            g = df[(df.space == "wm") & (df.variant == var) & (df.lam == 0.0) & (df.L == L)].sort_values("out")
            if g.empty:
                continue
            ax.plot(g.out, g[f"P_ext_{H}"], LS[L], marker="o", ms=2, lw=1, color=col, label=(var if L == 30 else None))
    ax.axvline(0.585, color="#777", lw=0.7, ls="-.")
    ax.set_xlim(0.3, 1.02); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\epsilon_{\rm ad}$"); ax.set_ylabel(f"P(extinct by week {H})")
    ax.set_title("(c) well-mixed ABM, $\\lambda=0$")
    ax.legend(frameon=False, loc="lower left", fontsize=5.8)
    # (d)-(e) escaleo de tamaño finito, espacial
    for j, lam in enumerate((0.9, 0.0)):
        ax = fig.add_subplot(gs[1, j])
        for L, col in ((15, "#9ecae1"), (30, "#4292c6"), (60, "#08306b")):
            g = df[(df.space == "sp") & (df.variant == "cell") & (df.lam == lam) & (df.L == L)].sort_values("out")
            if g.empty:
                continue
            k = np.round(g[f"P_ext_{H}"] * g.n)
            lo, hi = zip(*[_wilson(int(a), int(b)) for a, b in zip(k, g.n)])
            ax.errorbar(g.out, g[f"P_ext_{H}"], yerr=[g[f"P_ext_{H}"] - np.array(lo), np.array(hi) - g[f"P_ext_{H}"]],
                        fmt="o-", ms=2.5, lw=1, color=col, capsize=1.5, label=f"L = {L}")
        ax.set_xlim(-0.02, 1.02 if lam > 0.5 else 0.45); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"$\epsilon_{\rm ad}$"); ax.set_ylabel(f"P(extinct by week {H})")
        ax.set_title(f"({'d' if j == 0 else 'e'}) spatial, $\\lambda$={lam:g}: finite size")
        ax.legend(frameon=False, loc="upper right", fontsize=5.8)
    # (f) T_ext frente a ln|G| en b.m. (lambda = 0) y predicción fluida
    ax = fig.add_subplot(gs[1, 2])
    cols = {0.0: "#08306b", 0.05: "#2171b5", 0.1: "#6baed6", 0.2: "#fd8d3c", 0.4: "#d94801"}
    kj = {}
    fn = os.path.join(KIN, "e1_shared.jsonl")
    for line in open(fn):
        r = json.loads(line)
        if r["point"]["social_rec_lambda"] == 0.0:
            kj[r["point"]["plast_outside_rec"]] = r
    for out, col in cols.items():
        g = df[(df.space == "wm") & (df.variant == "cell") & (df.lam == 0.0) & (df.out == out)].sort_values("L")
        if not g.empty:
            ax.errorbar(np.log(g.L ** 2), g.Text_med, yerr=[g.Text_med - g.Text_med_lo, g.Text_med_hi - g.Text_med],
                        fmt="o", ms=3, color=col, label=rf"$\epsilon_{{\rm ad}}$={out:g}")
        if out in kj and kj[out]["cls"] in ("esteril", "allee"):
            Ls = np.array([15, 30, 60, 90]); T = [kj[out][f"Tfl_L{L}"] for L in Ls]
            ax.plot(np.log(Ls ** 2), T, "-", color=col, lw=0.8)
    ax.set_xlabel(r"$\ln|G|$"); ax.set_ylabel(r"median $T_{\rm ext}$ (weeks)")
    ax.set_title("(f) well mixed, $\\lambda=0$: $T_{\\rm ext}$ vs size")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.5 * (yhi - ylo))  # espacio para la leyenda sobre las curvas
    ax.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_adult_recovery.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _wilson(k, n, z=1.96):
    from f7_aggregate import wilson
    return wilson(k, n, z)


def main():
    df = pd.read_csv(os.path.join(E1, "summary.csv"))
    tr = transition(df)
    tr.to_csv(os.path.join(E1, "transicion.csv"), index=False)
    sc = scaling(df)
    sc.to_csv(os.path.join(E1, "escaleo.csv"), index=False)
    kin = kin_table("shared")
    if kin is not None:   # la figura (otro artículo) necesita la grilla cinética e1_shared.jsonl
        fig_adult_recovery(df, kin)
    print(tr.pivot_table(index=["space", "variant", "lam"], columns="L", values="out_c").round(3).to_string())


if __name__ == "__main__":
    main()
