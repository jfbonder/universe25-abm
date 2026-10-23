"""Fase 10a, punto 4 (e9): distribución de la ocupación por celda en el pico, C1 espacial (L = 30).

    python3 scripts/em/f10_occupancy.py

Entradas (existentes):
  results_em/wellmixed_c1/snapshots_seed1.npz  ocupación de las 900 celdas cada semana, una corrida de C1
                                                         espacial (preset calhoun_C1, SeedSequence([20261017, 1]))
  results_em/spatial_c1_rerun/series.parquet (o series_occupancy.parquet, reducida)  200 corridas de C1 de producción: N, occ_var, occ_max,
                                                         mstar (= media sobre individuos de m - 1), f_over (fracción
                                                         de individuos con m > m_c = 8)
Salidas: results_em/robustness_checks/occupancy_peak.json, occupancy_ensemble.csv,
         results_em/robustness_checks/fig_occupancy_peak.{png,pdf}
"Pico": t_pk = argmax N y t'_pk = primera semana con N >= 0.98 N_max (evaluador primario, plateau_start).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results_em")
OUT = os.path.join(R, "robustness_checks")
MC = 8
RNG = np.random.default_rng(7)


def dist(occ, ref):
    occ = occ.astype(int)
    w = np.repeat(occ, occ)                     # ocupación vista por cada individuo
    q = lambda x, p: float(np.quantile(x, p))   # noqa: E731
    return dict(N=int(occ.sum()), mean_cell=float(occ.mean()), var_cell=float(occ.var()),
                cell_median=q(occ, .5), cell_p90=q(occ, .9), cell_p99=q(occ, .99), cell_max=int(occ.max()),
                frac_cells_empty=float((occ == 0).mean()), frac_cells_gt_mc=float((occ > MC).mean()),
                ind_mean=float(w.mean()), ind_median=q(w, .5), ind_p10=q(w, .1), ind_p90=q(w, .9), ind_p99=q(w, .99),
                frac_ind_gt_mc=float((w > MC).mean()), frac_ind_ge15=float((w >= 15).mean()),
                refuge_cells_mean=float(occ[ref].mean()), refuge_cells_max=int(occ[ref].max()),
                nonref_cells_mean=float(occ[~ref].mean()), nonref_cell_median=q(occ[~ref], .5),
                nonref_cell_p99=q(occ[~ref], .99))


def boot(x, B=2000):
    x = np.asarray(x, float)
    m = RNG.choice(x, size=(B, x.size)).mean(1)
    return dict(mean=float(x.mean()), ci_lo=float(np.quantile(m, .025)), ci_hi=float(np.quantile(m, .975)),
                med=float(np.median(x)), p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)))


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(R, "wellmixed_c1", "snapshots_seed1.npz"))
    occ, N, ref = z["sp_occ"], z["sp_N"], z["sp_ref"]
    i_pk = int(np.argmax(N))
    i_p2 = int(np.flatnonzero(N >= 0.98 * N.max())[0])
    res = {"single_run": {"seed": "SeedSequence([20261017, 1])", "week_argmax": i_pk + 1, "week_plateau_start": i_p2 + 1,
                          "N_max": int(N.max()), "at_argmax": dist(occ[i_pk], ref), "at_plateau_start": dist(occ[i_p2], ref)}}
    # semanas del pico +-5 juntas (más estable)
    W = occ[max(i_pk - 5, 0):i_pk + 6]
    res["single_run"]["pooled_argmax_pm5"] = dist(W.reshape(-1), np.tile(ref, W.shape[0]))
    # ensamble: momentos en t_pk
    full = os.path.join(R, "spatial_c1_rerun", "series.parquet")     # serie completa (no versionada)
    red = os.path.join(R, "spatial_c1_rerun", "series_occupancy.parquet")   # columnas usadas, versionada
    b = pd.read_parquet(full if os.path.exists(full) else red)
    rows = []
    for rep, g in b.groupby("rep"):
        g = g.sort_values("t")
        Nn = g.N.to_numpy()
        for lab, i in (("argmax", int(Nn.argmax())), ("plateau", int(np.flatnonzero(Nn >= 0.98 * Nn.max())[0]))):
            r = g.iloc[i]
            rows.append(dict(rep=rep, at=lab, t=int(r.t), N=int(r.N), mean_cell=r.N / 900, var_cell=r.occ_var,
                             occ_max=int(r.occ_max), ind_mean=r.mstar + 1, f_over=r.f_over,
                             ID=r.occ_var / (r.N / 900)))
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "occupancy_ensemble.csv"), index=False)
    res["ensemble_f9_base"] = {lab: {c: boot(g[c]) for c in ("t", "N", "mean_cell", "var_cell", "occ_max", "ind_mean",
                                                                 "f_over", "ID")}
                               for lab, g in E.groupby("at")}
    json.dump(res, open(os.path.join(OUT, "occupancy_peak.json"), "w"), indent=1)
    print(json.dumps(res["single_run"]["at_argmax"], indent=0))
    print(json.dumps(res["single_run"]["pooled_argmax_pm5"], indent=0))
    for lab, d in res["ensemble_f9_base"].items():
        print(lab, {k: (round(v["med"], 3), round(v["p10"], 3), round(v["p90"], 3)) for k, v in d.items()})
    fig(occ[i_pk], ref, E[E["at"] == "argmax"], N.max(), i_pk + 1)


def fig(o, ref, E, Nmax, tpk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    bins = np.arange(-0.5, o.max() + 1.5)
    ax[0].hist([o[~ref], o[ref]], bins=bins, stacked=True, color=["#4292c6", "#fd8d3c"],
               label=["open cells", "retreat sites (capacity 2)"])
    ax[0].axvline(MC + 0.5, color="#555", ls="--", lw=0.7)
    ax[0].text(MC + 0.8, ax[0].get_ylim()[1] * 0.9, r"$m_c=8$", fontsize=7)
    ax[0].set_xlabel("animals per cell, $m$"); ax[0].set_ylabel("cells")
    ax[0].set_title(f"(a) cells at the peak (one run, week {tpk}, N = {Nmax})", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5)
    w = np.repeat(o, o)
    ax[1].hist(w, bins=bins, color="#08306b", density=True)
    ax[1].axvline(MC + 0.5, color="#555", ls="--", lw=0.7)
    ax[1].set_xlabel("occupancy of the animal's own cell, $m$"); ax[1].set_ylabel("fraction of animals")
    ax[1].set_title("(b) as experienced by the animals", fontsize=8)
    txt = (f"200 runs at the peak, median [p10, p90]:\nmean per cell {E.mean_cell.median():.2f}\n"
           f"experienced mean {E.ind_mean.median():.1f} [{E.ind_mean.quantile(.1):.1f}, {E.ind_mean.quantile(.9):.1f}]\n"
           f"animals with m > 8: {E.f_over.median():.2f} [{E.f_over.quantile(.1):.2f}, {E.f_over.quantile(.9):.2f}]\n"
           f"max per cell {E.occ_max.median():.0f} [{E.occ_max.quantile(.1):.0f}, {E.occ_max.quantile(.9):.0f}]")
    ax[1].text(0.98, 0.97, txt, transform=ax[1].transAxes, ha="right", va="top", fontsize=6.3, bbox=dict(fc="white", ec="none", alpha=0.9))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_occupancy_peak.{ext}"), dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
