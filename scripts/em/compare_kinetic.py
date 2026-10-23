"""Tarea 2: ecuación cinética (exact / closure / pair=indep) vs ABM bien mezclado sin refugios (promedio de réplicas).
Guarda results_em/kinetic/traj_<model>_<variant>.csv y figs/kin_vs_abm_<model>.png"""
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kinetic import Kinetic, params_C1, params_SL
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results_em", "kinetic"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(ROOT, "results_em", "extra_figs"); os.makedirs(FIG, exist_ok=True)

VARIANTS = {"exact": dict(mode="exact", pair="shared"), "closure": dict(mode="closure", pair="shared"),
            "exact_indep": dict(mode="exact", pair="indep"), "exact_M400": dict(mode="exact", pair="shared", M=400)}


def kin_traj(model, var, T=400):
    fn = os.path.join(OUT, f"traj_{model}_{var}.csv")
    if os.path.exists(fn):
        return pd.read_csv(fn)
    p = params_C1() if model == "C1" else params_SL()
    kw = dict(VARIANTS[var]); M = kw.pop("M", 200)
    t0 = time.time()
    df = Kinetic(p, M=M, **kw).run(T, stop_nu=1e-9)
    df.to_csv(fn, index=False)
    print(model, var, f"{time.time()-t0:.0f}s", flush=True)
    return df


def abm_mean(model, L, ref=False):
    d = os.path.join(ROOT, "results_em", "wellmixed_size_scaling", f"{model}_{'ref' if ref else 'noref'}_L{L}")
    fs = sorted(glob.glob(os.path.join(d, "seed*.npz")))
    T = 400
    acc = {k: np.full((len(fs), T + 1), np.nan) for k in ("N", "births", "S", "theta", "S_rep")}
    for i, f in enumerate(fs):
        z = np.load(f)
        t = z["s_t"]; n = min(len(t), T + 1)
        acc["N"][i, :] = 0.0; acc["births"][i, :] = 0.0
        acc["N"][i, t[:n]] = z["s_N"][:n]; acc["births"][i, t[:n]] = z["s_births"][:n]
        acc["S"][i, t[:n]] = z["s_S_mean"][:n]
        acc["S_rep"][i, t[:n]] = z["s_S_repro_f"][:n]
        tt = z["th_t"]; m = tt <= T
        acc["theta"][i, tt[m]] = z["th_theta"][m]
    G = L * L
    out = pd.DataFrame({"t": np.arange(T + 1)})
    for k, v in acc.items():
        sc = G if k in ("N", "births") else 1
        out[k] = np.nanmean(v, 0) / sc
        out[k + "_sd"] = np.nanstd(v, 0) / sc
    out["n"] = len(fs)
    return out


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    summ = []
    for model in ("C1", "SL"):
        ab = abm_mean(model, L)
        ab.to_csv(os.path.join(OUT, f"abm_mean_{model}_L{L}.csv"), index=False)
        kins = {v: kin_traj(model, v) for v in VARIANTS if not (model == "SL" and v == "closure")}
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
        for v, df in kins.items():
            ls = {"exact": "-", "closure": "--", "exact_indep": ":", "exact_M400": "-."}[v]
            ax[0].plot(df.t, df.nu, ls, label=f"cinética {v}"); ax[1].plot(df.t, df.births, ls)
            ax[2].plot(df.t, df.S, ls); ax[3].plot(df.t, df.theta, ls)
        ax[0].plot(ab.t, ab.N, "k", lw=2.5, alpha=.4, label=f"ABM b.m. L={L} (n={ab.n[0]})")
        ax[1].plot(ab.t, ab.births, "k", lw=2.5, alpha=.4); ax[2].plot(ab.t, ab.S, "k", lw=2.5, alpha=.4)
        ax[3].plot(ab.t, ab.theta, "k", lw=2.5, alpha=.4)
        for a, yl in zip(ax, ("densidad N/|G|", "natalidad / |G| / sem", "S medio", "θ (vecinos con s>0)")):
            a.set_xlabel("semana"); a.set_ylabel(yl); a.set_xlim(0, 200 if model == "C1" else 160)
        if model == "C1":
            ax[3].axhline(2 / 3, color="r", lw=.8); ax[3].text(150, .69, "θ*=2/3", color="r")
        ax[0].legend(fontsize=7); fig.suptitle(f"{model}: ecuación cinética vs ABM bien mezclado sin refugios")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"kin_vs_abm_{model}.png"), dpi=120); plt.close(fig)
        for v, df in kins.items():
            m = pd.merge(df, ab, on="t", suffixes=("_k", "_a"))
            m = m[m.t <= (170 if model == "C1" else 130)]
            summ.append(dict(model=model, variant=v, peak_kin=df.nu.max(), t_peak_kin=int(df.t[df.nu.idxmax()]),
                             peak_abm=ab.N.max(), t_peak_abm=int(ab.t[ab.N.idxmax()]),
                             t_nu_1G_kin=int(df.t[df.nu < 1 / L ** 2].min()) if (df.nu < 1 / L ** 2).any() else -1,
                             max_abs_dnu=float(np.max(np.abs(m.nu - m.N))), rel_dnu_peak=float(np.max(np.abs(m.nu - m.N)) / ab.N.max()),
                             max_abs_dtheta=float(np.nanmax(np.abs(m.theta_k - m.theta_a))),
                             t_theta_below=int(df.t[df.theta < 2 / 3].min()) if (df.theta < 2 / 3).any() else -1,
                             t_births_below_1pct=int(df.t[(df.t > 10) & (df.births < 0.01 * df.births.max())].min())))
    s = pd.DataFrame(summ); print(s.round(4).to_string(index=False)); s.to_csv(os.path.join(OUT, f"resumen_kin_vs_abm_L{L}.csv"), index=False)
