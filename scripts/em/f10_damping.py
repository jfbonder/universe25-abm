"""Amortiguamiento de las oscilaciones y resumen de los chequeos de la revisión (ronda 2).

    python3 scripts/em/f10_damping.py

Entradas: results_em/robustness_checks/{rev1000,revages,revages777}_osc_cycles.csv (f10_reversible.py),
series de results_em/reversible_null_T1000/ y summaries de f10_refr, f10_refr777, f10_c1litter.
Salida: results_em/robustness_checks/damping.json
Amortiguamiento: (a) amplitud pico->valle siguiente por ciclo k, media con IC95 bootstrap sobre corridas;
(b) ajuste A_k = A_inf + B r^(k-1) a las medias (k = 1..5), IC por bootstrap de corridas;
(c) envolvente de la autocorrelación de N (t >= 300) en los retardos m*T (m = 1, 2, 3, 4) — si el ciclo es un
foco amortiguado excitado por ruido (cuasi-ciclo) la envolvente decae como exp(-m T / tau);
(d) pendiente de la amplitud en los ciclos 3..5 (t >= 450).
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results_em"); OUT = os.path.join(R, "robustness_checks")
rng = np.random.default_rng(2026)


def ci(x, B=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    m = rng.choice(x, (B, x.size)).mean(1)
    return [round(float(x.mean()), 4), round(float(np.quantile(m, .025)), 4), round(float(np.quantile(m, .975)), 4), int(x.size)]


def amp_table(cyc):
    # amplitud del ciclo k: pico_prev(k+1) - valle(k+1) == pico k - valle siguiente; usamos peak_next(k) - trough(k+1)
    rows = []
    for rep, g in cyc.groupby("rep"):
        g = g.sort_values("k").reset_index(drop=True)
        for j in range(len(g) - 1):
            rows.append(dict(rep=rep, k=j + 1, t_peak=g.t_peak_next[j], peak=g.peak_next[j], trough_next=g.trough[j + 1],
                             amp=g.peak_next[j] - g.trough[j + 1]))
    return pd.DataFrame(rows)


def fit_geo(A):
    k = np.arange(1, A.shape[1] + 1)
    f = lambda k, a, b, r: a + b * r ** (k - 1)
    def one(M):
        y = np.nanmean(M, 0)
        p, _ = curve_fit(f, k, y, p0=(0.35, 0.15, 0.5), bounds=([0, 0, 0], [1, 1, 1.5]), maxfev=20000)
        return p
    p0 = one(A)
    bs = np.array([one(A[rng.integers(0, A.shape[0], A.shape[0])]) for _ in range(1000)])
    return {n: [round(float(p0[i]), 4), round(float(np.quantile(bs[:, i], .025)), 4), round(float(np.quantile(bs[:, i], .975)), 4)]
            for i, n in enumerate(("A_inf", "B", "r"))}


def acf_env(d, Tp=147):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "series", "p0_r*.parquet"))):
        S = pd.read_parquet(f, columns=["t", "N"]).sort_values("t")
        x = S.N.to_numpy(float)[S.t.to_numpy() >= 300]; x = x - x.mean()
        ac = lambda l: np.dot(x[:-l], x[l:]) / np.dot(x, x) * x.size / (x.size - l)   # estimador insesgado en n
        row = {}
        for m in (1, 2, 3, 4):
            lo, hi = int(m * Tp - 25), int(m * Tp + 25)
            row[f"acf_{m}T"] = max(ac(l) for l in range(lo, min(hi, x.size - 30)))
        out.append(row)
    return pd.DataFrame(out)


def main():
    if not os.path.isdir(os.path.join(R, "reversible_null_T1000", "series")):
        raise SystemExit("faltan las series crudas de results_em/reversible_null_T1000/ (correr antes f10_runs.py rev1000; make em-runs)")
    res = {}
    for nm, d in (("rev1000", "reversible_null_T1000"), ("revages", "reversible_ages"),
                  ("revages777", "reversible_ages_seed777")):
        cyc = pd.read_csv(os.path.join(OUT, f"{nm}_osc_cycles.csv"))
        A = amp_table(cyc)
        tab = {int(k): dict(t_peak_med=float(g.t_peak.median()), peak=ci(g.peak), trough_next=ci(g.trough_next), amp=ci(g.amp))
               for k, g in A.groupby("k")}
        W = A.pivot(index="rep", columns="k", values="amp")
        W = W[[k for k in W.columns if W[k].notna().mean() > 0.95]]
        r = dict(cycles=tab, geo_fit=fit_geo(W.to_numpy()))
        # pendiente de la amplitud en ciclos 3..5 (por corrida), en fracción del pico por 100 semanas
        sl = []
        for rep, g in A[A.k >= 3].groupby("rep"):
            if len(g) >= 3:
                sl.append(np.polyfit(g.t_peak, g.amp, 1)[0] * 100)
        r["amp_slope_k3plus_per100wk"] = ci(sl)
        sl2 = [np.polyfit(g.t_peak, g.amp, 1)[0] * 100 for rep, g in A[A.k >= 2].groupby("rep") if len(g) >= 3]
        r["amp_slope_k2plus_per100wk"] = ci(sl2)
        E = acf_env(os.path.join(R, d))
        r["acf_envelope"] = {c: ci(E[c]) for c in E.columns}
        res[nm] = r
        print(nm, json.dumps(r, indent=0)[:3000])
    # chequeos de la revisión (ronda 2)
    c1 = pd.read_csv(os.path.join(R, "c1_litters", "summary.csv")).set_index("rep").sort_index()
    for nm, dn in (("refr", "refractory_check"), ("refr777", "refractory_check_seed777")):
        s = pd.read_csv(os.path.join(R, dn, "summary.csv")).set_index("rep").sort_index()
        d = dict(n=len(s), extinct=int(s.extinct.sum()), O17p_pass=int((s.obj_O17p == "PASS").sum()),
                 O17_run_pass=int((s.obj_O17_run == "PASS").sum()), t_ext=ci(s.t_ext), N_max=ci(s.N_max),
                 t_peak=ci(s.t_peak))
        if nm == "refr":
            cc = c1.loc[s.index]
            d["paired_C1_t_ext"] = ci(cc.t_ext); d["paired_C1_N_max"] = ci(cc.N_max)
            d["diff_t_ext"] = ci(s.t_ext - cc.t_ext); d["diff_N_max"] = ci(s.N_max - cc.N_max)
            d["C1_first60_O17p"] = int((cc.obj_O17p == "PASS").sum())
        res[nm] = d
        print(nm, d)
    json.dump(res, open(os.path.join(OUT, "damping.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
