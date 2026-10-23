"""Fase 10a, punto 3: intervalos de verosimilitud perfilada de eps^c(L) con el modelo logístico del ajuste
(f7_analyze_e1.logistic_fit: P_ext(1000) = 1/(1+exp((eps - x0)/w)), binomial), y del ancho w.

    python3 scripts/em/f10_epsc_profile.py

Entrada: results_em/adult_recovery/summary.csv (n = 20, 20, 10 colonias por punto a L = 15, 30, 60).
Salida:  results_em/robustness_checks/epsc_profile.csv, epsc_width_lrt.csv
Intervalos: {x0 : 2[l(x0_hat) - max_w l(x0, w)] <= chi2_1(0.95) = 3.84}; w perfilado en [1e-5, 3] (w -> 0 es la
función escalón, permitida). Para w se informa el intervalo análogo; si su extremo inferior es el borde 1e-5, el
ancho no está acotado por abajo (no identificado).
LRT del "sharpening": H0 un mismo w para los tres L (x0 libre por L) frente a H1 un w por L (2 g.l.).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E1 = os.path.join(ROOT, "results_em", "adult_recovery")
OUT = os.path.join(ROOT, "results_em", "robustness_checks")
LW_LO, LW_HI = np.log(1e-5), np.log(3.0)
CRIT = chi2.ppf(0.95, 1) / 2


def nll(x0, lw, x, k, n):
    z = np.clip((x - x0) / np.exp(lw), -60, 60)
    p = np.clip(1 / (1 + np.exp(z)), 1e-12, 1 - 1e-12)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def prof_x0(x0, x, k, n):
    g = np.linspace(LW_LO, LW_HI, 61)
    v = [nll(x0, a, x, k, n) for a in g]
    i = int(np.argmin(v))
    r = minimize_scalar(lambda a: nll(x0, a, x, k, n), bounds=(g[max(i - 1, 0)], g[min(i + 1, 60)]), method="bounded")
    return min(r.fun, v[i]), (r.x if r.fun <= v[i] else g[i])


def prof_lw(lw, x, k, n):
    g = np.linspace(x.min(), x.max(), 201)
    v = [nll(a, lw, x, k, n) for a in g]
    i = int(np.argmin(v))
    r = minimize_scalar(lambda a: nll(a, lw, x, k, n), bounds=(g[max(i - 1, 0)], g[min(i + 1, 200)]), method="bounded")
    return min(r.fun, v[i])


def fit(x, k, n):
    gx = np.linspace(x.min() - 0.1, x.max() + 0.3, 5601)
    pv = [prof_x0(a, x, k, n) for a in gx]
    f = np.array([a[0] for a in pv])
    i = int(np.argmin(f))
    r = minimize_scalar(lambda a: prof_x0(a, x, k, n)[0], bounds=(gx[max(i - 2, 0)], gx[min(i + 2, len(gx) - 1)]),
                        method="bounded")
    best = min(r.fun, f[i])
    x0 = r.x if r.fun <= f[i] else gx[i]
    lw = prof_x0(x0, x, k, n)[1]
    ok = gx[f - best <= CRIT]
    lo, hi = ok.min(), ok.max()
    glw = np.linspace(LW_LO, LW_HI, 401)
    fw = np.array([prof_lw(a, x, k, n) for a in glw])
    okw = glw[fw - best <= CRIT]
    return dict(x0=x0, w=np.exp(lw), nll=best, x0_lo=lo, x0_hi=hi, w_lo=np.exp(okw.min()), w_hi=np.exp(okw.max()),
                w_lo_at_border=bool(okw.min() <= LW_LO + 1e-9), x0_lo_beyond_grid=bool(lo < x.min()),
                x0_hi_beyond_grid=bool(hi > x.max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(E1, "summary.csv"))
    rows, lrt, pair = [], [], []
    sel = [("sp", "cell", 0.0), ("sp", "cell", 0.9), ("sp", "cell", 1.0), ("wm", "cell", 0.0), ("wm", "global", 0.0)]
    for sp, var, lam in sel:
        data = {}
        for L in (15, 30, 60):
            g = df[(df.space == sp) & (df.variant == var) & (df.lam == lam) & (df.L == L)].sort_values("out")
            if g.empty:
                continue
            x = g.out.to_numpy(float)
            n = g.n.to_numpy(float)
            k = np.round(g.P_ext_1000.to_numpy(float) * n)
            data[L] = (x, k, n)
            r = fit(x, k, n)
            p = k / n
            mid = (p > 0) & (p < 1)
            r.update(space=sp, variant=var, lam=lam, L=L, n_per_point=int(n[0]), n_points=len(x),
                     n_intermediate=int(mid.sum()),
                     intermediate=";".join(f"{a:g}:{int(b)}/{int(c)}" for a, b, c in zip(x[mid], k[mid], n[mid])),
                     last_all_extinct=float(x[p >= 1].max()) if (p >= 1).any() else np.nan,
                     first_none_extinct=float(x[p <= 0].min()) if (p <= 0).any() else np.nan)
            rows.append(r)
            print(f"{sp} {var} lam={lam} L={L}: x0={r['x0']:.4f} [{r['x0_lo']:.4f}, {r['x0_hi']:.4f}]  "
                  f"w={r['w']:.4f} [{r['w_lo']:.2g}, {r['w_hi']:.3f}] border={r['w_lo_at_border']} "
                  f"mid={r['intermediate']}")
        # LRT ancho común
        if len(data) == 3:
            l1 = sum(r["nll"] for r in rows[-3:])

            def nll0(th):
                lw = th[0]
                return sum(prof_lw_fixed(lw, *data[L]) for L in data)

            def prof_lw_fixed(lw, x, k, n):
                return prof_lw(lw, x, k, n)
            glw = np.linspace(LW_LO, LW_HI, 241)
            v = [nll0([a]) for a in glw]
            i = int(np.argmin(v))
            l0 = v[i]
            stat = 2 * (l0 - l1)
            lrt.append(dict(space=sp, variant=var, lam=lam, w_common=np.exp(glw[i]), nll_common=l0, nll_sep=l1,
                            LR=stat, df=2, p=float(chi2.sf(max(stat, 0), 2))))
            print(f"   LRT ancho común: w={np.exp(glw[i]):.4f} LR={stat:.2f} p={chi2.sf(max(stat, 0), 2):.3f}")
        for a, b in ((15, 30), (30, 60), (15, 60)):
            if a in data and b in data:
                ra = [r for r in rows if (r["space"], r["variant"], r["lam"], r["L"]) == (sp, var, lam, a)][0]
                rb = [r for r in rows if (r["space"], r["variant"], r["lam"], r["L"]) == (sp, var, lam, b)][0]
                lo_ = min(data[a][0].min(), data[b][0].min()) - 0.1
                hi_ = max(data[a][0].max(), data[b][0].max()) + 0.3
                gg = np.linspace(lo_, hi_, 2801)
                v = [prof_x0(c, *data[a])[0] + prof_x0(c, *data[b])[0] for c in gg]
                j = int(np.argmin(v))
                st = 2 * (v[j] - ra["nll"] - rb["nll"])
                pair.append(dict(space=sp, variant=var, lam=lam, L1=a, L2=b, x0_common=gg[j], LR=st,
                                 p=float(chi2.sf(max(st, 0), 1))))
                print(f"   LRT x0({a}) = x0({b}): LR={st:.2f} p={chi2.sf(max(st, 0), 1):.2g}")
    pd.DataFrame(pair).to_csv(os.path.join(OUT, "epsc_pair_lrt.csv"), index=False)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "epsc_profile.csv"), index=False)
    pd.DataFrame(lrt).to_csv(os.path.join(OUT, "epsc_width_lrt.csv"), index=False)
    # solapamiento
    for (sp, var, lam), g in T.groupby(["space", "variant", "lam"]):
        g = g.set_index("L")
        s = []
        for a, b in ((15, 30), (30, 60), (15, 60)):
            if a in g.index and b in g.index:
                ov = not (g.loc[a, "x0_lo"] > g.loc[b, "x0_hi"] or g.loc[b, "x0_lo"] > g.loc[a, "x0_hi"])
                s.append(f"{a}-{b}: {'SE SOLAPAN' if ov else 'disjuntos'}")
        print(sp, var, lam, "; ".join(s))


if __name__ == "__main__":
    main()
