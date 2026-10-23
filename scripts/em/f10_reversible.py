"""Fase 10a, puntos 1 y 2: patrones reservados (camadas fallidas, hembras que nunca crían) y oscilaciones de la rama
reversible (eps_ad = 0.8, lambda = 0.9, C1, L = 30), con comparación contra C1 de producción.

    python3 scripts/em/f10_reversible.py

Entradas (existentes):
  results_em/spatial_reversible_null/series/p0_r*.parquet      nulo reversible, T = 600, n = 200 (series depositadas)
  results_em/spatial_reversible_null/series/p1_r*.parquet      control de vida corta, T = 600
  results_em/spatial_reversible_null_T1000/summary.csv           mismas semillas a T = 1000: picos (peak_times/values)
  results_em/spatial_c1_rerun/series.parquet          C1 espacial de producción (re-corrida bit a bit f2b_base)
Entradas (re-generación bit a bit con registro de camadas; f10_runs.py):
  results_em/reversible_null_T1000/                    nulo reversible a T = 1000 (verificado contra f9_sp1000)
  results_em/c1_litters/                   C1 de producción (verificado contra f9_base)
Salidas: results_em/robustness_checks/{rev_litters.csv, rev_cohorts.csv, rev_osc_runs.csv, rev_osc_cycles.csv,
  rev_numbers.json} y figura results_em/robustness_checks/fig_rev_osc.png
Estadística: por corrida y luego mediana [p10, p90] entre corridas y media con IC 95 % bootstrap (2000 remuestreos
de corridas).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results_em")
OUT = os.path.join(R, "robustness_checks")
LIT = np.array([3, 4, 5, 6, 7, 8, 9, 10])
LITP = np.array([0.03, 0.07, 0.15, 0.25, 0.23, 0.15, 0.08, 0.04])
LBAR = float(LIT @ LITP)
Q0 = 0.95
RNG = np.random.default_rng(12345)


def boot(x, B=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    m = RNG.choice(x, size=(B, x.size)).mean(1)
    return dict(n=int(x.size), mean=float(x.mean()), ci_lo=float(np.quantile(m, .025)), ci_hi=float(np.quantile(m, .975)),
                med=float(np.median(x)), p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)))


def fmt(d, k=3):
    if not d.get("n"):
        return "NA"
    return (f"media {d['mean']:.{k}f} [IC95 {d['ci_lo']:.{k}f}, {d['ci_hi']:.{k}f}]; mediana {d['med']:.{k}f} "
            f"[p10 {d['p10']:.{k}f}, p90 {d['p90']:.{k}f}] (n={d['n']})")


def load_dir(d, point=0):
    fs = sorted(glob.glob(os.path.join(d, "series", f"p{point}_r*.parquet")))
    return {int(os.path.basename(f).split("_r")[1].split(".")[0]): pd.read_parquet(f).sort_values("t") for f in fs}


def fail_bounds(phi):
    """Fracción de camadas sin supervivientes dado phi = q medio por cría (q = 0.95 s_madre).
    Cota inferior: q homogéneo (Jensen, (1-q)^L convexa). Cota superior: q en {0, 0.95} con media phi."""
    phi = np.clip(phi, 0, Q0)
    lo = float(LITP @ (1 - phi) ** LIT)
    hi = float(1 - (phi / Q0) * (1 - LITP @ (1 - Q0) ** LIT))
    return lo, hi


def windows(S):
    t, N = S.t.to_numpy(), S.N.to_numpy(float)
    Nmax = N.max()
    i_max = int(N.argmax())
    tpk2 = t[np.flatnonzero(N >= 0.98 * Nmax)[0]]
    i_tr = i_max + int(N[i_max:i_max + 150].argmin())
    # meseta: semanas con N >= 0.9 N_max, contiguas al máximo
    hi = np.flatnonzero(N >= 0.9 * Nmax)
    a = hi[0]
    b = i_max
    while b + 1 < len(N) and N[b + 1] >= 0.9 * Nmax:
        b += 1
    return dict(t_grow=(10, t[a] - 1), t_plat=(t[a], t[b]), t_decl=(t[b] + 1, t[i_tr]), t_late=(300, t[-1]),
                tpk2=tpk2, t_trough=t[i_tr], Nmax=Nmax, t_argmax=t[i_max])


def litter_run(S, W, exact):
    out = {}
    t = S.t.to_numpy()
    for name, (a, b) in (("grow", W["t_grow"]), ("plat", W["t_plat"]), ("decl", W["t_decl"]), ("late", W["t_late"])):
        m = (t >= a) & (t <= b)
        C = S.conceptions.to_numpy(float)[m].sum()
        B = S.births.to_numpy(float)[m].sum()
        nrf = S.N_repro_f.to_numpy(float)[m].sum()
        out[f"conc_rate_{name}"] = C / nrf if nrf else np.nan        # concepciones por hembra reproductiva y semana
        out[f"phi_{name}"] = B / (LBAR * C) if C else np.nan         # supervivencia media por cría
        out[f"births_rate_{name}"] = B / nrf if nrf else np.nan
        lo, hi = fail_bounds(out[f"phi_{name}"]) if C else (np.nan, np.nan)
        out[f"fail_lo_{name}"], out[f"fail_hi_{name}"] = lo, hi
        out[f"C_{name}"] = C
        if exact and "litters_failed" in S:
            Lc = S.litters.to_numpy(float)[m].sum()
            out[f"fail_exact_{name}"] = S.litters_failed.to_numpy(float)[m].sum() / Lc if Lc else np.nan
            if "pups" in S:
                P = S.pups.to_numpy(float)[m].sum()
                out[f"phi_exact_{name}"] = B / P if P else np.nan
    return out


def cohort_run(S, W, T):
    """Fracción de hembras nacidas en una ventana que concibieron alguna vez con edad <= 48 (coh_f_*; O11b).
    Sólo cohortes con 48 semanas de seguimiento (nacidas <= T - 48)."""
    t = S.t.to_numpy()
    fb, fc = S.coh_f_born.to_numpy(float), S.coh_f_ever_conceived.to_numpy(float)
    res = {}
    wins = dict(o11b=(W["tpk2"] - 7, W["tpk2"]), plat=W["t_plat"], decl=W["t_decl"],
                late=(300, T - 48), boom=(1, W["t_plat"][0] - 1))
    for k, (a, b) in wins.items():
        m = (t >= a) & (t <= min(b, T - 48))
        nb = fb[m].sum()
        res[f"never_{k}"] = 1 - fc[m].sum() / nb if nb else np.nan
        res[f"nborn_{k}"] = nb
    return res


def osc_run(S, T):
    t, N = S.t.to_numpy(), S.N.to_numpy(float)
    Nmax = N.max()
    pk, _ = find_peaks(N, prominence=max(0.10 * Nmax, 1.0), distance=40)
    tr, _ = find_peaks(-N, prominence=max(0.10 * Nmax, 1.0), distance=40)
    cyc = []
    for j, i in enumerate(tr):
        prev = pk[pk < i]
        nxt = pk[pk > i]
        cyc.append(dict(k=j + 1, t_trough=t[i], trough=N[i] / Nmax,
                        t_peak_next=t[nxt[0]] if nxt.size else np.nan,
                        peak_next=N[nxt[0]] / Nmax if nxt.size else np.nan,
                        peak_prev=N[prev[-1]] / Nmax if prev.size else np.nan))
    r = dict(n_peaks=len(pk), n_troughs=len(tr),
             period_peaks=float(np.mean(np.diff(t[pk]))) if len(pk) > 1 else np.nan,
             period_troughs=float(np.mean(np.diff(t[tr]))) if len(tr) > 1 else np.nan,
             period_late=float(np.mean(np.diff(t[pk[t[pk] >= 200]]))) if (t[pk] >= 200).sum() > 1 else np.nan)
    # autocorrelación de N en semanas >= 300
    m = t >= 300
    x = N[m] - N[m].mean()
    if x.size > 300:
        ac = np.array([np.dot(x[:-l], x[l:]) / np.dot(x, x) for l in range(1, min(260, x.size - 50))])
        lag = np.arange(1, ac.size + 1)
        ok = lag >= 60
        r["period_acf"] = float(lag[ok][np.argmax(ac[ok])])
        r["acf_at_period"] = float(ac[ok].max())
    for a, b in ((300, 600), (600, 1000), (700, 1000), (300, 1000)):
        mm = (t >= a) & (t <= b)
        if mm.sum() > 50 and t[-1] >= b - 1:
            r[f"min_{a}_{b}"] = N[mm].min() / Nmax
            r[f"max_{a}_{b}"] = N[mm].max() / Nmax
            r[f"cv_{a}_{b}"] = N[mm].std() / N[mm].mean()
            r[f"mean_{a}_{b}"] = N[mm].mean() / Nmax
    # amortiguamiento: amplitud pico->valle siguiente por ciclo (a partir del 2.º pico), regresión log-lineal
    amps = []
    for i in pk[1:]:
        nt = tr[tr > i]
        if nt.size:
            amps.append((t[i], (N[i] - N[nt[0]]) / Nmax))
    if len(amps) >= 3:
        tt, aa = np.array(amps).T
        r["amp_slope_per_100wk"] = float(np.polyfit(tt, np.log(aa), 1)[0] * 100)   # log-amplitud por 100 sem.
        r["amp_ratio_last_first"] = float(aa[-1] / aa[0])
        r["n_amps"] = len(aa)
    return r, cyc


def summarize_branch(series, T, exact, label):
    lit, coh, osc, cycles = [], [], [], []
    for rep, S in series.items():
        if S.N.iloc[-1] == 0 and S.t.iloc[-1] < T:
            pass
        W = windows(S)
        lit.append(dict(rep=rep, **{k: (v if not isinstance(v, tuple) else str(v)) for k, v in W.items()},
                        **litter_run(S, W, exact)))
        coh.append(dict(rep=rep, **cohort_run(S, W, T)))
        o, c = osc_run(S, T)
        osc.append(dict(rep=rep, **o))
        cycles += [dict(rep=rep, **x) for x in c]
    return (pd.DataFrame(lit).assign(branch=label), pd.DataFrame(coh).assign(branch=label),
            pd.DataFrame(osc).assign(branch=label), pd.DataFrame(cycles).assign(branch=label))


def ecol_desc(series):
    """Descriptores de la revisión (ronda 2): pico y valle en t < 600, media y CV
    tardíos (t >= 300, hasta t < 600 y hasta el final)."""
    rows = []
    for rep, g in series.items():
        g6 = g[g.t < 600]
        N, b, t = g6.N.to_numpy(float), g6.births.to_numpy(float), g6.t.to_numpy()
        ipk = int(N.argmax())
        bs = pd.Series(b).rolling(8, center=True, min_periods=1).mean().to_numpy()
        imin = ipk + int(N[ipk:ipk + 150].argmin())
        late = t >= 300
        NN, tt = g.N.to_numpy(float), g.t.to_numpy()
        r = dict(births_min_rel=bs[ipk:].min() / bs[:ipk + 20].max(), trough_rel=N[imin] / N[ipk], t_trough=t[imin],
                 late_mean_rel=N[late].mean() / N[ipk], late_cv=N[late].std() / N[late].mean(),
                 late_min_rel=N[late].min() / N[ipk], late_max_rel=N[late].max() / N[ipk])
        if tt[-1] >= 999:
            L2 = tt >= 300
            r.update(cv_300_1000=NN[L2].std() / NN[L2].mean(), mean_300_1000=NN[L2].mean() / N[ipk],
                     min_700_1000=NN[tt >= 700].min() / N[ipk], max_700_1000=NN[tt >= 700].max() / N[ipk])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    nums = {}
    br = {}
    # --- existentes (necesitan las series crudas: em_f9_runs.py sp / base y f10_runs.py)
    if not os.path.exists(os.path.join(R, "spatial_c1_rerun", "series.parquet")):
        raise SystemExit("faltan las series crudas (results_em/spatial_c1_rerun/series.parquet, spatial_reversible_null/series/): "
                         "correr antes em_f9_runs.py base sp y f10_runs.py (make em-runs)")
    rev600 = load_dir(os.path.join(R, "spatial_reversible_null"), 0)
    sl600 = load_dir(os.path.join(R, "spatial_reversible_null"), 1)
    b = pd.read_parquet(os.path.join(R, "spatial_c1_rerun", "series.parquet"))
    c1 = {int(r): g.sort_values("t") for r, g in b.groupby("rep")}
    br["rev600"] = summarize_branch(rev600, 600, False, "rev600")
    br["c1"] = summarize_branch(c1, 600, False, "C1")
    # --- re-generadas (si están)
    d1000 = os.path.join(R, "reversible_null_T1000")
    if os.path.exists(os.path.join(d1000, "summary.csv")):
        rev1000 = load_dir(d1000, 0)
        # verificación bit a bit contra f9_sp (prefijo) y f9_sp1000 (resumen)
        s1000 = pd.read_csv(os.path.join(R, "spatial_reversible_null_T1000", "summary.csv"))
        s1000 = s1000[s1000.point == 0].set_index("rep")
        mine = pd.read_csv(os.path.join(d1000, "summary.csv")).set_index("rep")
        cols = ["N_max", "N_final", "total_births", "peak_times", "extinct", "min_N_after_peak"]
        okb = int((mine.loc[s1000.index, cols].astype(str) == s1000[cols].astype(str)).all(1).sum())
        okp = sum(int((rev1000[r].N.to_numpy()[:601] == rev600[r].N.to_numpy()).all()
                      and (rev1000[r].births.to_numpy()[:601] == rev600[r].births.to_numpy()).all()) for r in rev600)
        nums["check_rev1000"] = dict(summary_identical=okb, n=len(s1000), prefix600_identical=okp)
        print("rev1000 vs f9_sp1000:", okb, "/", len(s1000), "; prefijo 600 vs f9_sp:", okp)
        br["rev1000"] = summarize_branch(rev1000, 1000, True, "rev1000")
        # camadas exactas en T <= 600 (mismo prefijo) para comparar con las cotas
    for nm, dn in (("revages777", "reversible_ages_seed777"), ("revages", "reversible_ages")):
        dd = os.path.join(R, dn)
        if os.path.exists(os.path.join(dd, "summary.csv")):
            br[nm] = summarize_branch(load_dir(dd, 0), 1000, True, nm)
    for nm, dd in (("rev1000", d1000), ("revages777", os.path.join(R, "reversible_ages_seed777")),
                   ("revages", os.path.join(R, "reversible_ages")),
                   ("rev600", os.path.join(R, "spatial_reversible_null"))):
        if os.path.exists(os.path.join(dd, "series")):
            E = ecol_desc(load_dir(dd, 0))
            E.to_csv(os.path.join(OUT, f"{nm}_ecol_desc.csv"), index=False)
            nums[f"ecol_{nm}"] = {c: boot(E[c]) for c in E.columns}
            print(f"\n== descriptores de la revisión (ronda 2) {nm}")
            for c in E.columns:
                print(f"  {c}: {fmt(nums[f'ecol_{nm}'][c])}")
    dc1 = os.path.join(R, "c1_litters")
    if os.path.exists(os.path.join(dc1, "summary.csv")):
        c1x = load_dir(dc1, 0)
        okc = sum(int((c1x[r].N.to_numpy() == c1[r].N.to_numpy()).all()) for r in c1 if r in c1x)
        nums["check_c1litter"] = dict(series_identical=okc, n=len(c1))
        print("c1litter vs f9_base:", okc, "/", len(c1))
        br["c1x"] = summarize_branch(c1x, 600, True, "C1x")
    for k, (lit, coh, osc, cyc) in br.items():
        lit.to_csv(os.path.join(OUT, f"{k}_litters.csv"), index=False)
        coh.to_csv(os.path.join(OUT, f"{k}_cohorts.csv"), index=False)
        osc.to_csv(os.path.join(OUT, f"{k}_osc_runs.csv"), index=False)
        cyc.to_csv(os.path.join(OUT, f"{k}_osc_cycles.csv"), index=False)
        print(f"\n===== {k} =====")
        nb = {}
        for c in [c for c in lit.columns if c.split("_")[0] in ("conc", "phi", "births", "fail")]:
            nb[c] = boot(lit[c]); print(f"  {c}: {fmt(nb[c], 4)}")
        for c in [c for c in coh.columns if c.startswith("never")]:
            nb[c] = boot(coh[c]); print(f"  {c}: {fmt(nb[c])}")
        for c in [c for c in osc.columns if c not in ("rep", "branch")]:
            nb[c] = boot(osc[c]); print(f"  {c}: {fmt(nb[c])}")
        for c in ("t_plat", "t_decl", "t_trough", "tpk2", "t_argmax"):
            if c in lit:
                print(f"  {c}: ", lit[c].astype(str).value_counts().head(3).to_dict())
        if len(cyc):
            g = cyc.groupby("k")
            tab = g.agg(n=("trough", "size"), t_tr=("t_trough", "median"), trough_med=("trough", "median"),
                        trough_p10=("trough", lambda x: x.quantile(.1)), trough_p90=("trough", lambda x: x.quantile(.9)),
                        peak_med=("peak_next", "median"), peak_p10=("peak_next", lambda x: x.quantile(.1)),
                        peak_p90=("peak_next", lambda x: x.quantile(.9)), t_pk=("t_peak_next", "median"))
            print(tab.round(3).to_string())
            nb["cycles"] = tab.reset_index().round(4).to_dict("records")
        nums[k] = nb
    # picos del ensamble existente a T = 1000 (f9_sp1000)
    s = pd.read_csv(os.path.join(R, "spatial_reversible_null_T1000", "summary.csv"))
    for p, lab in ((0, "rev"), (1, "SL")):
        g = s[s.point == p]
        rows = []
        for _, r in g.iterrows():
            if not isinstance(r.peak_times, str):
                continue
            tt = np.array([float(v) for v in r.peak_times.split(";")])
            vv = np.array([float(v) for v in r.peak_values.split(";")]) / r.N_max
            if tt.size < 2:
                continue
            d = np.diff(tt)
            rows.append(dict(rep=r.rep, extinct=r.extinct, n_peaks=tt.size, period=d.mean(), period_first=d[0],
                             period_rest=d[1:].mean() if d.size > 1 else np.nan,
                             peak2=vv[1] if vv.size > 1 else np.nan, peak_last=vv[-1],
                             peak_slope_per_100wk=(np.polyfit(tt[1:], vv[1:], 1)[0] * 100) if tt.size > 3 else np.nan,
                             peaks_after2=";".join(f"{v:.3f}" for v in vv[1:])))
        P = pd.DataFrame(rows)
        P.to_csv(os.path.join(OUT, f"sp1000_peaks_{lab}.csv"), index=False)
        nums[f"sp1000_{lab}"] = {c: boot(P[c]) for c in ("n_peaks", "period", "period_first", "period_rest", "peak2",
                                                          "peak_last", "peak_slope_per_100wk")}
        nums[f"sp1000_{lab}"]["n_runs_with_2peaks"] = len(P)
        nums[f"sp1000_{lab}"]["n_extinct"] = int(g.extinct.sum())
        print(f"\n== f9_sp1000 {lab}: corridas con >=2 picos {len(P)}/{len(g)}, extintas {int(g.extinct.sum())}")
        for c in ("n_peaks", "period", "period_first", "period_rest", "peak2", "peak_last", "peak_slope_per_100wk"):
            print(f"  {c}: {fmt(nums[f'sp1000_{lab}'][c])}")
    json.dump(nums, open(os.path.join(OUT, "rev_numbers.json"), "w"), indent=1, default=float)
    fig(br)


def fig(br):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    key = "rev1000" if "rev1000" in br else "rev600"
    d = os.path.join(R, "reversible_null_T1000" if key == "rev1000" else "spatial_reversible_null")
    ser = load_dir(d, 0)
    M = pd.concat([(S.set_index("t").N / S.N.max()).rename(r) for r, S in ser.items()], axis=1)
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    for r in list(M.columns)[:6]:
        ax.plot(M.index, M[r], lw=0.4, color="#9ecae1")
    ax.fill_between(M.index, M.quantile(.1, axis=1), M.quantile(.9, axis=1), color="#4292c6", alpha=.25, lw=0)
    ax.plot(M.index, M.median(axis=1), color="#08306b", lw=1.2, label="median (200 runs)")
    ax.set_xlabel("week"); ax.set_ylabel(r"$N/N_{\max}$"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(r"Reversible null ($\epsilon_{\rm ad}=0.8$, $\lambda=0.9$, $L=30$): band p10–p90", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig_rev_osc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
