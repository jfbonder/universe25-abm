"""Informes listos: línea base (supervivencia + picos), Morris, Sobol."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import io, survival, peaks, metrics, sensitivity, multiple, plots, design


def _md(df: pd.DataFrame, index=True, floatfmt=".3g"):
    """Tabla markdown sin depender de `tabulate`."""
    d = df.reset_index() if index else df
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "NaN" if not np.isfinite(v) else format(v, floatfmt)
        return str(v)
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


pd.DataFrame.to_markdown = _md


def _fmt(x, nd=3):
    return "NaN" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}g}" if isinstance(x, (float, np.floating)) else str(x))


def baseline_report(run_dir, out_dir, t_split=None, B_gof=300, min_peaks=4, prominence_rel=0.10):
    run = io.load_run(run_dir, series=True)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    s = run["summary"]; lines = [f"# Informe de línea base: `{run_dir}`", ""]
    lines += [f"n = {len(s)} corridas, T = {int(s['T'].max())}, puntos = {s['point'].nunique()}", ""]
    # --- supervivencia
    time, ev = io.event_table(s)
    sv = survival.survival_summary(time, ev, t_split=t_split, B_gof=B_gof)
    sv["km"].to_csv(out / "km.csv", index=False)
    lines += ["## Supervivencia (tiempo de extinción, censura a la derecha)", ""]
    lines += [f"- eventos {sv['events']}, censuradas {sv['censored']}; mediana KM = {_fmt(sv['median']['median'])} "
              f"[{_fmt(sv['median']['lo'])}, {_fmt(sv['median']['hi'])}]"]
    lines += ["- P_ext(T): " + ", ".join(f"T={T}: {p:.3f}" for T, p in sv["P_ext"].items())]
    if "mixture" in sv:
        m = sv["mixture"]
        lines += [f"- mezcla temprana/tardía (corte t = {sv['t_split']:.0f}): p_early = {m['p_early']:.3f} "
                  f"[{m['lo']:.3f}, {m['hi']:.3f}]; cola: {m['tail_events']} eventos, hazard = {_fmt(m['tail_rate'])} /sem "
                  f"[{_fmt(m['tail_rate_lo'])}, {_fmt(m['tail_rate_hi'])}]"]
    if sv["fits"]:
        lines += ["", "| ajuste (cola, t > t0) | parámetros | logL | AIC | GOF p (bootstrap) |", "|---|---|---|---|---|"]
        for d, f in sv["fits"].items():
            g = sv.get("gof", {}).get(d, {})
            lines += [f"| {d} | {f['params']} | {f['loglik']:.2f} | {f['aic']:.2f} | {_fmt(g.get('p'))} (D={_fmt(g.get('D'))}) |"]
        lr = sv["lr_weibull_vs_exp"]
        lines += ["", f"- LR Weibull vs exponencial: LR = {lr['LR']:.2f}, p = {lr['p']:.3g} "
                  f"(forma Weibull k = {sv['fits']['weibull']['params']['shape']:.2f}; k<1 ⇒ hazard decreciente / mezcla)"]
    pw, lrpw = survival.piecewise_hazard(time, ev, cuts=[sv["t_split"], 2 * sv["t_split"]] if np.isfinite(sv["t_split"]) else [200, 500])
    pw.to_csv(out / "hazard_tramos.csv", index=False)
    lines += ["", "Hazard a trozos:", "", pw.to_markdown(index=False, floatfmt=".4g"), "",
              f"LR hazard constante vs a trozos: LR = {lrpw['LR']:.2f}, df = {lrpw['df']}, p = {lrpw['p']:.3g}"]
    plots.plot_km(sv, out / "fig_km.png")
    # --- picos / drift
    ser = run["series"]
    tests = []
    if ser is not None:
        cyc = peaks.cycles_per_run(ser, prominence_rel=prominence_rel)
        cyc.to_csv(out / "ciclos.csv", index=False)
        dr = peaks.ensemble_drift_test(cyc, min_peaks=min_peaks)
        wr = peaks.within_regression(ser, min_peaks=min_peaks, prominence_rel=prominence_rel)
        lines += ["", "## Picos sucesivos y drift", ""]
        lines += [f"- corridas con ≥{min_peaks} picos: {dr.get('n_runs', 0)} de {len(cyc)}; período medio = "
                  f"{_fmt(cyc['period'].mean())} ± {_fmt(cyc['period'].std())} sem; primer pico {_fmt(cyc['first_peak'].mean(), 4)} "
                  f"vs picos posteriores {_fmt(cyc['mean_rest'].mean(), 4)}"]
        if dr.get("n_runs", 0) >= 3:
            lines += [f"- pendiente media de log(pico) por ciclo = {dr['mean_slope']:+.4f} [IC95 bootstrap {dr['ci_lo']:+.4f}, {dr['ci_hi']:+.4f}], "
                      f"sd entre corridas {dr['sd_slope']:.4f}; Wilcoxon vs 0: p = {_fmt(dr['wilcoxon_p'])}; "
                      f"signos +/−: {dr['sign_pos']}/{dr['sign_neg']}; MK significativo negativo en {dr['frac_mk_sig_neg']:.0%} "
                      f"de las corridas, positivo en {dr['frac_mk_sig_pos']:.0%}; mediana de Sen = {dr['median_sen']:+.4f}"]
            lines += [f"- regresión within (efectos fijos por corrida): pendiente = {wr['slope']:+.4f} ± {wr['se']:.4f}, p = {_fmt(wr['p'])}"]
            tests += [dict(test="drift_wilcoxon", p=dr["wilcoxon_p"]), dict(test="drift_within", p=wr["p"])]
        plots.plot_peaks(ser, cyc, out / "fig_picos.png", prominence_rel=prominence_rel)
        plots.plot_series_sample(ser, out / "fig_series.png")
    # --- regímenes
    pp = metrics.per_point(s)
    pp.to_csv(out / "por_punto.csv", index=False)
    lines += ["", "## Regímenes por corrida", "", pp[[c for c in pp.columns if c.startswith("frac_") or c in ("point", "n")]].to_markdown(index=False, floatfmt=".2f")]
    # --- comparaciones múltiples sobre los tests del informe
    if sv["fits"]:
        tests += [dict(test="weibull_vs_exp", p=sv["lr_weibull_vs_exp"]["p"]), dict(test="hazard_tramos", p=lrpw["p"])]
    if tests:
        tt = multiple.adjust_table(pd.DataFrame(tests))
        tt.to_csv(out / "tests.csv", index=False)
        lines += ["", "## Tests del informe con corrección múltiple", "", tt.to_markdown(index=False, floatfmt=".3g")]
    (out / "informe.md").write_text("\n".join(lines))
    return sv, lines


def morris_report(design_dir, run_dir, out_dir, metrics_list=None, num_levels=4):
    info, X, dd = design.load_design(design_dir)
    run = io.load_run(run_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pp = metrics.per_point(run["summary"]); pp.to_csv(out / "por_punto.csv", index=False)
    if metrics_list is None:
        metrics_list = [c for c in ["N_max_mean", "t_peak_mean", "S_at_peak_mean", "min_N_after_peak_mean", "n_regrowths_mean",
                                    "frac_extinct", "t_ext_median_km"] + [c for c in pp.columns if c.startswith("P_ext_") and not c.endswith(("_lo", "_hi"))]
                        if c in pp.columns]
    mo = sensitivity.morris_analyze(info["problem"], X, pp, metrics_list, num_levels=info.get("num_levels", num_levels))
    mo.to_csv(out / "morris.csv", index=False)
    lines = [f"# Morris: `{design_dir}` ({info['n_points']} puntos, {run['summary']['rep'].nunique()} réplicas)", ""]
    for m in mo["metric"].unique():
        d = mo[mo["metric"] == m].sort_values("mu_star", ascending=False)
        lines += [f"## {m}", "", d[["name", "mu_star", "mu_star_conf", "sigma", "mu", "rank"]].to_markdown(index=False, floatfmt=".3g"), ""]
        plots.plot_morris(mo, out / f"fig_morris_{m}.png", m)
    # ranking agregado
    agg = mo.groupby("name")["mu_star_rel"].mean().sort_values(ascending=False)
    lines += ["## Ranking agregado (μ* relativo medio sobre métricas)", "", agg.to_frame("mu_star_rel_medio").to_markdown(floatfmt=".3f")]
    (out / "informe_morris.md").write_text("\n".join(lines))
    return mo


def sobol_report(design_dir, run_dir, out_dir, metrics_list=None):
    info, X, dd = design.load_design(design_dir)
    run = io.load_run(run_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pp = metrics.per_point(run["summary"]); pp.to_csv(out / "por_punto.csv", index=False)
    if metrics_list is None:
        metrics_list = [c for c in ["N_max_mean", "S_at_peak_mean", "min_N_after_peak_mean", "frac_extinct"] if c in pp.columns]
    so, S2 = sensitivity.sobol_analyze(info["problem"], pp, metrics_list, calc_second_order=info.get("calc_second_order", False))
    so.to_csv(out / "sobol.csv", index=False)
    lines = [f"# Sobol: `{design_dir}` (N_base = {info.get('N')}, {info['n_points']} puntos)", ""]
    for m in so["metric"].unique():
        d = so[so["metric"] == m].sort_values("ST", ascending=False)
        lines += [f"## {m}", "", d[["name", "S1", "S1_conf", "ST", "ST_conf", "interaction"]].to_markdown(index=False, floatfmt=".3f")]
        base_metric = m.replace("_mean", "")
        if base_metric in run["summary"]:
            sh = metrics.stochastic_share(run["summary"], base_metric)
            lines += ["", f"fracción de varianza estocástica (intra-punto/total) de {base_metric}: {sh['share_stochastic']:.3f}"]
        lines += [""]
        plots.plot_sobol(so, out / f"fig_sobol_{m}.png", m)
    (out / "informe_sobol.md").write_text("\n".join(lines))
    return so
