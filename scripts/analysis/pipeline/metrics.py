"""Métricas por punto de diseño (agregación de réplicas) a partir de summary.csv (+ series opcional)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .survival import kaplan_meier, km_median, km_at, wilson
from .io import event_table

CONT = ["N_max", "t_peak", "S_min", "S_at_peak", "collapse_duration", "min_N_after_peak", "n_regrowths",
        "total_births", "max_occ", "t_S_below_0.2"]


def regime(row, rel=0.3):
    """Clasificación de régimen por corrida."""
    if row.get("exploded", False):
        return "explota"
    if row["N_max"] < 2 * row.get("N0", 400):
        return "sin_despegue"
    if row["extinct"] and row.get("n_regrowths", 0) == 0:
        return "colapso_unico"
    if row.get("n_regrowths", 0) >= 1:
        return "oscilatorio" if row.get("min_N_after_peak", 0) < rel * row["N_max"] else "meseta"
    return "colapso_unico" if row["extinct"] else "meseta"


def per_point(summary: pd.DataFrame, probs_at=(300, 500, 1000, 2000), cont=CONT):
    """Una fila por punto: medias/sd de métricas continuas, P_ext(T) con IC Wilson, mediana KM, regímenes."""
    rows = []
    for pt, g in summary.groupby("point", sort=True):
        row = dict(point=pt, n=len(g))
        for c in cont:
            if c in g:
                v = g[c].to_numpy(float)
                row[f"{c}_mean"] = np.nanmean(v) if np.isfinite(v).any() else np.nan
                row[f"{c}_sd"] = np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1 else np.nan
        time, ev = event_table(g)
        km = kaplan_meier(time, ev)
        for T in probs_at:
            if T <= g["T"].max():
                k = int((ev & (time <= T)).sum()); lo, hi = wilson(k, len(g))
                row[f"P_ext_{T}"] = k / len(g); row[f"P_ext_{T}_lo"] = lo; row[f"P_ext_{T}_hi"] = hi
        med = km_median(km); row["t_ext_median_km"] = med["median"]
        row["t_ext_median_lo"] = med["lo"]; row["t_ext_median_hi"] = med["hi"]
        row["frac_extinct"] = float(ev.mean())
        regs = g.apply(regime, axis=1)
        for r in ("colapso_unico", "oscilatorio", "meseta", "sin_despegue", "explota"):
            row[f"frac_{r}"] = float((regs == r).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def stochastic_share(summary: pd.DataFrame, metric):
    """Fracción de la varianza total de `metric` que es intra-punto (azar) vs entre puntos (parámetros)."""
    g = summary.groupby("point")[metric]
    within = g.var(ddof=1).mean(); total = summary[metric].var(ddof=1)
    return dict(within=float(within), total=float(total), share_stochastic=float(within / total) if total > 0 else np.nan)
