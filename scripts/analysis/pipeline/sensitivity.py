"""Análisis de sensibilidad global: Morris (mu*, sigma) y Sobol (S1, ST, IC bootstrap) con SALib."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .design import _salib_problem


def _Y_by_point(per_point_df: pd.DataFrame, metric: str, n_points: int, fill="mean"):
    y = np.full(n_points, np.nan)
    y[per_point_df["point"].to_numpy(int)] = per_point_df[metric].to_numpy(float)
    nan = ~np.isfinite(y)
    if nan.any():
        y[nan] = np.nanmean(y) if fill == "mean" else 0.0
    return y, int(nan.sum())


def morris_analyze(problem, X, per_point_df, metrics, num_levels=4, num_resamples=1000, conf_level=0.95, seed=0):
    """Devuelve DataFrame largo: metric, name, mu, mu_star, sigma, mu_star_conf, n_nan."""
    from SALib.analyze import morris
    sp = _salib_problem(problem)
    rows = []
    for m in metrics:
        Y, n_nan = _Y_by_point(per_point_df, m, len(X))
        if np.nanstd(Y) == 0:
            continue
        Si = morris.analyze(sp, X, Y, num_levels=num_levels, num_resamples=num_resamples, conf_level=conf_level, seed=seed)
        for i, name in enumerate(sp["names"]):
            rows.append(dict(metric=m, name=name, mu=Si["mu"][i], mu_star=Si["mu_star"][i], sigma=Si["sigma"][i],
                             mu_star_conf=Si["mu_star_conf"][i], n_nan=n_nan))
    df = pd.DataFrame(rows)
    if len(df):
        df["mu_star_rel"] = df["mu_star"] / df.groupby("metric")["mu_star"].transform("max")
        df["rank"] = df.groupby("metric")["mu_star"].rank(ascending=False)
    return df


def sobol_analyze(problem, per_point_df, metrics, calc_second_order=False, num_resamples=1000, conf_level=0.95, seed=0):
    """Índices de Sobol (Saltelli) sobre la métrica agregada por punto. DataFrame largo con S1, S1_conf, ST, ST_conf."""
    from SALib.analyze import sobol
    sp = _salib_problem(problem)
    rows = []; S2 = {}
    n_points = int(per_point_df["point"].max()) + 1
    for m in metrics:
        Y, n_nan = _Y_by_point(per_point_df, m, n_points)
        if np.nanstd(Y) == 0:
            continue
        Si = sobol.analyze(sp, Y, calc_second_order=calc_second_order, num_resamples=num_resamples,
                           conf_level=conf_level, seed=seed)
        for i, name in enumerate(sp["names"]):
            rows.append(dict(metric=m, name=name, S1=Si["S1"][i], S1_conf=Si["S1_conf"][i], ST=Si["ST"][i],
                             ST_conf=Si["ST_conf"][i], interaction=Si["ST"][i] - Si["S1"][i], n_nan=n_nan))
        if calc_second_order:
            S2[m] = pd.DataFrame(Si["S2"], index=sp["names"], columns=sp["names"])
    return pd.DataFrame(rows), S2
