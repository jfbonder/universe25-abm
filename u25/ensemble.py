"""Ensambles paralelos y guardado de resultados."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .params import Params
from .model import Universe


def rep_seed(base_seed: int, point: int, rep: int, crn: bool = False) -> np.random.SeedSequence:
    """Semilla determinística por (base_seed, punto, réplica).

    crn=True usa números aleatorios comunes: la réplica r usa la misma semilla en todos los puntos.
    """
    ent = [int(base_seed), int(rep)] if crn else [int(base_seed), int(point), int(rep)]
    return np.random.SeedSequence(ent)


def run_one(task):
    """Corre una réplica. task = (params_dict, point, rep, T, base_seed, crn, model_cls,
    save_series, record_every). Devuelve (summary_dict, series_df | None)."""
    pdict, point, rep, T, base_seed, crn, model_cls, save_series, record_every = task
    p = Params.from_dict(pdict)
    ss = rep_seed(base_seed, point, rep, crn)
    rng = np.random.default_rng(ss)
    U = (model_cls or Universe)(p, rng=rng)
    series, summ = U.run(T, record_every=record_every)
    summ = {"point": point, "rep": rep, "base_seed": base_seed, **summ}
    if save_series:
        series.insert(0, "rep", rep)
        series.insert(0, "point", point)
        return summ, series
    return summ, None


def _param_columns(plist: Sequence[Params]) -> pd.DataFrame:
    """Columnas de parámetros que varían entre puntos (más alpha siempre)."""
    dicts = [p.to_dict() for p in plist]
    keys = [k for k in dicts[0] if not isinstance(dicts[0][k], list)]
    df = pd.DataFrame([{k: d[k] for k in keys} for d in dicts])
    var = [k for k in keys if df[k].nunique(dropna=False) > 1]
    cols = sorted(set(var) | {"alpha"})
    df = df[cols]
    df.insert(0, "point", np.arange(len(plist)))
    return df


def run_ensemble(params: Params | Sequence[Params], n_reps: int, T: int, base_seed: int = 0,
                 n_jobs: int = 4, out: str | os.PathLike | None = None, save_series: bool = True,
                 record_every: int = 1, model_cls=None, crn: bool = False,
                 series_format: str = "auto", progress: bool = False) -> pd.DataFrame:
    """Corre n_reps réplicas de cada punto de parámetros en paralelo.

    Devuelve el DataFrame de resúmenes (una fila por corrida, con columnas de parámetros que
    varían).  Si `out` no es None guarda en ese directorio:
      summary.csv, series.parquet (o series.csv.gz), params.json, meta.json
    """
    plist = [params] if isinstance(params, Params) else list(params)
    for p in plist:
        p.validate()
    tasks = [(p.to_dict(), i, r, T, base_seed, crn, model_cls, save_series, record_every)
             for i, p in enumerate(plist) for r in range(n_reps)]
    t0 = time.perf_counter()
    summaries, series = [], []
    if n_jobs == 1 or len(tasks) == 1:
        it = map(run_one, tasks)
        pool = None
    else:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        pool = ctx.Pool(processes=n_jobs, maxtasksperchild=50)
        it = pool.imap_unordered(run_one, tasks, chunksize=1)
    try:
        for k, (summ, ser) in enumerate(it):
            summaries.append(summ)
            if ser is not None:
                series.append(ser)
            if progress:
                print(f"[{k + 1}/{len(tasks)}] point={summ['point']} rep={summ['rep']} "
                      f"N_max={summ['N_max']} t_ext={summ['t_ext']} wall={summ['wall_time_s']:.1f}s",
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    wall = time.perf_counter() - t0
    df = pd.DataFrame(summaries).sort_values(["point", "rep"]).reset_index(drop=True)
    pc = _param_columns(plist)
    df = pc.merge(df, on="point")
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "summary.csv", index=False)
        with open(out / "params.json", "w") as f:
            json.dump([json.loads(p.to_json()) for p in plist], f, indent=1)
        meta = dict(n_reps=n_reps, T=T, base_seed=base_seed, n_jobs=n_jobs, crn=crn,
                    record_every=record_every, n_points=len(plist), wall_time_s=wall,
                    model_cls=(model_cls.__module__ + "." + model_cls.__qualname__) if model_cls else "u25.model.Universe")
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        if series:
            S = pd.concat(series, ignore_index=True).sort_values(["point", "rep", "t"])
            fmt = series_format
            if fmt == "auto":
                try:
                    import pyarrow  # noqa: F401
                    fmt = "parquet"
                except Exception:
                    fmt = "csv"
            if fmt == "parquet":
                S.to_parquet(out / "series.parquet", index=False)
            else:
                S.to_csv(out / "series.csv.gz", index=False)
    df.attrs["wall_time_s"] = wall
    return df
