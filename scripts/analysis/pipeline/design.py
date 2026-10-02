"""Diseños de experimentos (Morris, LHS, Saltelli/Sobol) -> listas de u25.Params.

Un *problema* es un dict JSON:
{
  "names":  ["alpha", "rec", ...],
  "bounds": [[0, 4], [0.002, 0.05], ...],      # en escala física
  "scale":  ["lin", "log", ...],               # 'log' => se muestrea log10 y se exponencia
  "type":   ["float", "float", "int", ...],    # 'int' => round; 'bool' => > 0.5
  "fixed":  {"a_max": 120, "repro_age_max": 80}   # overrides fijos de Params (opcional)
}
La matriz X que devuelve SALib está en la escala *muestreada* (log10 para 'log'); el DataFrame de diseño
está en escala física. Ambos se guardan (design.csv, X.npy, problem.json) para el análisis posterior.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PROBLEMA_DEFAULT = {
    "names": ["alpha", "rec", "gamma", "m_c", "p0", "delta", "a_max", "repro_age_max"],
    "bounds": [[0.0, 4.0], [0.002, 0.05], [0.02, 0.5], [4, 16], [0.1, 0.6], [0.002, 0.05], [40, 130], [30, 90]],
    "scale": ["lin", "log", "log", "lin", "lin", "log", "lin", "lin"],
    "type": ["float", "float", "float", "int", "float", "float", "float", "int"],
    "fixed": {},
}


def _salib_problem(problem):
    b = []
    for (lo, hi), sc in zip(problem["bounds"], problem.get("scale", ["lin"] * len(problem["names"]))):
        b.append([math.log10(lo), math.log10(hi)] if sc == "log" else [lo, hi])
    return {"num_vars": len(problem["names"]), "names": list(problem["names"]), "bounds": b}


def to_physical(problem, X):
    """X (escala muestreada) -> DataFrame en escala física con tipos."""
    df = pd.DataFrame(np.asarray(X, float), columns=problem["names"])
    scale = problem.get("scale", ["lin"] * len(problem["names"]))
    types = problem.get("type", ["float"] * len(problem["names"]))
    for name, sc, ty in zip(problem["names"], scale, types):
        v = df[name].to_numpy()
        if sc == "log":
            v = 10.0 ** v
        if ty == "int":
            v = np.round(v).astype(int)
        elif ty == "bool":
            v = v > 0.5
        df[name] = v
    return df


def morris_design(problem, r=20, num_levels=4, seed=0, optimal_trajectories=None):
    from SALib.sample import morris
    sp = _salib_problem(problem)
    X = morris.sample(sp, N=r, num_levels=num_levels, optimal_trajectories=optimal_trajectories, seed=seed)
    return X, to_physical(problem, X)


def lhs_design(problem, n=256, seed=0):
    from scipy.stats import qmc
    sp = _salib_problem(problem)
    k = sp["num_vars"]
    U = qmc.LatinHypercube(d=k, optimization="random-cd", seed=seed).random(n)
    lo = np.array([b[0] for b in sp["bounds"]]); hi = np.array([b[1] for b in sp["bounds"]])
    X = lo + U * (hi - lo)
    return X, to_physical(problem, X)


def saltelli_design(problem, N=256, calc_second_order=False, seed=0):
    from SALib.sample import sobol
    sp = _salib_problem(problem)
    X = sobol.sample(sp, N, calc_second_order=calc_second_order, seed=seed)
    return X, to_physical(problem, X)


def to_params(df_phys: pd.DataFrame, problem=None, base=None):
    """DataFrame físico -> lista de u25.Params (importa u25 sólo acá)."""
    from u25 import Params
    base = base or Params()
    fixed = (problem or {}).get("fixed", {}) or {}
    base = base.replace(**_cast(base, fixed))
    plist = []
    for _, row in df_phys.iterrows():
        kw = _cast(base, row.to_dict())
        plist.append(base.replace(**kw))
    return plist


def _cast(base, kw):
    out = {}
    for k, v in kw.items():
        cur = getattr(base, k)
        if isinstance(cur, bool):
            out[k] = bool(v)
        elif isinstance(cur, int):
            out[k] = int(round(float(v)))
        elif isinstance(cur, float):
            out[k] = float(v) if v is not None else math.inf
        else:
            out[k] = v
    return out


def save_design(out_dir, kind, problem, X, df_phys, **meta):
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X.npy", np.asarray(X, float))
    df = df_phys.copy(); df.insert(0, "point", np.arange(len(df)))
    df.to_csv(d / "design.csv", index=False)
    json.dump({"kind": kind, "problem": problem, "n_points": len(df), **meta}, open(d / "problem.json", "w"), indent=1)
    return d


def load_design(design_dir):
    d = Path(design_dir)
    info = json.load(open(d / "problem.json"))
    X = np.load(d / "X.npy")
    df = pd.read_csv(d / "design.csv")
    return info, X, df


def run_design(design_dir, n_reps=5, T=1000, out=None, n_jobs=4, base_seed=0, crn=True, save_series=False,
               record_every=1, base=None, progress=False):
    """Corre el diseño con u25.run_ensemble. Guarda en out (por defecto el mismo directorio)."""
    from u25 import run_ensemble
    info, X, df = load_design(design_dir)
    plist = to_params(df.drop(columns=["point"]), info["problem"], base=base)
    out = Path(out) if out else Path(design_dir)
    summ = run_ensemble(plist, n_reps=n_reps, T=T, base_seed=base_seed, n_jobs=n_jobs, out=out,
                        save_series=save_series, record_every=record_every, crn=crn, progress=progress)
    return summ
