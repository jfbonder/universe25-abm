"""Experimentos in silico. Por ahora: transplante (O12; especificación §O12; P_T).

Protocolo (especificación §O12, versión verificada contra Calhoun 1973)
----------------------------------------------------------------------
1. Corrida "madre" por réplica (semilla propia) hasta `T_main`. Sobre su serie se calculan
   - t_D = t_pk + 0.75 t_pk (= 1.75 t_pk, "mid-third of Phase D"); si en t_D no hay 4+4 adultos
     reproductivos (o la madre ya se extinguió) se usa t_D' = t_pk + max(4, 0.1 t_pk) y la fila lleva
     early_D = True;
   - t_B = punto medio de W_B = [t_1, t_2] (fase B, control; ver objectives.characteristic_times).
   La corrida madre se repite con la misma semilla (es determinística) para tomar los snapshots en
   t_B y t_D.
2. De cada snapshot se eligen al azar k machos y k hembras en edad reproductiva (k=4); si no hay
   suficientes se toman los disponibles y se reporta (n_M, n_F). Se ubican juntos en la celda central
   de una grilla vacía (founders_same_cell), sin afinidades, refractario 0, con su edad, s y techo.
3. Se corre T_after = 150 semanas con los mismos parámetros. Fundación = N >= factor * N_inicial
   (10 x 8 = 80) en algún momento.
4. O12b (transplante mixto): 4 individuos del snapshot D de un sexo + 4 del sexo opuesto sanos (s=1,
   edad `healthy_age`, criados sin hacinamiento), en las dos combinaciones: fases "MIX_M" (machos D +
   hembras sanas) y "MIX_F" (hembras D + machos sanos).
Salida: formato largo, una fila por transplante (rep, phase ∈ {B, D, MIX_M, MIX_F}, success, ...).
Con n_trials=1 cada transplante viene de una madre distinta (lo que pide la especificación).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from .params import Params
from .model import Universe
from .ensemble import rep_seed


def _snap(U: Universe) -> dict:
    return dict(t=U.t, N=U.N, male=U.male.copy(), age=U.age.copy(), s=U.s.copy(), cap=U.cap.copy(),
                h_mat=U.h_mat.copy())


def snapshot_times(series: pd.DataFrame, params: Params) -> dict:
    from .objectives import characteristic_times
    ct = characteristic_times(series, params)
    tD = float(round(1.75 * ct["t_pk"]))
    tDe = float(ct["t_pk"] + max(4.0, round(0.1 * ct["t_pk"])))
    tB = np.nan
    if np.isfinite(ct["t1"]) and np.isfinite(ct["t2"]):
        tB = float(np.floor((ct["t1"] + ct["t2"]) / 2))
    return dict(t_B=tB, t_D=tD, t_De=tDe, t_pk=ct["t_pk"], N_pk=ct["N_pk"])


def _n_repro(snap, p):
    if snap is None:
        return 0, 0
    rep = (snap["age"] >= p.repro_age_min) & (snap["age"] <= p.repro_age_max)
    return int((rep & snap["male"]).sum()), int((rep & ~snap["male"]).sum())


def colony_snapshots(params: Params, seed_seq, T_main: int = 400, model_cls=Universe):
    """Corre la madre, calcula t_B, t_D y t_D' y la repite para tomar los snapshots. Devuelve
    (times, snap_B, snap_D, summary); times["early_D"] indica si se usó el fallback t_D'."""
    U = model_cls(params, rng=np.random.default_rng(seed_seq))
    series, summ = U.run(T_main)
    times = snapshot_times(series, params)
    if times["t_D"] > T_main:          # la madre no llegó a t_D: extenderla
        U = model_cls(params, rng=np.random.default_rng(seed_seq))
        series, summ = U.run(int(times["t_D"]))
        times = snapshot_times(series, params)
    want = {k: times[k] for k in ("t_B", "t_D", "t_De") if np.isfinite(times[k])}
    snaps = {}
    if want:
        U2 = model_cls(params, rng=np.random.default_rng(seed_seq))

        def cb(U):
            for k, tt in want.items():
                if U.t == tt:
                    snaps[k] = _snap(U)

        U2.run(int(max(want.values())), callback=cb)
    sD = snaps.get("t_D")
    nM, nF = _n_repro(sD, params)
    times["early_D"] = False
    if sD is None or nM < 4 or nF < 4:
        sD = snaps.get("t_De")
        times["early_D"] = True
        times["t_D"] = times["t_De"]
    return times, snaps.get("t_B"), sD, summ


def transplant_trial(params: Params, snap: dict | None, rng: np.random.Generator, k_pairs: int = 4,
                     T_after: int = 150, factor: float = 10.0, same_cell: bool = True,
                     model_cls=Universe) -> dict:
    """Un transplante. Devuelve dict(success ∈ {1, 0, NaN}, n_M, n_F, s_mean, N_max)."""
    out = dict(success=np.nan, n_M=0, n_F=0, s_mean=np.nan, N_max=np.nan)
    if snap is None:
        return out
    p = params
    rep = (snap["age"] >= p.repro_age_min) & (snap["age"] <= p.repro_age_max)
    M = np.flatnonzero(rep & snap["male"])
    F = np.flatnonzero(rep & ~snap["male"])
    kM, kF = min(k_pairs, M.size), min(k_pairs, F.size)
    out.update(n_M=kM, n_F=kF)
    if kM + kF == 0:
        return out
    sel = np.concatenate([rng.choice(M, kM, replace=False), rng.choice(F, kF, replace=False)])
    out["s_mean"] = float(snap["s"][sel].mean())
    U = model_cls.from_agents(p.replace(founders_same_cell=same_cell), rng, snap["male"][sel],
                              snap["age"][sel], snap["s"][sel], cap=snap["cap"][sel],
                              h_mat=snap["h_mat"][sel] if "h_mat" in snap else None)
    target = factor * 2 * k_pairs
    nmax = U.N
    while U.t < T_after and U.N > 0:
        U.step()
        nmax = max(nmax, U.N)
        if U.N >= target:
            out.update(success=1.0, N_max=nmax)
            return out
    out.update(success=0.0, N_max=nmax)
    return out


def mixed_trial(params: Params, snap: dict | None, rng: np.random.Generator, d_sex_male: bool,
                k: int = 4, healthy_age: int = 10, T_after: int = 150, factor: float = 10.0,
                model_cls=Universe) -> dict:
    """O12b: k individuos D del sexo indicado + k sanos (s=1) del sexo opuesto."""
    out = dict(success=np.nan, n_M=0, n_F=0, s_mean=np.nan, N_max=np.nan)
    if snap is None:
        return out
    p = params
    rep = (snap["age"] >= p.repro_age_min) & (snap["age"] <= p.repro_age_max)
    pool = np.flatnonzero(rep & (snap["male"] if d_sex_male else ~snap["male"]))
    kd = min(k, pool.size)
    if kd == 0:
        return out
    sel = rng.choice(pool, kd, replace=False)
    male = np.r_[np.full(kd, d_sex_male), np.full(k, not d_sex_male)]
    age = np.r_[snap["age"][sel], np.full(k, healthy_age)]
    s = np.r_[snap["s"][sel], np.ones(k)]
    cap = np.r_[snap["cap"][sel], np.full(k, np.inf)]
    hm = np.r_[snap["h_mat"][sel] if "h_mat" in snap else np.ones(kd), np.ones(k)]
    out.update(n_M=int(male.sum()), n_F=int((~male).sum()), s_mean=float(snap["s"][sel].mean()))
    U = model_cls.from_agents(p.replace(founders_same_cell=True), rng, male, age, s, cap=cap, h_mat=hm)
    target = factor * 2 * k
    nmax = U.N
    while U.t < T_after and U.N > 0:
        U.step()
        nmax = max(nmax, U.N)
        if U.N >= target:
            out.update(success=1.0, N_max=nmax)
            return out
    out.update(success=0.0, N_max=nmax)
    return out


def _one(task):
    pdict, rep, base_seed, T_main, k_pairs, n_trials, T_after, factor, same_cell, model_cls, mixed = task
    p = Params.from_dict(pdict)
    mc = model_cls or Universe
    times, sB, sD, summ = colony_snapshots(p, rep_seed(base_seed, 0, rep), T_main, mc)
    trng = np.random.default_rng(rep_seed(base_seed, 1, rep))
    rows = []
    for phase, snap in (("B", sB), ("D", sD)):
        for k in range(n_trials):
            r = transplant_trial(p, snap, trng, k_pairs, T_after, factor, same_cell, mc)
            r.update(rep=rep, trial=k, phase=phase, t_snap=times[f"t_{phase}"], t_pk=times["t_pk"],
                     early_D=bool(times["early_D"]) if phase == "D" else False,
                     S_snap=float(snap["s"].mean()) if snap is not None and snap["N"] else np.nan)
            rows.append(r)
    if mixed:
        for phase, dm in (("MIX_M", True), ("MIX_F", False)):
            for k in range(n_trials):
                r = mixed_trial(p, sD, trng, dm, k_pairs, T_after=T_after, factor=factor, model_cls=mc)
                r.update(rep=rep, trial=k, phase=phase, t_snap=times["t_D"], t_pk=times["t_pk"],
                         early_D=bool(times["early_D"]),
                         S_snap=float(sD["s"].mean()) if sD is not None and sD["N"] else np.nan)
                rows.append(r)
    return rows


def transplant_experiment(params: Params, n_reps: int = 30, base_seed: int = 0, k_pairs: int = 4,
                          n_trials: int = 1, T_main: int = 400, T_after: int = 150, factor: float = 10.0,
                          same_cell: bool = True, n_jobs: int = 4, model_cls=None,
                          out: str | None = None, mixed: bool = True) -> pd.DataFrame:
    """Experimento de transplante. Devuelve el DataFrame largo (una fila por transplante) con
    columnas rep, trial, phase, success, n_M, n_F, s_mean, N_max, t_snap, t_pk, S_snap.
    df.attrs["O12"] = Result del criterio (objectives.transplant_status)."""
    from .objectives import transplant_status, mixed_transplant_status
    tasks = [(params.to_dict(), r, base_seed, T_main, k_pairs, n_trials, T_after, factor, same_cell,
              model_cls, mixed) for r in range(n_reps)]
    if n_jobs == 1 or n_reps == 1:
        res = list(map(_one, tasks))
    else:
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        with ctx.Pool(n_jobs) as pool:
            res = pool.map(_one, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df = df.sort_values(["phase", "rep", "trial"]).reset_index(drop=True)
    o12 = transplant_status(df.loc[df.phase == "B", "success"], df.loc[df.phase == "D", "success"])
    df.attrs["O12"] = o12
    df.attrs["O12b"] = mixed_transplant_status(df.loc[df.phase.str.startswith("MIX"), "success"])
    df.attrs["P_T_B"] = float(df.loc[df.phase == "B", "success"].mean())
    df.attrs["P_T_D"] = float(df.loc[df.phase == "D", "success"].mean())
    if out is not None:
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "transplant.csv"), index=False)
        with open(os.path.join(out, "params.json"), "w") as f:
            f.write(params.to_json())
        with open(os.path.join(out, "O12.txt"), "w") as f:
            f.write(f"{o12.status}\t{o12.value}\t{o12.detail}\n")
            o = df.attrs["O12b"]
            f.write(f"O12b\t{o.status}\t{o.value}\t{o.detail}\n")
    return df
