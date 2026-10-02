"""Transplantes de la V2 (etapa `trans`).

Por réplica: una colonia madre (semilla propia; con la misma semilla en todos los brazos que comparten madre, como
el cruzado X) y, sobre ella, estos transplantes de 4 machos + 4 hembras en edad reproductiva a una grilla vacía
(protocolo O12 de u25.experiments: celda central, T_after semanas, éxito = N >= 80):

  B          fase B (control sano), como en la v1;
  D          t_D = 1,75 t_pk con el t_pk PRIMARIO de la V2 (v2_api.json: plateau_start = t'_pk, primera semana con
             N >= 0,98 N_max), con el fallback early-D: Q1 y O12;
  Darg       1,75 t_pk con t_pk = argmax de N: el protocolo de la v1 y del prerregistro (sensibilidad);
  Dfix       tiempo ABSOLUTO común a todos los brazos, t_fix (95 semanas): compara brazos sin confundir la regla
             con el momento del muestreo;
  MIX_M/F    O12b en t_D: animales D de un sexo + sanos del otro (s = 1, edad 10), como en la v1;
  MIXfix_M/F lo mismo en t_fix;
  BAND       4 + 4 animales reproductivos, pasada la ventana crítica (edad >= a_w), con s en [0,2; 0,5]
             (primera semana en que hay 4 de cada sexo en la banda): ¿su descendencia recupera la competencia?;
  BAND_MIX_M/F  animales de la banda de un sexo + sanos del otro.

Cada fila registra, de los animales trasladados (en los mixtos, de los de la madre): s_mean, s_med, s_max,
frac_s0 (s < 1e-3), age_med/min/max; y de la colonia nueva: success, t_found, N_max y, al final (hasta
cont_after semanas después de fundar o N > n_cap), la competencia de los nacidos en ella: n_desc, n_desc_ad
(edad >= a_w), s_desc_ad (media de s de esos adultos) y s_desc_mat (media de s al cumplir repro_age_min).
Cada fase usa su propio flujo de números aleatorios: los brazos que comparten madre trasladan los mismos animales.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from u25 import Params
from u25.ensemble import rep_seed
from u25.experiments import snapshot_times, _snap

PHASES = ["B", "D", "Darg", "Dfix", "MIX_M", "MIX_F", "MIXfix_M", "MIXfix_F", "BAND", "BAND_MIX_M", "BAND_MIX_F"]
PEAK_PRIMARY = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_api.json")))[
    "evaluador"].get("primario", {}).get("peak_time", "argmax")


def _plateau_start(series, tol=0.02):
    N = series["N"].to_numpy(float)
    t = series["t"].to_numpy(float)
    return float(t[np.flatnonzero(N >= (1 - tol) * N.max())[0]])


def mother(params: Params, seed_seq, T_main, model_cls, t_fix, band, k=4):
    """Corre la madre dos veces (determinística con la misma semilla) y toma los snapshots."""
    p = params
    lo, hi = band
    snaps = {}
    amin = _aw(p)

    def cb1(U):
        if U.t == t_fix:
            snaps["Dfix"] = _snap(U)
        if "BAND" not in snaps and U.N:
            rep = ((U.age >= max(p.repro_age_min, amin)) & (U.age <= p.repro_age_max) & (U.s >= lo)
                   & (U.s <= hi))
            if (rep & U.male).sum() >= k and (rep & ~U.male).sum() >= k:
                snaps["BAND"] = _snap(U)
    U = model_cls(p, rng=np.random.default_rng(seed_seq))
    series, summ = U.run(int(max(T_main, t_fix)), callback=cb1)
    times = snapshot_times(series, p)                     # t_B (no depende de t_pk) y t_pk = argmax
    Nn = series["N"].to_numpy(float)
    tt = series["t"].to_numpy(float)
    try:
        from u25.objectives import peak_index
        tpk1 = float(tt[peak_index(Nn, PEAK_PRIMARY)])
    except Exception:  # noqa: BLE001
        tpk1 = times["t_pk"] if PEAK_PRIMARY == "argmax" else _plateau_start(series)
    times["t_pk_arg"] = times["t_pk"]
    times["t_pk"] = tpk1
    times["t_pk2"] = _plateau_start(series)
    times["t_Darg"] = float(round(1.75 * times["t_pk_arg"]))
    times["t_D"] = float(round(1.75 * tpk1))
    times["t_De"] = float(tpk1 + max(4.0, round(0.1 * tpk1)))
    times["t_Dfix"] = float(t_fix)
    times["t_BAND"] = float(snaps["BAND"]["t"]) if "BAND" in snaps else np.nan
    want = {k_: times[k_] for k_ in ("t_B", "t_D", "t_De", "t_Darg") if np.isfinite(times[k_])}
    if want:
        U2 = model_cls(p, rng=np.random.default_rng(seed_seq))

        def cb2(U):
            for k_, tq in want.items():
                if U.t == tq:
                    snaps[k_] = _snap(U)
        U2.run(int(max(want.values())), callback=cb2)
    sD = snaps.get("t_D")
    rep = None if sD is None else (sD["age"] >= p.repro_age_min) & (sD["age"] <= p.repro_age_max)
    times["early_D"] = False
    if sD is None or (rep & sD["male"]).sum() < k or (rep & ~sD["male"]).sum() < k:
        sD = snaps.get("t_De")
        times["early_D"] = True
        times["t_D"] = times["t_De"]
    out = dict(B=snaps.get("t_B"), D=sD, Darg=snaps.get("t_Darg"), Dfix=snaps.get("Dfix"), BAND=snaps.get("BAND"))
    return times, out


def _aw(p):
    """Edad mínima de los animales de la banda: pasada la ventana crítica (a_w; 12 si no hay ventana)."""
    return int(p.plast_age_max) if math.isfinite(p.plast_age_max) else 12


def _pick(snap, p, rng, k, male: bool, band=None):
    rep = (snap["age"] >= p.repro_age_min) & (snap["age"] <= p.repro_age_max) & (snap["male"] == male)
    if band is not None:
        rep &= (snap["s"] >= band[0]) & (snap["s"] <= band[1]) & (snap["age"] >= _aw(p))
    pool = np.flatnonzero(rep)
    kk = min(k, pool.size)
    return rng.choice(pool, kk, replace=False) if kk else np.zeros(0, np.int64)


def _desc(U, p):
    born = U.birth_t >= 1
    aw = p.plast_age_max if math.isfinite(p.plast_age_max) else 12
    ad = born & (U.age >= aw)
    mat = born & (U.age == p.repro_age_min)
    return dict(n_desc=int(born.sum()), n_desc_ad=int(ad.sum()),
                s_desc_ad=float(U.s[ad].mean()) if ad.any() else np.nan,
                s_desc_all=float(U.s[born].mean()) if born.any() else np.nan,
                s_desc_mat=float(U.s[mat].mean()) if mat.any() else np.nan)


def trial(pc: Params, snap, rng, k=4, mixed=None, band=None, T_after=150, factor=10.0, model_cls=None,
          cont_after=0, n_cap=400, healthy_age=10):
    """Un transplante. mixed: None (k+k de la madre), 'M' (machos de la madre + hembras sanas), 'F' (al revés)."""
    keys = ("success", "t_found", "N_max", "n_M", "n_F", "s_mean", "s_med", "s_max", "frac_s0", "age_med",
            "age_min", "age_max", "n_desc", "n_desc_ad", "s_desc_ad", "s_desc_all", "s_desc_mat")
    out = {k_: np.nan for k_ in keys}
    if snap is None:
        return out
    if mixed is None:
        selM, selF = _pick(snap, pc, rng, k, True, band), _pick(snap, pc, rng, k, False, band)
        sel = np.concatenate([selM, selF])
        if sel.size == 0:
            return out
        male, age, s = snap["male"][sel], snap["age"][sel], snap["s"][sel]
        cap = snap["cap"][sel]
        hm = snap["h_mat"][sel] if "h_mat" in snap else np.ones(sel.size)
        own = np.ones(sel.size, bool)
    else:
        dm = mixed == "M"
        sel = _pick(snap, pc, rng, k, dm, band)
        if sel.size == 0:
            return out
        male = np.r_[np.full(sel.size, dm), np.full(k, not dm)]
        age = np.r_[snap["age"][sel], np.full(k, healthy_age)]
        s = np.r_[snap["s"][sel], np.ones(k)]
        cap = np.r_[snap["cap"][sel], np.full(k, np.inf)]
        hm = np.r_[snap["h_mat"][sel] if "h_mat" in snap else np.ones(sel.size), np.ones(k)]
        own = np.r_[np.ones(sel.size, bool), np.zeros(k, bool)]
    so, ao = s[own], age[own]
    out.update(n_M=int(male.sum()), n_F=int((~male).sum()), s_mean=float(so.mean()), s_med=float(np.median(so)),
               s_max=float(so.max()), frac_s0=float((so < 1e-3).mean()), age_med=float(np.median(ao)),
               age_min=float(ao.min()), age_max=float(ao.max()))
    U = model_cls.from_agents(pc.replace(founders_same_cell=True), rng, male, age, s, cap=cap, h_mat=hm)
    target = factor * len(male)
    nmax, t_found = U.N, np.nan
    while U.t < T_after and U.N > 0:
        U.step()
        nmax = max(nmax, U.N)
        if np.isnan(t_found) and U.N >= target:
            t_found = float(U.t)
            if cont_after <= 0:
                break
        if np.isfinite(t_found) and (U.t >= t_found + cont_after or U.N > n_cap):
            break
    out.update(success=float(np.isfinite(t_found)), t_found=t_found, N_max=float(nmax))
    if cont_after > 0 and U.N:
        out.update(_desc(U, pc))
    return out


def _one(task):
    pm, pc, rep, base_seed, T_main, T_after, t_fix, band, cont_after, model_name = task
    from u25x import model_class
    mc = model_class(model_name)
    pm, pc = Params.from_dict(pm), Params.from_dict(pc)
    times, sn = mother(pm, rep_seed(base_seed, 0, rep), T_main, mc, t_fix, band)
    rows = []
    for i, ph in enumerate(PHASES):
        rng = np.random.default_rng(rep_seed(base_seed, 1 + i, rep))
        if ph in ("B", "D", "Darg", "Dfix"):
            r = trial(pc, sn[ph], rng, T_after=T_after, model_cls=mc)
            tsnap = times.get(f"t_{ph}", np.nan)
        elif ph.startswith("MIXfix"):
            r = trial(pc, sn["Dfix"], rng, mixed=ph[-1], T_after=T_after, model_cls=mc)
            tsnap = times["t_Dfix"]
        elif ph.startswith("MIX"):
            r = trial(pc, sn["D"], rng, mixed=ph[-1], T_after=T_after, model_cls=mc)
            tsnap = times["t_D"]
        elif ph == "BAND":
            r = trial(pc, sn["BAND"], rng, band=band, T_after=T_after, model_cls=mc, cont_after=cont_after)
            tsnap = times["t_BAND"]
        else:
            r = trial(pc, sn["BAND"], rng, mixed=ph[-1], band=band, T_after=T_after, model_cls=mc,
                      cont_after=cont_after)
            tsnap = times["t_BAND"]
        snap = sn.get("D" if ph.startswith("MIX_") else ("Dfix" if ph.startswith("MIXfix") else
                                                          ("BAND" if ph.startswith("BAND") else ph)))
        r.update(rep=rep, phase=ph, t_snap=tsnap, t_pk=times["t_pk"], t_pk_arg=times["t_pk_arg"],
                 early_D=bool(times["early_D"]) if ph in ("D", "MIX_M", "MIX_F") else False,
                 S_snap=float(snap["s"].mean()) if snap is not None and snap["N"] else np.nan)
        rows.append(r)
    return rows


def experiment(pm: Params, pc: Params, n_reps, base_seed, T_main=300, T_after=150, t_fix=95, band=(0.2, 0.5),
               cont_after=40, n_jobs=4, out=None, model_name="v2"):
    from u25.objectives import transplant_status, mixed_transplant_status
    tasks = [(pm.to_dict(), pc.to_dict(), r, base_seed, T_main, T_after, t_fix, tuple(band), cont_after, model_name)
             for r in range(n_reps)]
    if n_jobs == 1:
        res = list(map(_one, tasks))
    else:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_one, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows]).sort_values(["phase", "rep"]).reset_index(drop=True)
    o12 = transplant_status(df.loc[df.phase == "B", "success"], df.loc[df.phase == "D", "success"])
    o12b = mixed_transplant_status(df.loc[df.phase.isin(["MIX_M", "MIX_F"]), "success"])
    o12f = transplant_status(df.loc[df.phase == "B", "success"], df.loc[df.phase == "Dfix", "success"])
    o12a = transplant_status(df.loc[df.phase == "B", "success"], df.loc[df.phase == "Darg", "success"])
    df.attrs.update(O12=o12, O12b=o12b, O12fix=o12f, O12arg=o12a)
    if out is not None:
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "transplant.csv"), index=False)
        with open(os.path.join(out, "params.json"), "w") as f:
            f.write(json.dumps({"madre": json.loads(pm.to_json()), "colonia": json.loads(pc.to_json()),
                                "t_fix": t_fix, "band": list(band), "cont_after": cont_after,
                                "peak_time_primario": PEAK_PRIMARY}, indent=1))
        with open(os.path.join(out, "O12.txt"), "w") as f:
            f.write(f"{o12.status}\t{o12.value}\t{o12.detail}\n")
            f.write(f"O12b\t{o12b.status}\t{o12b.value}\t{o12b.detail}\n")
            f.write(f"O12fix\t{o12f.status}\t{o12f.value}\t{o12f.detail}\n")
            f.write(f"O12arg\t{o12a.status}\t{o12a.value}\t{o12a.detail}\n")
    return df
