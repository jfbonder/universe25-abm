"""Fase 9a (revisión EM, ronda 1): ensambles nuevos y baratos para A1, A2 y A5.

    python3 scripts/em/em_f9_runs.py GRUPO

Grupos (todos con el evaluador primario de PRE: results_v2/f2b_base/eval_kw.json):

  sp        ABM espacial, u25.sweep + f2b_model_v2.U2v2 (mismo runner que f2b_base), L = 30, n = 200, T = 600,
            series completas -> results_em/spatial_reversible_null/
              p0  nulo de supresión reversible: C1 con recuperación adulta eps_ad = plast_outside_rec = 0.8 y
                  lambda = social_rec_lambda = 0.9 (resto de C1 igual). Umbral de rescate espacial a lambda = 0.9:
                  0.46 / 0.38 / 0.31 a L = 15 / 30 / 60 (informe_f7, Tabla 1.3); 0.8 está por encima en los tres
                  tamaños y P_ext(1000) = 0 en 20 + 20 + 10 colonias (f7_e1).
              p1  control de vida corta (SL), parámetros de results_v2/f2b_ref punto 0 (a_max 60, alpha 1.4,
                  recuperación individual, sin reglas ni refugios).
  sp1000    las mismas dos configuraciones y semillas a T = 1000 (sólo supervivencia) -> ..._f9_sp1000/
            (el modelo no depende de T: las corridas de T = 600 son el prefijo de éstas).
  nulos     re-corrida bit a bit de results_v2/f2b_nulos puntos 0 (gamma = 0) y 3 (núcleo), misma semilla, ahora
            guardando las series (para puntuar O11 a la edad a_w) -> ..._f9_nulos/
  wm        bien mezclado (F7WM = WMUniverse de wm_abm.py con apareamiento global de f7_common: el padre se elige
            al azar entre todos los machos reproductivos de la arena), evaluador primario y O11 a a_w:
              L = 30, T = 600 (patrones) y T = 1000 (supervivencia; mismas semillas), n = 200:
                C1_ref (sitios de retiro como capacidad, = punto 0 de em_wm_run.py), C1_noref, SL, core
                (core = "WM_none" de la matriz de patrones: C1 sin sus cuatro reglas, longevo, a_max 120);
              L = 45 y 60 (densidad inicial fija 400/900), T = 1000, n = 200: C1_ref, SL, core.
            -> results_em/wellmixed_global_mating/per_run.csv
Semillas: sp/sp1000 SeedSequence([20261018, punto, réplica]); nulos las de f2b_nulos (25330); wm
SeedSequence([20261018, crc32(cfg), réplica]).
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "scripts" / "production"), str(ROOT / "scripts" / "analysis"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from u25.params import Params, preset     # noqa: E402

RES = ROOT / "results_em"
EVAL_KW = json.load(open(ROOT / "results_v2" / "f2b_base" / "eval_kw.json"))
EPS_REV, LAM_REV = 0.8, 0.9


def _sp_pmap(T):
    c1 = preset("calhoun_C1").replace(plast_outside_rec=EPS_REV, social_rec_lambda=LAM_REV, T=T)
    sl = Params.from_dict(dict(json.load(open(ROOT / "results_v2" / "f2b_ref" / "params.json"))[0], T=T))
    return {0: c1, 1: sl}


def run_sp(T=600, out="spatial_reversible_null", series=1):
    from u25.sweep import run_sweep, collect_series
    from f2b_model_v2 import U2v2
    pm = _sp_pmap(T)
    design = pd.DataFrame(dict(point=[0, 1], label=["C1 reversible (eps_ad=0.8, lambda=0.9)", "SL control"]))
    o = RES / out
    o.mkdir(parents=True, exist_ok=True)
    json.dump(["C1 reversible adults (eps_ad=0.8, lambda=0.9)", "short-lived control (f2b_ref p0)"],
              open(o / "labels.json", "w"), indent=1)
    res = run_sweep(design[["point"]], reps=200, out=o, T=T, base_seed=20261018, n_jobs=4, crn=False,
                    series_every=series, evaluate=True, model_cls=U2v2, eval_kw=EVAL_KW, params_map=pm)
    print(res)
    if series:
        print(collect_series(o))


def run_base():
    """Re-corrida bit a bit de results_v2/f2b_base punto 0 (C1 espacial de referencia) con las series del código
    vigente, que registra s a la edad a_w (O11 a a_w; las series de PRE no tienen esas columnas)."""
    from u25.sweep import run_sweep
    from f2b_model_v2 import U2v2
    src = ROOT / "results_v2" / "f2b_base"
    P = json.load(open(src / "params.json"))
    meta = json.load(open(src / "meta.json"))
    o = RES / "spatial_c1_rerun"
    print(run_sweep(pd.DataFrame(dict(point=[0])), reps=meta["n_reps"], out=o, T=meta["T"],
                    base_seed=meta["base_seed"], n_jobs=4, crn=False, series_every=1, evaluate=True, model_cls=U2v2,
                    eval_kw=meta["eval_kw"], params_map={0: Params.from_dict(P[0])}))
    from u25.sweep import collect_series
    print(collect_series(o))
    a = pd.read_csv(src / "summary.csv")
    a = a[a.point == 0].set_index(["point", "rep"]).sort_index()
    b = pd.read_csv(o / "summary.csv").set_index(["point", "rep"]).sort_index()
    cols = [c for c in a.columns if c.startswith("obj_") and c in b.columns] + ["t_ext", "N_max", "t_end"]
    bad = [c for c in cols if not (a[c].fillna("NA").astype(str) == b[c].fillna("NA").astype(str)).all()]
    print("bit a bit frente a f2b_base:", "OK" if not bad else f"difieren {bad}")
    json.dump(dict(identical=not bad, differ=bad), open(o / "check_vs_f2b_base.json", "w"), indent=1)


def run_nulos():
    from u25.sweep import run_sweep, collect_series
    from f2b_model_v2 import U2v2
    src = ROOT / "results_v2" / "f2b_nulos"
    P = json.load(open(src / "params.json"))
    meta = json.load(open(src / "meta.json"))
    pm = {0: Params.from_dict(P[0]), 3: Params.from_dict(P[3])}
    o = RES / "spatial_nulls_rerun"
    res = run_sweep(pd.DataFrame(dict(point=[0, 3])), reps=meta["n_reps"], out=o, T=meta["T"],
                    base_seed=meta["base_seed"], n_jobs=4, crn=False, series_every=1, evaluate=True, model_cls=U2v2,
                    eval_kw=meta["eval_kw"], params_map=pm)
    print(res)
    print(collect_series(o))
    a = pd.read_csv(src / "summary.csv")
    b = pd.read_csv(o / "summary.csv")
    a = a[a.point.isin([0, 3])].set_index(["point", "rep"]).sort_index()
    b = b.set_index(["point", "rep"]).sort_index()
    cols = [c for c in a.columns if c.startswith("obj_") and c in b.columns] + ["t_ext", "N_max", "t_end"]
    same = {c: bool((a[c].fillna("NA").astype(str) == b[c].fillna("NA").astype(str)).all()) for c in cols}
    bad = [c for c, v in same.items() if not v]
    print("bit a bit frente a f2b_nulos:", "OK" if not bad else f"difieren {bad}")
    json.dump(dict(identical=not bad, differ=bad), open(o / "check_vs_f2b_nulos.json", "w"), indent=1)


# ----------------------------------------------------------------------------- bien mezclado, apareamiento global
def wm_params(cfg, L, T):
    from wm_abm import make_params
    if cfg == "C1_ref":
        return make_params("C1", L, refuges=True, T=T)
    if cfg == "C1_noref":
        return make_params("C1", L, refuges=False, T=T)
    if cfg == "SL":
        return make_params("SL", L, refuges=False, T=T)
    if cfg == "core":
        return make_params("SL", L, refuges=False, T=T, a_max=120.0, repro_age_max=80,
                           max_pop=int(15000 * L * L / 900))
    raise KeyError(cfg)


def phase_c(S):
    """Duración relativa de la fase C (copia de em_figs.phase_times): t_BC = la tasa de crecimiento de ln N
    (diferencia centrada a 4 semanas) cae bajo la mitad de su máximo; t_D = comienzo de la meseta N >= 0.98 max N.
    Devuelve t_BC, t_D, (t_D - t_BC)/t_D (cláusula de O3: >= 0.10) y r_B/r_C."""
    t = S["t"].to_numpy(float); N = S["N"].to_numpy(float); B = S["births"].to_numpy(float)
    try:
        tA = t[np.flatnonzero(B > 0)[0]]
        lN = np.log(np.maximum(N, 1e-12))
        r = np.full(N.size, np.nan); r[2:-2] = (lN[4:] - lN[:-4]) / 4
        imax = int(np.nanargmax(np.where(t <= 60, r, np.nan)))
        tBC = t[np.flatnonzero((np.arange(N.size) > imax) & (r < 0.5 * r[imax]))[0]]
        iD = int(np.flatnonzero(N >= 0.98 * N.max())[0]); tD = t[iD]
        iBC = int(np.searchsorted(t, tBC))
        rB = (lN[iBC] - lN[int(np.searchsorted(t, tA))]) / max(tBC - tA, 1)
        rC = (lN[iD] - lN[iBC]) / max(tD - tBC, 1)
        return dict(ph_tBC=tBC, ph_tD=tD, ph_Cfrac=(tD - tBC) / tD, ph_rBrC=rB / rC if rC > 0 else np.inf)
    except Exception:  # noqa: BLE001
        return dict(ph_tBC=np.nan, ph_tD=np.nan, ph_Cfrac=np.nan, ph_rBrC=np.nan)


def _eval(S, p, summ):
    from u25.objectives import evaluate_run
    r = evaluate_run(S, p, summ, **EVAL_KW)
    out = {}
    for c, v in r.items():
        if not c.startswith("_"):
            out[c] = v.status
            out[c + "_val"] = v.value
    for c in ("t_pk", "N_pk", "t_B99", "t_B0", "extinct", "t_end"):
        out["x_" + c] = r["_times"].get(c, np.nan) if isinstance(r.get("_times"), dict) else np.nan
    raw = evaluate_run(S, p, summ, **EVAL_KW, o11_age="aw")["O11"]
    out["O11aw"] = raw.status
    out["O11aw_val"] = raw.value
    out.update(phase_c(S))
    return out


def one_wm(task):
    from f7_common import F7WM
    cfg, L, k, do_eval = task
    key = f"{cfg}_global_L{L}"
    ss = np.random.SeedSequence([20261018, zlib.crc32(key.encode()), k])
    out = dict(cfg=cfg, L=L, rep=k, mate="global")
    T_long = 1000
    p = wm_params(cfg, L, T_long)
    U = F7WM(p, rng=np.random.default_rng(ss), opts=dict(mate="global", obs=False))
    U.measure_theta = False
    S, summ = U.run(T_long)
    out.update(extinct=bool(summ["extinct"]), t_ext=float(summ["t_ext"]) if summ["extinct"] else np.nan,
               exploded=bool(summ.get("exploded", False)), N_max=int(summ["N_max"]), t_end=int(summ["t_end"]),
               N_final=int(S["N"].iloc[-1]))
    B = S["births"].to_numpy()
    nz = np.flatnonzero(B > 0)
    out["t_lastbirth"] = int(S["t"].to_numpy()[nz[-1]]) if nz.size else -1
    if do_eval:
        p6 = wm_params(cfg, L, 600)
        U6 = F7WM(p6, rng=np.random.default_rng(np.random.SeedSequence([20261018, zlib.crc32(key.encode()), k])),
                  opts=dict(mate="global", obs=False))
        U6.measure_theta = False
        S6, s6 = U6.run(600)
        assert np.array_equal(S6["N"].to_numpy(), S["N"].to_numpy()[: len(S6)]), "prefijo distinto"
        out.update(_eval(S6, p6, s6))
        out["t_ext600"] = float(s6["t_ext"]) if s6["extinct"] else np.nan
    return out


def run_wm(n=200):
    tasks = [(c, 30, k, True) for c in ("C1_ref", "C1_noref", "SL", "core") for k in range(n)]
    tasks += [(c, L, k, False) for L in (45, 60) for c in ("C1_ref", "SL", "core") for k in range(n)]
    o = RES / "wellmixed_global_mating"
    o.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(4) as pool:
        rows = pool.map(one_wm, tasks, chunksize=2)
    pd.DataFrame(rows).to_csv(o / "per_run.csv", index=False)
    json.dump(dict(n=n, seeds="SeedSequence([20261018, crc32(cfg_global_L), rep])", eval_kw=EVAL_KW,
                   model="f7_common.F7WM(mate='global')", T_eval=600, T_surv=1000,
                   wall_s=round(time.time() - t0)), open(o / "meta.json", "w"), indent=1)
    print("wm global:", len(rows), "corridas,", round(time.time() - t0), "s")


def one_wmcell(task):
    """Re-corrida bit a bit de la primera corrida del ensamble bien mezclado (wellmixed_cell_reference) (WMUniverse, apareamiento en la celda, mismas
    semillas SeedSequence([20261017, crc32(nombre), k])) para agregar O11 a a_w (ya estaba) y la fase C."""
    from wm_abm import WMUniverse
    name, k = task
    cfg = {"WM_SL": "SL", "WM_none": "core"}[name]
    p = wm_params(cfg, 30, 600)
    U = WMUniverse(p, rng=np.random.default_rng(np.random.SeedSequence([20261017, zlib.crc32(name.encode()), k])))
    U.measure_theta = False
    S, summ = U.run(600)
    out = dict(cfg=name, rep=k, t_ext=float(summ["t_ext"]) if summ["extinct"] else np.nan)
    out.update(_eval(S, p, summ))
    return out


def run_wmcell(n=200):
    tasks = [(c, k) for c in ("WM_SL", "WM_none") for k in range(n)]
    with Pool(4) as pool:
        rows = pd.DataFrame(pool.map(one_wmcell, tasks, chunksize=4))
    o = RES / "wellmixed_cell_rerun"
    o.mkdir(parents=True, exist_ok=True)
    rows.to_csv(o / "per_run.csv", index=False)
    ref = pd.read_csv(RES / "wellmixed_cell_reference" / "per_run.csv")
    ref = ref[ref.cfg.isin(["WM_SL", "WM_none"])].set_index(["cfg", "rep"]).sort_index()
    new = rows.set_index(["cfg", "rep"]).sort_index()
    cols = [c for c in ("O5", "O6", "O7", "O8", "O9", "O10", "O10b", "O11", "O11aw", "O13", "O14", "O15") if c in ref]
    bad = [c for c in cols if not (ref[c].fillna("NA").astype(str) == new[c].fillna("NA").astype(str)).all()]
    same_t = bool(np.allclose(ref.t_ext.fillna(-1), new.t_ext.fillna(-1)))
    json.dump(dict(identical=(not bad) and same_t, differ=bad, t_ext_same=same_t), open(o / "check_vs_reference.json",
              "w"), indent=1)
    print("wmcell: idéntico a la referencia" if not bad and same_t else f"wmcell: difiere {bad} t_ext {same_t}")


if __name__ == "__main__":
    g = sys.argv[1]
    if g == "sp":
        run_sp()
    elif g == "sp1000":
        run_sp(T=1000, out="spatial_reversible_null_T1000", series=0)
    elif g == "base":
        run_base()
    elif g == "nulos":
        run_nulos()
    elif g == "wmcell":
        run_wmcell()
    elif g == "wm":
        run_wm(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        raise SystemExit(__doc__)
