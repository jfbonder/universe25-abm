"""Lanzador de los barridos de producción sobre el runner reanudable de u25 (no modifica u25/).

  python3 scripts/production/u25x.py sweep --design D --reps N --T T --out O [mismas opciones que `python -m u25 sweep`]
         [--full-series] [--no-extras]
  python3 scripts/production/u25x.py transplant --design D --reps N --out O [--preset P] [--set k=v] [--T-main 300]
         [--T-after 150] [--base-seed S] [--n-jobs 4]

sweep: igual que `python -m u25 sweep` (mismo run_sweep, mismas semillas y reanudación) con dos cambios:
  * model_cls = f2b_model.U2b, que agrega columnas al resumen (obj_O13p, x_*; ver f2b_model.py) sin tocar la
    dinámica ni el RNG;
  * la serie guardada (si --series-every > 0) incluye además occ_frac_cells, low_s_frac, I_D, crowded_frac,
    phi_BO, N_adult, frac_in_refuge; con --full-series guarda todas las columnas (para `python -m u25 objectives`).
transplant2: transplantes de la V2 (trans_v2.py; tiempos de muestreo igualados, banda de s intermedia).
  --model v2 en sweep usa f2b_model_v2.U2v2 (re-puntuaciones con t'_pk y umbral fijo).
transplant: un experimento de O12/O12b (u25.experiments.transplant_experiment, con mixed=True) por punto del diseño,
  en <out>/arm_<punto>/; un brazo con O12.txt ya escrito se saltea (reanudable por brazo).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "scripts" / "production"
for p in (str(ROOT), str(ROOT / "scripts" / "analysis"), str(MAT)):
    if p not in sys.path:
        sys.path.insert(0, p)

EXTRA_SERIES = ["occ_frac_cells", "low_s_frac", "I_D", "crowded_frac", "phi_BO", "N_adult", "frac_in_refuge",
                "S_in_refuge", "phi_w", "S_repro_f"]


def model_class(name="v1"):
    """v1: f2b_model.U2b (Fase 2b); v2: f2b_model_v2.U2v2 (producción); none: Universe."""
    if name == "v2":
        from f2b_model_v2 import U2v2
        return U2v2
    if name in ("none", "u25"):
        from u25.model import Universe
        return Universe
    from f2b_model import U2b
    return U2b


def _base(preset, sets):
    from u25.__main__ import _params_from_sets
    return _params_from_sets(sets, preset)


def cmd_sweep(argv):
    ap = argparse.ArgumentParser(prog="u25x.py sweep")
    ap.add_argument("--design", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--crn", action="store_true")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE")
    ap.add_argument("--tie", action="append", metavar="DST=SRC")
    ap.add_argument("--series-every", type=int, default=0)
    ap.add_argument("--no-objectives", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--full-series", action="store_true")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--model", default="v1", choices=["v1", "v2"])
    ap.add_argument("--eval-kw", default=None,
                    help="opciones del evaluador de u25 en JSON (peak_time, o5_birth_ref, o4_version,"
                         " crowd_class_threshold); con --model v2 el default es v2_api.json evaluador.primario")
    a = ap.parse_args(argv)
    import u25.sweep as sw
    if a.full_series:
        from u25 import Params, simulate
        S0, _ = simulate(Params(n_males=10, n_females=10), T=3, seed=1)
        sw.SERIES_COLS[:] = list(S0.columns)
    else:
        extra = EXTRA_SERIES
        if a.model == "v2":
            from f2b_model_v2 import SERIES_V2
            extra = EXTRA_SERIES + SERIES_V2 + ["S_adult", "N_adult"]
        sw.SERIES_COLS[:] = list(dict.fromkeys(sw.SERIES_COLS + extra))
    mc = None
    if not a.no_extras:
        mc = model_class(a.model)
    ekw = json.loads(a.eval_kw) if a.eval_kw else None
    if ekw is None and a.model == "v2":
        from f2b_model_v2 import primary_kw
        ekw = primary_kw()
    if ekw:
        import u25.objectives as OBJ
        if hasattr(OBJ, "validate_eval_kw"):
            OBJ.validate_eval_kw(ekw)          # falla antes del barrido si la API no tiene alguna opción
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "eval_kw.json").write_text(json.dumps(ekw))
    r = sw.run_sweep(a.design, a.reps, a.out, T=a.T, base_seed=a.base_seed, n_jobs=a.n_jobs, crn=a.crn,
                     base=_base(a.preset, a.set), ties=sw.parse_ties(a.tie), series_every=a.series_every,
                     evaluate=not a.no_objectives, model_cls=mc, force=a.force, quiet=a.quiet,
                     log_every=a.log_every, **({"eval_kw": ekw} if ekw else {}))
    print(r, flush=True)
    return 0 if r["failed"] == 0 else 1


def _cross_one(task):
    """Transplante cruzado: snapshots de una madre con params `pm` y colonias nuevas con params `pc`.
    Con la misma base_seed que el brazo del candidato, las madres y los 8 trasladados son los mismos (pareado)."""
    import numpy as np
    from u25 import Params
    from u25.ensemble import rep_seed
    from u25.experiments import colony_snapshots, transplant_trial, mixed_trial
    from f2b_model import U2b
    pm, pc, rep, base_seed, T_main, T_after = task
    pm, pc = Params.from_dict(pm), Params.from_dict(pc)
    times, sB, sD, _ = colony_snapshots(pm, rep_seed(base_seed, 0, rep), T_main, U2b)
    # un flujo de números aleatorios por fase: la selección de los 8 trasladados de cada fase no depende de lo
    # que consumió la colonia de la fase anterior (pareado exacto entre brazos con la misma madre)
    rngs = {ph: np.random.default_rng(rep_seed(base_seed, 1 + i, rep)) for i, ph in enumerate(("B", "D", "MIX_M", "MIX_F"))}
    rows = []
    for phase, snap in (("B", sB), ("D", sD)):
        r = transplant_trial(pc, snap, rngs[phase], 4, T_after, 10.0, True, U2b)
        r.update(rep=rep, trial=0, phase=phase, t_snap=times[f"t_{phase}"], t_pk=times["t_pk"],
                 early_D=bool(times["early_D"]) if phase == "D" else False,
                 S_snap=float(snap["s"].mean()) if snap is not None and snap["N"] else np.nan)
        rows.append(r)
    for phase, dm in (("MIX_M", True), ("MIX_F", False)):
        r = mixed_trial(pc, sD, rngs[phase], dm, 4, T_after=T_after, factor=10.0, model_cls=U2b)
        r.update(rep=rep, trial=0, phase=phase, t_snap=times["t_D"], t_pk=times["t_pk"], early_D=bool(times["early_D"]),
                 S_snap=float(sD["s"].mean()) if sD is not None and sD["N"] else np.nan)
        rows.append(r)
    return rows


def cross_experiment(pm, pc, n_reps, base_seed, T_main, T_after, n_jobs, out):
    import multiprocessing as mp
    import pandas as pd
    from u25.objectives import transplant_status, mixed_transplant_status
    tasks = [(pm.to_dict(), pc.to_dict(), r, base_seed, T_main, T_after) for r in range(n_reps)]
    ctx = mp.get_context("fork")
    with ctx.Pool(n_jobs) as pool:
        res = pool.map(_cross_one, tasks, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows]).sort_values(["phase", "rep"]).reset_index(drop=True)
    o12 = transplant_status(df.loc[df.phase == "B", "success"], df.loc[df.phase == "D", "success"])
    o12b = mixed_transplant_status(df.loc[df.phase.str.startswith("MIX"), "success"])
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "transplant.csv"), index=False)
    with open(os.path.join(out, "params.json"), "w") as f:
        f.write(json.dumps({"madre": json.loads(pm.to_json()), "colonia": json.loads(pc.to_json())}, indent=1))
    with open(os.path.join(out, "O12.txt"), "w") as f:
        f.write(f"{o12.status}\t{o12.value}\t{o12.detail}\n")
        f.write(f"O12b\t{o12b.status}\t{o12b.value}\t{o12b.detail}\n")
    df.attrs.update(O12=o12, O12b=o12b)
    return df


def cmd_transplant(argv):
    ap = argparse.ArgumentParser(prog="u25x.py transplant")
    ap.add_argument("--design", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE")
    ap.add_argument("--T-main", type=int, default=300)
    ap.add_argument("--T-after", type=int, default=150)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args(argv)

    def _term(signum, frame):            # Pool como context manager: terminate() de los workers al salir
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    import time
    import pandas as pd
    from u25.sweep import load_design, design_to_params
    from u25.experiments import transplant_experiment
    from f2b_model import U2b
    ddf, prob, ddir = load_design(a.design)
    pmap = design_to_params(ddf, _base(a.preset, a.set), prob.get("fixed"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ddf.to_csv(out / "design.csv", index=False)
    if ddir is not None and (ddir / "labels.json").exists():
        (out / "labels.json").write_text((ddir / "labels.json").read_text())
    log = open(out / "progress.log", "a", buffering=1)
    cross = {}
    if ddir is not None and (ddir / "cross.json").exists():      # {punto: punto_madre}
        cross = {int(k): int(v) for k, v in json.load(open(ddir / "cross.json")).items()}
        (out / "cross.json").write_text(json.dumps(cross))
    n_fail = 0
    try:
        for pt in sorted(pmap):
            d = out / f"arm_{pt}"
            if (d / "O12.txt").exists():
                continue
            t0 = time.perf_counter()
            log.write(f"[{time.strftime('%H:%M:%S')}] brazo {pt}: {a.reps} réplicas\n")
            try:
                if pt in cross:      # madre del brazo cross[pt] (misma semilla: pareado), colonia con pmap[pt]
                    m = cross[pt]
                    df = cross_experiment(pmap[m], pmap[pt], a.reps, a.base_seed + 1000 * m, a.T_main, a.T_after,
                                          a.n_jobs, str(d))
                else:
                    df = transplant_experiment(pmap[pt], n_reps=a.reps, base_seed=a.base_seed + 1000 * pt,
                                               T_main=a.T_main, T_after=a.T_after, n_jobs=a.n_jobs, out=str(d),
                                               mixed=True, model_cls=U2b)
                log.write(f"   O12 {df.attrs['O12']}  O12b {df.attrs['O12b']}  "
                          f"({time.perf_counter() - t0:.0f} s)\n")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log.write(f"   ERROR {type(e).__name__}: {e}\n")
    except KeyboardInterrupt:
        log.write(f"[{time.strftime('%H:%M:%S')}] interrumpido; relanzar para retomar\n")
        return 143
    rows = []
    for pt in sorted(pmap):
        f = out / f"arm_{pt}" / "transplant.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(point=pt))
    if rows:
        pd.concat(rows).to_csv(out / "transplant.csv", index=False)
    json.dump({"reps": a.reps, "T_main": a.T_main, "T_after": a.T_after, "base_seed": a.base_seed,
               "n_points": len(pmap), "params": {str(k): json.loads(v.to_json()) for k, v in pmap.items()}},
              open(out / "meta.json", "w"), indent=1)
    log.write(f"[{time.strftime('%H:%M:%S')}] fin: {len(rows)}/{len(pmap)} brazos, {n_fail} fallas\n")
    print({"arms": len(rows), "total": len(pmap), "failed": n_fail}, flush=True)
    return 0 if n_fail == 0 else 1


def cmd_transplant2(argv):
    """Transplantes de la V2 (trans_v2.py): fases B, D, Dfix, Dtp, MIX*, BAND*; reanudable por brazo."""
    ap = argparse.ArgumentParser(prog="u25x.py transplant2")
    ap.add_argument("--design", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE")
    ap.add_argument("--T-main", type=int, default=300)
    ap.add_argument("--T-after", type=int, default=150)
    ap.add_argument("--t-fix", type=int, default=95)
    ap.add_argument("--band", default="0.2,0.5")
    ap.add_argument("--cont-after", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--model", default="v2", choices=["v1", "v2", "none"])
    a = ap.parse_args(argv)

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    import time
    import pandas as pd
    from u25.sweep import load_design, design_to_params
    import trans_v2
    ddf, prob, ddir = load_design(a.design)
    pmap = design_to_params(ddf, _base(a.preset, a.set), prob.get("fixed"))
    band = tuple(float(x) for x in a.band.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ddf.to_csv(out / "design.csv", index=False)
    if ddir is not None and (ddir / "labels.json").exists():
        (out / "labels.json").write_text((ddir / "labels.json").read_text())
    cross = {}
    if ddir is not None and (ddir / "cross.json").exists():      # {punto: punto_madre}
        cross = {int(k): int(v) for k, v in json.load(open(ddir / "cross.json")).items()}
        (out / "cross.json").write_text(json.dumps(cross))
    log = open(out / "progress.log", "a", buffering=1)
    n_fail = 0
    try:
        for pt in sorted(pmap):
            d = out / f"arm_{pt}"
            if (d / "O12.txt").exists():
                continue
            m = cross.get(pt, pt)              # punto cuya madre se usa (misma semilla: pareado)
            t0 = time.perf_counter()
            log.write(f"[{time.strftime('%H:%M:%S')}] brazo {pt} (madre {m}): {a.reps} réplicas\n")
            try:
                df = trans_v2.experiment(pmap[m], pmap[pt], a.reps, a.base_seed + 1000 * m, a.T_main, a.T_after,
                                         a.t_fix, band, a.cont_after, a.n_jobs, str(d), a.model)
                log.write(f"   O12 {df.attrs['O12']}  O12fix {df.attrs['O12fix']}  "
                          f"({time.perf_counter() - t0:.0f} s)\n")
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                import traceback
                log.write(f"   ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}\n")
    except KeyboardInterrupt:
        log.write(f"[{time.strftime('%H:%M:%S')}] interrumpido; relanzar para retomar\n")
        return 143
    rows = []
    for pt in sorted(pmap):
        f = out / f"arm_{pt}" / "transplant.csv"
        if f.exists():
            rows.append(pd.read_csv(f).assign(point=pt))
    if rows:
        pd.concat(rows).to_csv(out / "transplant.csv", index=False)
    json.dump({"reps": a.reps, "T_main": a.T_main, "T_after": a.T_after, "t_fix": a.t_fix, "band": list(band),
               "cont_after": a.cont_after, "base_seed": a.base_seed, "n_points": len(pmap), "cross": cross,
               "params": {str(k): json.loads(v.to_json()) for k, v in pmap.items()}},
              open(out / "meta.json", "w"), indent=1)
    log.write(f"[{time.strftime('%H:%M:%S')}] fin: {len(rows)}/{len(pmap)} brazos, {n_fail} fallas\n")
    print({"arms": len(rows), "total": len(pmap), "failed": n_fail}, flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    sys.exit({"sweep": cmd_sweep, "transplant": cmd_transplant, "transplant2": cmd_transplant2}[cmd](rest))
