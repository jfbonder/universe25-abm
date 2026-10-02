"""CLI:  python -m u25 {run,ensemble} ...

Ejemplos
  python -m u25 run --T 500 --seed 3 --set alpha=1.4
  python -m u25 ensemble --n-reps 20 --T 2000 --base-seed 0 --n-jobs 4 --out results/baseline
  python -m u25 ensemble --n-reps 10 --T 1000 --sweep alpha=0,0.5,1,1.5,2 --out results/alpha_sweep
  python -m u25 ensemble --preset longevo --n-reps 30 --T 1000 --out results/longevo
  python -m u25 objectives results/longevo [--transplant results/longevo_tr/transplant.csv]
  python -m u25 transplant --preset longevo --n-reps 30 --out results/longevo_tr
  nohup python -m u25 sweep --design results/lhs_X/design.csv --reps 30 --T 1000 --out results/lhs_X_run \
        --preset longevo --tie n_females=n_males --series-every 5 --crn --quiet &
  python -m u25 collect results/lhs_X_run       # series/ -> series.parquet
  python -m u25 reeval results/f2b_base --out results_v2/reeval_v1/f2b_base   # re-evaluación (v1 intacto)
"""
from __future__ import annotations

import argparse
import itertools
import sys

from .params import Params, parse_value


def _params_from_sets(sets, preset_name=None):
    from .params import preset
    p = preset(preset_name) if preset_name else Params()
    kw = {}
    for item in sets or []:
        k, v = item.split("=", 1)
        kw[k] = parse_value(p, k, v)
    return p.replace(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m u25", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="una realización")
    r.add_argument("--T", type=int, default=None)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--set", action="append", metavar="KEY=VALUE")
    r.add_argument("--out", default=None, help="CSV de la serie temporal")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--preset", default=None)

    e = sub.add_parser("ensemble", help="ensamble en paralelo")
    e.add_argument("--n-reps", type=int, default=10)
    e.add_argument("--T", type=int, default=2000)
    e.add_argument("--base-seed", type=int, default=0)
    e.add_argument("--n-jobs", type=int, default=4)
    e.add_argument("--set", action="append", metavar="KEY=VALUE")
    e.add_argument("--sweep", action="append", metavar="KEY=v1,v2,...",
                   help="producto cartesiano de valores (puede repetirse)")
    e.add_argument("--out", required=True)
    e.add_argument("--no-series", action="store_true")
    e.add_argument("--record-every", type=int, default=1)
    e.add_argument("--crn", action="store_true", help="números aleatorios comunes entre puntos")
    e.add_argument("--quiet", action="store_true")
    e.add_argument("--preset", default=None, help="notebook | longevo | calhoun_candidato")

    o = sub.add_parser("objectives", help="evalúa O1–O17/A1–A4 sobre un directorio de ensamble")
    o.add_argument("dir")
    o.add_argument("--transplant", default=None, help="transplant.csv (para O12)")
    o.add_argument("--control-dir", default=None, help="ensamble de control con alpha=0 (O14 G_B)")

    tr = sub.add_parser("transplant", help="experimento de transplante (O12)")
    tr.add_argument("--n-reps", type=int, default=30)
    tr.add_argument("--base-seed", type=int, default=0)
    tr.add_argument("--T-main", type=int, default=400)
    tr.add_argument("--T-after", type=int, default=150)
    tr.add_argument("--n-jobs", type=int, default=4)
    tr.add_argument("--set", action="append", metavar="KEY=VALUE")
    tr.add_argument("--preset", default=None)
    tr.add_argument("--out", required=True)

    sw = sub.add_parser("sweep", help="barrido de producción reanudable (ver u25/sweep.py)")
    sw.add_argument("--design", required=True, help="design.csv (o directorio con design.csv/problem.json)")
    sw.add_argument("--reps", type=int, required=True)
    sw.add_argument("--T", type=int, default=1000)
    sw.add_argument("--out", required=True)
    sw.add_argument("--base-seed", type=int, default=0)
    sw.add_argument("--n-jobs", type=int, default=4)
    sw.add_argument("--crn", action="store_true", help="números aleatorios comunes entre puntos")
    sw.add_argument("--preset", default=None)
    sw.add_argument("--set", action="append", metavar="KEY=VALUE", help="base (el diseño tiene prioridad)")
    sw.add_argument("--tie", action="append", metavar="DST=SRC", help="p.ej. n_females=n_males")
    sw.add_argument("--series-every", type=int, default=0, help="0 = no guardar series; k = cada k pasos")
    sw.add_argument("--no-objectives", action="store_true")
    sw.add_argument("--force", action="store_true", help="ignorar incompatibilidad con sweep.json")
    sw.add_argument("--quiet", action="store_true", help="sólo al progress.log")
    sw.add_argument("--log-every", type=int, default=1)
    sw.add_argument("--peak-time", default=None, help="argmax | plateau_start | plateau_end | plateau_mid")
    sw.add_argument("--crowd-class-threshold", type=int, default=None, help="umbral fijo m > k de la clase hacinada")
    sw.add_argument("--o5-birth-ref", default=None, choices=["last", "B99"], help="fin de la natalidad en O5")
    sw.add_argument("--o4-version", default=None, choices=["v1", "v2"], help="O4 v2: además f_0 >= 0.10")
    sw.add_argument("--o11-age", default=None, choices=["mature6", "aw"], help="O11: s a la edad 6 o al cierre de a_w")

    co = sub.add_parser("collect", help="concatena <out>/series/*.parquet en <out>/series.parquet")
    co.add_argument("out")

    rv = sub.add_parser("reeval", help="re-evalúa un barrido con el evaluador vigente (ver u25/reeval.py)")
    rv.add_argument("src", nargs="+", help="directorio(s) de barrido de origen (no se modifican)")
    rv.add_argument("--out", required=True, help="directorio de salida (con varios src: raíz; un subdirectorio por src)")
    rv.add_argument("--n-jobs", type=int, default=4)
    rv.add_argument("--legacy-r7", choices=["auto", "yes", "no"], default="auto")
    rv.add_argument("--model-cls", default="auto", help="auto (el de sweep.json) | modulo.Clase | none")
    rv.add_argument("--from-series", action="store_true", help="re-evaluar desde las series, sin simular")
    rv.add_argument("--peak-time", default=None, help="argmax | plateau_start | plateau_end | plateau_mid")
    rv.add_argument("--crowd-class-threshold", type=int, default=None, help="umbral fijo m > k de la clase hacinada")
    rv.add_argument("--o5-birth-ref", default=None, choices=["last", "B99"], help="fin de la natalidad en O5")
    rv.add_argument("--o4-version", default=None, choices=["v1", "v2"], help="O4 v2: además f_0 >= 0.10")
    rv.add_argument("--o11-age", default=None, choices=["mature6", "aw"], help="O11: s a la edad 6 o al cierre de a_w")
    rv.add_argument("--series-every", type=int, default=0, help="guardar series compactas (re-simulación)")
    rv.add_argument("--quiet", action="store_true")

    a = ap.parse_args(argv)

    def _eval_kw():
        kw = {}
        for k in ("peak_time", "crowd_class_threshold", "o5_birth_ref", "o4_version", "o11_age"):
            v = getattr(a, k, None)
            if v is not None:
                kw[k] = v
        return kw or None

    if a.cmd == "sweep":
        from .sweep import run_sweep, parse_ties
        base = _params_from_sets(a.set, a.preset)
        r = run_sweep(a.design, a.reps, a.out, T=a.T, base_seed=a.base_seed, n_jobs=a.n_jobs, crn=a.crn,
                      base=base, ties=parse_ties(a.tie), series_every=a.series_every,
                      evaluate=not a.no_objectives, force=a.force, quiet=a.quiet, log_every=a.log_every,
                      eval_kw=_eval_kw())
        print(r)
        return 0 if r["failed"] == 0 else 1
    if a.cmd == "reeval":
        from pathlib import Path
        from .reeval import reeval_dir
        kw = _eval_kw() or {}
        many = len(a.src) > 1
        for src in a.src:
            out = Path(a.out) / Path(src).name if many else Path(a.out)
            info = reeval_dir(src, out, n_jobs=a.n_jobs, legacy_r7=a.legacy_r7, model_cls=a.model_cls,
                              from_series=a.from_series, eval_kw=kw or None, quiet=a.quiet,
                              series_every=a.series_every)
            chk = info["check"]
            print(f"{src} -> {out}: {chk['n_rows']} filas; columnas no-objetivo distintas: "
                  f"{chk['differing_non_objective_columns'] or 'ninguna'}; cambios en obj_*: "
                  f"{ {k: v for k, v in chk['objective_status_changes'].items()} or 'ninguno'}", flush=True)
        return 0
    if a.cmd == "collect":
        from .sweep import collect_series
        print(collect_series(a.out))
        return 0
    if a.cmd == "objectives":
        import json
        from pathlib import Path
        import pandas as pd
        from .objectives import evaluate_ensemble
        d = Path(a.dir)
        summ = pd.read_csv(d / "summary.csv")
        ser = pd.read_parquet(d / "series.parquet") if (d / "series.parquet").exists() \
            else pd.read_csv(d / "series.csv.gz")
        plist = [Params.from_dict(x) for x in json.load(open(d / "params.json"))]
        trdf = pd.read_csv(a.transplant) if a.transplant else None
        cg = None
        if a.control_dir:
            from .objectives import evaluate_ensemble as ev
            cd = Path(a.control_dir)
            cser = pd.read_parquet(cd / "series.parquet") if (cd / "series.parquet").exists() \
                else pd.read_csv(cd / "series.csv.gz")
            cpl = [Params.from_dict(x) for x in json.load(open(cd / "params.json"))]
            _, cper = ev(cser, pd.read_csv(cd / "summary.csv"), cpl)
            cg = cper["G_B_val"].to_numpy()
        tab, per = evaluate_ensemble(ser, summ, plist, trdf, cg)
        tab.to_csv(d / "objectives.csv", index=False)
        per.to_csv(d / "objectives_per_run.csv", index=False)
        print(tab.pivot(index="code", columns="point", values="verdict").to_string())
        print(f"-> {d / 'objectives.csv'}, {d / 'objectives_per_run.csv'}")
        return 0

    if a.cmd == "transplant":
        from .experiments import transplant_experiment
        p = _params_from_sets(a.set, a.preset)
        df = transplant_experiment(p, n_reps=a.n_reps, base_seed=a.base_seed, T_main=a.T_main,
                                   T_after=a.T_after, n_jobs=a.n_jobs, out=a.out)
        print(df.groupby("phase")[["success", "t_snap", "S_snap", "s_mean"]].mean().to_string())
        print("O12:", df.attrs["O12"])
        return 0

    if a.cmd == "run":
        from .model import simulate
        p = _params_from_sets(a.set, a.preset)
        series, summ = simulate(p, T=a.T, seed=a.seed, verbose=a.verbose)
        for k, v in summ.items():
            print(f"{k:>20s}: {v}")
        if a.out:
            series.to_csv(a.out, index=False)
        return 0

    from .ensemble import run_ensemble
    base = _params_from_sets(a.set, a.preset)
    plist = [base]
    if a.sweep:
        axes = []
        for item in a.sweep:
            k, vals = item.split("=", 1)
            axes.append([(k, parse_value(base, k, v)) for v in vals.split(",")])
        plist = [base.replace(**dict(combo)) for combo in itertools.product(*axes)]
    df = run_ensemble(plist, n_reps=a.n_reps, T=a.T, base_seed=a.base_seed, n_jobs=a.n_jobs,
                      out=a.out, save_series=not a.no_series, record_every=a.record_every,
                      crn=a.crn, progress=not a.quiet)
    cols = [c for c in ["point", "alpha", "N_max", "t_peak", "t_ext", "censored", "S_min",
                        "max_aff", "wall_time_s"] if c in df.columns]
    print(df[cols].to_string())
    print(f"total wall: {df.attrs.get('wall_time_s', float('nan')):.1f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
