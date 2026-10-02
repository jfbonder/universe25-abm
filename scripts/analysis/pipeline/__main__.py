"""CLI del pipeline.

  python -m pipeline design {morris|lhs|sobol} --out DIR [--problem problem.json] [--r 20 | --n 256 | --N 256] [--seed 0]
  python -m pipeline run DESIGN_DIR --n-reps 5 --T 1000 [--out DIR] [--n-jobs 4] [--series]
  python -m pipeline baseline RUN_DIR --out DIR [--t-split 200] [--B-gof 300]
  python -m pipeline analyze {morris|sobol} DESIGN_DIR [--run RUN_DIR] --out DIR
  python -m pipeline f2b STAGE RUN_DIR [--out DIR]            # Fase 2b (docs/preregistered_plan.md), por etapa
  python -m pipeline f2b final RESULTS_DIR [--suffix S] [--out DIR]   # Q1–Q7 con Holm, exploratorios con BH
  python -m pipeline v2 STAGE RUN_DIR [--out DIR]             # V2: etapas nuevas
  python -m pipeline v2 final RESULTS_DIR [--suffix S]        # V2: nulos, re-puntuaciones, región, Q6, dispersión y ensambles
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design"); d.add_argument("kind", choices=["morris", "lhs", "sobol"]); d.add_argument("--out", required=True)
    d.add_argument("--problem", default=None); d.add_argument("--r", type=int, default=20); d.add_argument("--n", type=int, default=256)
    d.add_argument("--N", type=int, default=256); d.add_argument("--levels", type=int, default=4); d.add_argument("--seed", type=int, default=0)
    d.add_argument("--second-order", action="store_true")
    r = sub.add_parser("run"); r.add_argument("design_dir"); r.add_argument("--n-reps", type=int, default=5); r.add_argument("--T", type=int, default=1000)
    r.add_argument("--out", default=None); r.add_argument("--n-jobs", type=int, default=4); r.add_argument("--base-seed", type=int, default=0)
    r.add_argument("--series", action="store_true"); r.add_argument("--record-every", type=int, default=1); r.add_argument("--no-crn", action="store_true")
    b = sub.add_parser("baseline"); b.add_argument("run_dir"); b.add_argument("--out", required=True); b.add_argument("--t-split", type=float, default=None)
    b.add_argument("--B-gof", type=int, default=300); b.add_argument("--min-peaks", type=int, default=4); b.add_argument("--prominence", type=float, default=0.10)
    a = sub.add_parser("analyze"); a.add_argument("kind", choices=["morris", "sobol"]); a.add_argument("design_dir"); a.add_argument("--run", default=None)
    a.add_argument("--out", required=True)
    f = sub.add_parser("f2b"); f.add_argument("stage"); f.add_argument("run_dir"); f.add_argument("--out", default=None)
    f.add_argument("--suffix", default="")
    v = sub.add_parser("v2"); v.add_argument("stage"); v.add_argument("run_dir"); v.add_argument("--out", default=None)
    v.add_argument("--suffix", default="")
    args = ap.parse_args(argv)
    if args.cmd == "v2":
        from . import v2
        if args.stage == "final":
            print(v2.final_v2(args.run_dir, sfx=args.suffix))
        else:
            print(v2.analyze_stage_v2(args.stage, args.run_dir, args.out))
        return 0
    if args.cmd == "f2b":
        from . import f2b
        if args.stage == "final":
            print(f2b.final(args.run_dir, sfx=args.suffix, out=args.out))
        else:
            print(f2b.analyze_stage(args.stage, args.run_dir, args.out))
        return 0
    from . import design, report
    if args.cmd == "design":
        problem = json.load(open(args.problem)) if args.problem else design.PROBLEMA_DEFAULT
        if args.kind == "morris":
            X, df = design.morris_design(problem, r=args.r, num_levels=args.levels, seed=args.seed)
            design.save_design(args.out, "morris", problem, X, df, r=args.r, num_levels=args.levels, seed=args.seed)
        elif args.kind == "lhs":
            X, df = design.lhs_design(problem, n=args.n, seed=args.seed)
            design.save_design(args.out, "lhs", problem, X, df, n=args.n, seed=args.seed)
        else:
            X, df = design.saltelli_design(problem, N=args.N, calc_second_order=args.second_order, seed=args.seed)
            design.save_design(args.out, "sobol", problem, X, df, N=args.N, calc_second_order=args.second_order, seed=args.seed)
        print(f"diseño {args.kind}: {len(df)} puntos -> {args.out}/design.csv")
    elif args.cmd == "run":
        summ = design.run_design(args.design_dir, n_reps=args.n_reps, T=args.T, out=args.out, n_jobs=args.n_jobs,
                                 base_seed=args.base_seed, crn=not args.no_crn, save_series=args.series,
                                 record_every=args.record_every, progress=True)
        print(f"{len(summ)} corridas, pared {summ.attrs.get('wall_time_s', float('nan')):.0f} s")
    elif args.cmd == "baseline":
        sv, lines = report.baseline_report(args.run_dir, args.out, t_split=args.t_split, B_gof=args.B_gof,
                                           min_peaks=args.min_peaks, prominence_rel=args.prominence)
        print("\n".join(lines))
    else:
        run_dir = args.run or args.design_dir
        if args.kind == "morris":
            print(report.morris_report(args.design_dir, run_dir, args.out).to_string())
        else:
            print(report.sobol_report(args.design_dir, run_dir, args.out).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
