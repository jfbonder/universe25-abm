"""Fase 8a: ensamble bien mezclado de C1 con el MISMO runner, clase de modelo y evaluador que el ensamble de
referencia de PRE (results_v2/f2b_base: u25.sweep + f2b_model_v2.U2v2, eval_kw primario, n = 200, T = 600).

    python3 scripts/em/em_wm_run.py [--reps 200] [--n-jobs 4]

Puntos (base = preset calhoun_C1, L = 30, N0 = 400, edad 4, s = 1):
  0  C1 well-mixed            jump_prob = 1, jump_mode = "uniform" (refugios del preset: sólo actúan como capacidad,
                              porque con saltos uniformes la preferencia pi y la aversión AV no tienen efecto)
  1  C1 well-mixed, no refuges  ídem con refuge_frac = 0 (como C0: refuge_capacity 1, pref 0, hard False)
Salida: results_em/wellmixed_c1/ (summary.csv con obj_*, series/ y series.parquet). Reanudable.
Semillas: SeedSequence([20261017, punto, réplica]) (u25.sweep, crn = False).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts" / "production"), str(ROOT / "scripts" / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd                       # noqa: E402
from u25.params import preset             # noqa: E402
from u25.sweep import run_sweep, collect_series   # noqa: E402
from f2b_model_v2 import U2v2             # noqa: E402

OUT = ROOT / "results_em" / "wellmixed_c1"
LABELS = ["C1 well-mixed", "C1 well-mixed, no refuges"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--T", type=int, default=600)
    ap.add_argument("--out", default=str(OUT), help="output directory (default results_em/wellmixed_c1)")
    a = ap.parse_args()
    eval_kw = json.load(open(ROOT / "results_v2" / "f2b_base" / "eval_kw.json"))
    design = pd.DataFrame([
        dict(point=0, jump_prob=1.0, jump_mode="uniform", refuge_frac=0.3, refuge_capacity=2, refuge_pref=100.0,
             refuge_hard="strict"),
        dict(point=1, jump_prob=1.0, jump_mode="uniform", refuge_frac=0.0, refuge_capacity=1, refuge_pref=0.0,
             refuge_hard=False),
    ])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(LABELS, open(out / "labels.json", "w"), indent=1)
    res = run_sweep(design, reps=a.reps, out=out, T=a.T, base_seed=20261017, n_jobs=a.n_jobs, crn=False,
                    base=preset("calhoun_C1"), series_every=1, evaluate=True, model_cls=U2v2, eval_kw=eval_kw)
    print(res)
    print(collect_series(out))


if __name__ == "__main__":
    main()
