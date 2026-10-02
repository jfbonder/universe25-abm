"""u25: modelo de agentes mínimo para Universe 25 (Calhoun).

Uso rápido:
    from u25 import Params, simulate, run_ensemble
    series, summary = simulate(Params(alpha=1.4), T=2000, seed=1)
    df = run_ensemble(Params(), n_reps=20, T=2000, base_seed=0, n_jobs=4, out="results/baseline")
"""
from .params import Params, preset, PRESETS
from .model import Universe, AffinityStore, simulate
from .ensemble import run_ensemble, run_one

__all__ = ["Params", "preset", "PRESETS", "Universe", "AffinityStore", "simulate", "run_ensemble", "run_one"]
