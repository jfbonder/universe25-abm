"""Fase 8a: mapas de una corrida (semilla 1) para el panel de instantáneas de fig_em_wellmixed.

    python3 scripts/em/em_snapshots.py

Corre C1 espacial (preset calhoun_C1) y C1 bien mezclado con los parámetros del punto 0 de em_wm_run.py
(jump_prob = 1, jump_mode = "uniform", sitios de retiro como capacidad), L = 30, T = 600, con
rng = default_rng(SeedSequence([20261017, 1])). Guarda, para cada semana, la ocupación por celda y el número de
adultos retirados (edad >= repro_age_min, s < obs_s_low = 0.2, iso_run >= obs_iso_weeks = 4, la condición phi_BO del
paquete), más la máscara de sitios de retiro, en results_em/wellmixed_c1/snapshots_seed1.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                     # noqa: E402
from u25.model import Universe         # noqa: E402
from u25.params import preset          # noqa: E402

OUT = ROOT / "results_em" / "wellmixed_c1" / "snapshots_seed1.npz"


def run(p, seed=1):
    U = Universe(p, rng=np.random.default_rng(np.random.SeedSequence([20261017, seed])))
    G = p.grid_size
    occ, wd, Ns = [], [], []

    def cb(U):
        cell = U.x * G + U.y
        occ.append(np.bincount(cell, minlength=G * G).astype(np.int16))
        bo = (U.age >= p.repro_age_min) & (U.s < p.obs_s_low) & (U.iso_run >= p.obs_iso_weeks)
        wd.append(np.bincount(cell[bo], minlength=G * G).astype(np.int16))
        Ns.append(U.N)
    U.run(p.T, callback=cb)
    ref = U.refuge if U.refuge is not None else np.zeros(G * G, bool)
    return np.array(occ), np.array(wd), np.array(Ns), ref


def main():
    base = preset("calhoun_C1").replace(T=600)
    wm = base.replace(jump_prob=1.0, jump_mode="uniform")
    d = {}
    for name, p in (("sp", base), ("wm", wm)):
        occ, wd, N, ref = run(p)
        d[f"{name}_occ"], d[f"{name}_wd"], d[f"{name}_N"], d[f"{name}_ref"] = occ, wd, N, ref
        print(name, "N_max", N.max(), "t_max", int(np.argmax(N)) + 1, "T_ext", len(N))
    np.savez_compressed(OUT, G=base.grid_size, **d)
    print(OUT)


if __name__ == "__main__":
    main()
