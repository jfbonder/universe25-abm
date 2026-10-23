""" barridos reanudables del ABM (espacial y bien mezclado) para las tareas 1, 2, 3 y 6.

Uso (desde la raíz del repositorio): python3 scripts/em/f7_sweep.py <experimento> [--nproc 4] [--dry]
Cada corrida -> results_em/<grupo>/ (ver f7_common.group_dir)<config>/seed<k>.npz (se saltean las existentes).
Semillas: default_rng([20261005, crc32(clave de configuración), k]).

Experimentos (ver EXPERIMENTS):
  e1sp          C1 espacial (preset, con refugios): grilla (lambda, plast_outside_rec), L = 15/30/60
  e1wm_cell     C1 bien mezclado sin refugios, apareamiento en la celda (u25)
  e1wm_moore    ídem, apareamiento en el bloque de Moore (mate_radius = 1)
  e1wm_global   ídem, apareamiento global (padre uniforme en toda la arena)
  sl_mate       SL bien mezclado: celda / Moore / global, L = 10..90 (¿metaestable sin Allee de apareamiento?)
  hyd_sp, hyd_wm   retiro de una fracción f de adultos no reproductores (s < s_v) en la semana t_r
  hyd_op        ídem con la definición operativa (no crió en las últimas 8 semanas), espacial
  hyd_per       retiro periódico (cada 4 semanas desde t_r), espacial y b.m.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from f7_common import build, run_and_pack, save_npz, group_dir, ROOT

OUTS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
LAMS = (1.0, 0.95, 0.9, 0.8, 0.5, 0.0)


def _e1(group, wm, mate, Ls=((15, 20, 2000), (30, 20, 1000), (60, 10, 1000)), outs=OUTS, lams=LAMS, model="C1"):
    """Horizonte por tamaño: L = 15: 2000 semanas; L = 30: 1000 (las primeras corridas de L = 30 se hicieron con 2000);
    L = 60: 1000 y sólo lambda in {1, 0.9, 0.5, 0} (costo de las colonias persistentes)."""
    jobs = []
    for L, ns, T in Ls:
        for lam in (lams if L < 60 else ((1.0, 0.9, 0.5, 0.0) if wm else (1.0, 0.9, 0.0))):
            for out in (outs if L < 60 else (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)):
                cfg = f"{model}_{'wm' if wm else 'sp'}_{mate}_l{lam:g}_o{out:g}_L{L}"
                for k in range(ns):
                    jobs.append(dict(group=group, cfg=cfg, model=model, L=L, wm=wm, T=T, k=k,
                                     opts=dict(mate=mate), over=dict(social_rec_lambda=lam, plast_outside_rec=out)))
    return jobs


def _sl_global_big():
    """SL b.m. con apareamiento global a L = 120 (metaestabilidad con valle profundo)."""
    return [dict(group="f7_mate", cfg="SL_wm_global_L120", model="SL", L=120, wm=True, T=3000, k=k,
                 opts=dict(mate="global", obs=False), over={}) for k in range(10)]


def _sl_mate():
    jobs = []
    for mate in ("cell", "moore", "global"):
        for L, ns in ((10, 30), (15, 30), (20, 30), (30, 30), (45, 20), (60, 20), (90, 10)):
            cfg = f"SL_wm_{mate}_L{L}"
            for k in range(ns):
                jobs.append(dict(group="f7_mate", cfg=cfg, model="SL", L=L, wm=True, T=3000, k=k,
                                 opts=dict(mate=mate), over={}))
    return jobs


def _hyd(group, wm, trs, fs=(0.25, 0.5, 0.75, 0.9, 1.0), L=30, ns=20, rule="s", period=0, T=1500, extra=""):
    jobs = []
    for tr in trs:
        for f in fs:
            cfg = f"C1_{'wm' if wm else 'sp'}_{rule}{'_per%d' % period if period else ''}_tr{tr}_f{f:g}_L{L}{extra}"
            for k in range(ns):
                jobs.append(dict(group=group, cfg=cfg, model="C1", L=L, wm=wm, T=T, k=k,
                                 opts=dict(removal=dict(t_r=tr, f=f, rule=rule, period=period)), over={}))
    # control sin retiro
    cfg = f"C1_{'wm' if wm else 'sp'}_none_L{L}{extra}"
    for k in range(ns):
        jobs.append(dict(group=group, cfg=cfg, model="C1", L=L, wm=wm, T=T, k=k, opts={}, over={}))
    return jobs


def _hyd_stop():
    """Retiro periódico (cada 4 semanas desde t_r = 30) que se suspende en t_stop = t_r + 52, 104 o 208."""
    jobs = []
    for wm in (False, True):
        for f in (0.25, 0.5):
            for dur in (52, 104, 208):
                cfg = f"C1_{'wm' if wm else 'sp'}_s_per4_tr30_f{f:g}_stop{dur}_L30"
                for k in range(10):
                    jobs.append(dict(group="f7_hyd", cfg=cfg, model="C1", L=30, wm=wm, T=1500, k=k,
                                     opts=dict(removal=dict(t_r=30, f=f, rule="s", period=4, t_stop=30 + dur)), over={}))
    return jobs


def _mubg():
    """E5/P2: mortalidad de fondo mu_bg en el ABM (C1 exacto y con algo de recuperación adulta)."""
    jobs = []
    for wm in (False, True):
        for lam, out in ((1.0, 0.0), (0.0, 0.0), (0.9, 0.3)):
            for mu in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                cfg = f"C1_{'wm' if wm else 'sp'}_mubg{mu:g}_l{lam:g}_o{out:g}_L30"
                for k in range(10):
                    jobs.append(dict(group="f7_mubg", cfg=cfg, model="C1", L=30, wm=wm, T=1000, k=k, opts={},
                                     over=dict(mu_bg=mu, social_rec_lambda=lam, plast_outside_rec=out)))
    return jobs


def _p3():
    """P3: separar theta* (demostradores) de nu* (hacinamiento de los juveniles). Durante [t1, t1 + w]
    los juveniles en la ventana tienen demostradores perfectos (theta1) o no sufren daño (nodamage)."""
    jobs = []
    for wm in (False, True):
        for kind in ("theta1", "nodamage"):
            for t1 in (15, 20, 25, 30, 40):
                for w in (12, 26, 52, 2000):
                    cfg = f"C1_{'wm' if wm else 'sp'}_{kind}_t{t1}_w{w}_L30"
                    for k in range(10):
                        jobs.append(dict(group="f7_p3", cfg=cfg, model="C1", L=30, wm=wm, T=1500, k=k,
                                         opts=dict(manip=dict(kind=kind, t1=t1, t2=t1 + w)), over={}))
    return jobs


EXPERIMENTS = {
    "p3": _p3,
    "mubg": _mubg,
    "hyd_stop": _hyd_stop,
    "sl_big": _sl_global_big,
    "e1sp": lambda: _e1("f7_e1", False, "cell"),
    "e1wm_cell": lambda: _e1("f7_e1", True, "cell"),
    "e1wm_moore": lambda: _e1("f7_e1", True, "moore"),
    "e1wm_global": lambda: _e1("f7_e1", True, "global"),
    "sl_mate": _sl_mate,
    "hyd_sp": lambda: _hyd("f7_hyd", False, (20, 25, 30, 35, 40, 45, 50, 60, 80, 100)),
    "hyd_wm": lambda: _hyd("f7_hyd", True, (15, 20, 24, 27, 30, 33, 36, 40, 50)),
    "hyd_op": lambda: _hyd("f7_hyd", False, (30, 40, 50, 60), fs=(0.5, 0.9, 1.0), rule="op"),
    "hyd_per": lambda: _hyd("f7_hyd", False, (30, 40, 50), fs=(0.25, 0.5), period=4)
                       + _hyd("f7_hyd", True, (20, 25, 30), fs=(0.25, 0.5), period=4),
}


def outpath(job):
    d = os.path.join(group_dir(job["group"]), job["cfg"])
    return d, os.path.join(d, f"seed{job['k']}.npz")


def run_job(job):
    d, fn = outpath(job)
    if os.path.exists(fn):
        return None
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    U = build(job["model"], job["L"], job["wm"], f"{job['group']}|{job['cfg']}", job["k"], T=job["T"],
              opts=job["opts"], **job["over"])
    out = run_and_pack(U)
    out["wall"] = time.time() - t0
    out["L"] = job["L"]
    save_npz(fn, out)
    return (job["cfg"], job["k"], out["extinct"], out["t_ext"], out["t_last"], out["N_max"], round(out["wall"], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="L descendente (para correr en paralelo con otra cola)")
    a = ap.parse_args()
    jobs = []
    for e in a.exps:
        jobs += EXPERIMENTS[e]()
    todo = [j for j in jobs if not os.path.exists(outpath(j)[1])]
    # primero las grillas chicas (L ascendente) para tener resultados parciales útiles
    todo.sort(key=lambda j: (j["L"], j["k"]), reverse=a.reverse)
    print(f"{len(jobs)} trabajos, {len(todo)} pendientes", flush=True)
    if a.dry:
        return
    with Pool(a.nproc) as pool:
        for info in pool.imap_unordered(run_job, todo, chunksize=1):
            if info is not None:
                print(*info, flush=True)


if __name__ == "__main__":
    main()
