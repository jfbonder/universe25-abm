"""Runner de barridos de producción, reanudable y seguro frente a interrupciones.

    python -m u25 sweep --design DIR_O_CSV --reps N --T 1000 --out results/<nombre> [--n-jobs 4]
    nohup python -m u25 sweep ... > /dev/null 2>&1 &      # el progreso va a <out>/progress.log

Diseño
------
`--design` es un CSV con una columna opcional `point` y una columna por parámetro de `Params`
(escala física), por ejemplo el `design.csv` de `scripts/analysis/pipeline` (`python -m pipeline design`).
Si junto al CSV hay un `problem.json` del pipeline, se aplican sus `fixed` y se copian `problem.json` y
`X.npy` a `--out`, de modo que `python -m pipeline analyze {morris,sobol} <out> --out ...` lee la salida
directamente. Base: `--preset` y `--set k=v` (los valores del diseño tienen prioridad). `--tie
n_females=n_males` liga parámetros (se aplica después del diseño).

Robustez
--------
* Cada (punto, réplica) se guarda apenas termina: el proceso padre (único escritor) agrega una línea
  completa a `summary.csv` y hace flush + fsync; las series compactas (opcionales) las escribe cada worker
  en `series/p<punto>_r<rep>.parquet` de forma atómica (archivo temporal + rename) **antes** de que la fila
  aparezca en el summary.
* Al relanzar se leen las filas completas de `summary.csv` (una última línea truncada se descarta) y se
  saltean los (punto, réplica) hechos. Se pueden subir `--reps` para extender un barrido.
* `sweep.json` guarda la configuración (hash del diseño, T, base_seed, crn, ...). Si se relanza con una
  configuración incompatible se aborta (salvo `--force`).
* Un lock (`fcntl.flock` sobre `<out>/.lock`) impide dos runners simultáneos sobre el mismo directorio.
* SIGTERM/SIGINT: se termina el pool; lo ya escrito queda. Nada se pierde salvo las corridas en curso.

Semillas: `SeedSequence([base_seed, point, rep])` (o `[base_seed, rep]` con `--crn`, números aleatorios
comunes entre puntos, que es el default del pipeline estadístico). Dependen sólo de (punto, réplica):
el resultado no depende del orden, de n_jobs ni de las interrupciones.

Salida en `<out>/`
------------------
`summary.csv`  una fila por corrida: point, <columnas del diseño>, rep, base_seed, <métricas de summarize>,
               obj_O1..obj_O16, obj_A1..obj_A4, obj_O17_run (= corrida "tipo Calhoun"), obj_G_B, wall_time_s.
               (las columnas de parámetros van antes de N_max, como espera pipeline.io.design_columns)
`params.json`  lista de Params por punto (índice = point); `meta.json` (T, n_reps, design_columns, ...);
`design.csv`   copia del diseño; `progress.log`; `series/` (si --series-every); `failures.csv` si hubo errores.
`python -m u25 collect <out>` concatena las series en `series.parquet` (formato de run_ensemble).
"""
from __future__ import annotations

import csv
import hashlib
import io as _io
import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Params, preset as _preset
from .model import Universe
from .ensemble import rep_seed

SERIES_COLS = ["t", "N", "N_m", "N_f", "births", "deaths", "conceptions", "S_mean", "S_adult", "s_q10",
               "s_q50", "s_q90", "h", "ref_frac", "N_repro_f", "mean_age", "age_min", "occ_max", "occ_var",
               "mstar", "f_over", "frac_withdrawn", "frac_crowded_low", "corr_s_m", "n_aff", "s_mat",
               "n_mat", "m_autocorr1", "phi_BO", "N_adult", "frac_in_refuge", "S_in_refuge",
               "coh_f_born", "coh_f_ever_conceived", "phi_w",
               # v2: clase hacinada con umbral fijo (re-puntuación de O13/O13p sin simular) y contadores de R7
               "frac_crowded_low_m6", "frac_crowded_low_m8", "frac_crowded_low_m10", "frac_crowded_low_m12",
               "n_refuge_rejected", "n_births_relocated",
               # v3: s al cierre de la ventana R1 (O11 con o11_age="aw")
               "s_aw", "n_aw"]
U25_VERSION = 2       # se guarda en meta.json/sweep.json; v1 = commit 253b3fc (refuge_hard=True era 'legacy')
_FIELD = {f.name: f for f in fields(Params)}


# ---------------------------------------------------------------- diseño -> Params
def _cast(base: Params, k: str, v):
    if k not in _FIELD:
        raise KeyError(f"columna de diseño desconocida (no es campo de Params): {k}")
    cur = getattr(base, k)
    if k == "refuge_hard":            # bool o 'strict' / 'legacy' / 'off' (Params.refuge_hard_mode valida)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return cur
        sv = str(v).strip()
        return {"true": True, "1": True, "false": False, "0": False}.get(sv.lower(), sv)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None if cur is None else (math.inf if isinstance(cur, float) else cur)
    if isinstance(cur, bool):
        return str(v).strip().lower() in ("1", "true", "yes") if isinstance(v, str) else bool(v)
    if isinstance(cur, int):
        return int(round(float(v)))
    if isinstance(cur, float) or cur is None:
        return float(v)
    return v


def load_design(path) -> tuple[pd.DataFrame, dict, Path | None]:
    """Devuelve (design_df con columna point, problem dict o {}, directorio del diseño)."""
    p = Path(path)
    csvp = p / "design.csv" if p.is_dir() else p
    df = pd.read_csv(csvp)
    if "point" not in df:
        df.insert(0, "point", np.arange(len(df)))
    if df["point"].duplicated().any():
        raise ValueError("puntos duplicados en el diseño")
    prob = {}
    pj = csvp.parent / "problem.json"
    if pj.exists():
        info = json.load(open(pj))
        prob = info.get("problem", info)
    return df, prob, csvp.parent


def design_to_params(df: pd.DataFrame, base: Params, fixed: dict | None = None,
                     ties: dict | None = None) -> dict[int, Params]:
    """{point: Params}. Orden de prioridad: base < fixed < diseño < ties (k=v: k toma el valor de v)."""
    if fixed:
        base = base.replace(**{k: _cast(base, k, v) for k, v in fixed.items()})
    out = {}
    cols = [c for c in df.columns if c != "point"]
    for _, row in df.iterrows():
        kw = {k: _cast(base, k, row[k]) for k in cols}
        p = base.replace(**kw)
        for k, src in (ties or {}).items():
            p = p.replace(**{k: getattr(p, src)})
        p.validate()
        out[int(row["point"])] = p
    return out


def parse_ties(items) -> dict:
    ties = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in _FIELD or v not in _FIELD:
            raise KeyError(f"--tie inválido: {it}")
        ties[k] = v
    return ties


# ---------------------------------------------------------------- worker
def _worker(task):
    (point, rep, pdict, T, base_seed, crn, series_dir, series_every, evaluate, model_cls, eval_kw) = task
    t0 = time.perf_counter()
    try:
        p = Params.from_dict(pdict)
        U = (model_cls or Universe)(p, rng=np.random.default_rng(rep_seed(base_seed, point, rep, crn)))
        series, summ = U.run(T)
        row = {"point": point, "rep": rep, "base_seed": base_seed, **summ}
        if evaluate:
            from .objectives import evaluate_run
            res = evaluate_run(series, p, summ, **(eval_kw or {}))
            for k, v in res.items():
                if k == "_times":
                    continue
                name = "obj_O17_run" if k == "calhoun_like" else f"obj_{k}"
                row[name] = v.value if k in ("G_B",) else v.status
                if k == "R_B":
                    row["obj_R_B_val"] = v.value
        if series_dir is not None:
            cols = [c for c in SERIES_COLS if c in series.columns]
            S = series[cols]
            if series_every > 1:
                keep = (S["t"] % series_every == 0) | (S.index == len(S) - 1)
                S = S[keep]
            S = S.assign(point=point, rep=rep)
            fn = Path(series_dir) / f"p{point}_r{rep}.parquet"
            tmp = fn.with_suffix(f".tmp{os.getpid()}")
            S.to_parquet(tmp, index=False)
            os.replace(tmp, fn)
        row["wall_time_s"] = time.perf_counter() - t0
        return ("ok", point, rep, row)
    except Exception:
        return ("err", point, rep, traceback.format_exc())


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # el padre maneja Ctrl-C y termina el pool
    signal.signal(signal.SIGTERM, signal.SIG_DFL)     # pool.terminate() debe matar al worker


# ---------------------------------------------------------------- summary append-only
def read_done(summary_path: Path):
    """Lee las filas completas de summary.csv. Devuelve (header | None, set de (point, rep)).
    Si la última línea está truncada (sin '\\n'), se recorta el archivo."""
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return None, set()
    raw = summary_path.read_bytes()
    if not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with open(summary_path, "r+b") as f:
            f.truncate(cut)
        raw = raw[:cut]
    text = raw.decode()
    rd = csv.reader(_io.StringIO(text))
    header = next(rd, None)
    if header is None:
        return None, set()
    ip, ir = header.index("point"), header.index("rep")
    done = set()
    for r in rd:
        if len(r) == len(header):
            done.add((int(r[ip]), int(r[ir])))
    return header, done


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "nan" if math.isnan(v) else repr(float(v))
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.bool_,)):
        return str(bool(v))
    return str(v)


class _Log:
    def __init__(self, path: Path, quiet: bool):
        self.f = open(path, "a", buffering=1)
        self.quiet = quiet

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.f.write(line + "\n")
        self.f.flush()
        if not self.quiet:
            print(line, flush=True)


def _hash_design(pmap: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(pmap):
        h.update(f"{k}:{pmap[k].to_json()}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- runner
def run_sweep(design, reps: int, out, T: int = 1000, base_seed: int = 0, n_jobs: int = 4,
              crn: bool = False, base: Params | None = None, ties: dict | None = None,
              series_every: int = 0, evaluate: bool = True, model_cls=None, force: bool = False,
              quiet: bool = False, max_runs: int | None = None, log_every: int = 1,
              params_map: dict | None = None, eval_kw: dict | None = None) -> dict:
    """Corre (o retoma) el barrido. `design`: ruta a CSV/directorio o DataFrame. Devuelve un dict con
    total, done, new, failed. `max_runs` limita las corridas nuevas de esta invocación (tests).
    `params_map` ({point: Params}) reemplaza la construcción desde el diseño (re-evaluación, u25.reeval);
    `eval_kw` se pasa a objectives.evaluate_run (peak_time, crowd_class_threshold, o5_birth_ref, o4_version) y
    queda en sweep.json y meta.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lockf = open(out / ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"otro runner está usando {out} (lock)")
    log = _Log(out / "progress.log", quiet)
    try:
        if isinstance(design, pd.DataFrame):
            ddf, prob, ddir = design.copy(), {}, None
            if "point" not in ddf:
                ddf.insert(0, "point", np.arange(len(ddf)))
        else:
            ddf, prob, ddir = load_design(design)
        base = base or Params()
        pmap = params_map if params_map is not None else design_to_params(ddf, base, prob.get("fixed"), ties)
        design_cols = [c for c in ddf.columns if c != "point"]
        cfg = dict(design_hash=_hash_design(pmap), T=int(T), base_seed=int(base_seed), crn=bool(crn),
                   series_every=int(series_every), evaluate=bool(evaluate),
                   model_cls=(model_cls.__module__ + "." + model_cls.__qualname__) if model_cls else
                   "u25.model.Universe")
        if eval_kw:
            from .objectives import validate_eval_kw
            eval_kw = validate_eval_kw(eval_kw)
            cfg["eval_kw"] = dict(eval_kw)
        sj = out / "sweep.json"
        if sj.exists():
            old = json.load(open(sj))
            diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
            if diff and not force:
                raise RuntimeError(f"configuración incompatible con el barrido existente en {out}: {diff} "
                                   f"(usar --force o otro --out)")
        json.dump({**cfg, "reps": max(reps, json.load(open(sj)).get("reps", 0) if sj.exists() else 0),
                   "n_points": len(pmap)}, open(sj, "w"), indent=1)
        # archivos auxiliares (compatibles con pipeline.io.load_run y pipeline analyze)
        ddf.to_csv(out / "design.csv", index=False)
        if ddir is not None:
            for fn in ("problem.json", "X.npy"):
                if (ddir / fn).exists() and (ddir / fn).resolve() != (out / fn).resolve():
                    shutil.copy2(ddir / fn, out / fn)
        npts = max(pmap) + 1
        plist_json = [json.loads(pmap[k].to_json()) if k in pmap else None for k in range(npts)]
        json.dump(plist_json, open(out / "params.json", "w"), indent=1)
        json.dump(dict(n_reps=reps, T=T, base_seed=base_seed, n_jobs=n_jobs, crn=crn, n_points=len(pmap),
                       design_columns=design_cols, ties=ties or {}, series_every=series_every,
                       record_every=1, runner="u25.sweep", model_cls=cfg["model_cls"],
                       u25_version=U25_VERSION,
                       refuge_hard_modes=sorted({pmap[k].refuge_hard_mode for k in pmap}),
                       eval_kw=eval_kw or {}),
                  open(out / "meta.json", "w"), indent=1)
        series_dir = None
        if series_every:
            series_dir = out / "series"
            series_dir.mkdir(exist_ok=True)

        summ_path = out / "summary.csv"
        header, done = read_done(summ_path)
        todo = [(pt, r) for pt in sorted(pmap) for r in range(reps) if (pt, r) not in done]
        total = len(pmap) * reps
        n_done0 = sum(1 for (pt, r) in done if pt in pmap and r < reps)
        if max_runs is not None:
            todo = todo[:max_runs]
        log(f"inicio: {n_done0}/{total} hechas, {len(todo)} por correr, n_jobs={n_jobs}, T={T}, out={out}")
        if any(pmap[k].refuge_hard_mode == "legacy" for k in pmap):
            log("aviso: hay puntos con refuge_hard='legacy' (capacidad de R7 de v1, no estrictamente dura)")
        if not todo:
            return dict(total=total, done=n_done0, new=0, failed=0)

        dvals = {int(r["point"]): r for _, r in ddf.iterrows()}
        tasks = [(pt, r, pmap[pt].to_dict(), T, base_seed, crn, str(series_dir) if series_dir else None,
                  series_every, evaluate, model_cls, eval_kw) for (pt, r) in todo]
        fsum = open(summ_path, "a", newline="")
        writer = csv.writer(fsum)
        t_start = time.perf_counter()
        n_new = n_fail = 0
        cpu_sum = 0.0
        stop = {"flag": False}

        def _term(signum, frame):
            stop["flag"] = True
            raise KeyboardInterrupt

        pool = None
        if n_jobs == 1:
            it = map(_worker, tasks)
        else:
            ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
            pool = ctx.Pool(n_jobs, initializer=_init_worker, maxtasksperchild=200)
            it = pool.imap_unordered(_worker, tasks, chunksize=1)
        old_term = signal.signal(signal.SIGTERM, _term)   # instalado después del fork de los workers
        try:
            for status, pt, r, payload in it:
                if status == "err":
                    n_fail += 1
                    with open(out / "failures.csv", "a") as ff:
                        ff.write(json.dumps(dict(point=pt, rep=r, error=payload)) + "\n")
                    log(f"ERROR point={pt} rep={r}: {payload.strip().splitlines()[-1]}")
                    continue
                row = payload
                drow = dvals[pt]
                for c in design_cols:
                    row[c] = drow[c]
                if header is None:
                    rest = [k for k in row if k not in ("point", "rep", "base_seed") and k not in design_cols]
                    header = ["point", *design_cols, "rep", "base_seed", *rest]
                    writer.writerow(header)
                extra = set(row) - set(header)
                writer.writerow([_fmt(row.get(k)) for k in header])
                fsum.flush()
                os.fsync(fsum.fileno())
                if extra and n_new == 0:
                    log(f"aviso: columnas no presentes en el header existente, se omiten: {sorted(extra)}")
                n_new += 1
                cpu_sum += row.get("wall_time_s", 0.0)
                if n_new % log_every == 0 or n_new == len(todo):
                    el = time.perf_counter() - t_start
                    rate = el / n_new
                    rem = len(todo) - n_new - n_fail
                    log(f"{n_done0 + n_new}/{total} hechas | {el / 60:.1f} min | {rate:.2f} s/corrida (pared) "
                        f"{cpu_sum / n_new:.2f} s/corrida (CPU) | ETA {rem * rate / 60:.1f} min | "
                        f"último p={pt} r={r} N_max={row.get('N_max')} t_ext={row.get('t_ext')}")
        except KeyboardInterrupt:
            log(f"interrumpido: {n_done0 + n_new}/{total} guardadas; relanzar el mismo comando para retomar")
            if pool is not None:
                pool.terminate()
                pool = None
            if stop["flag"]:
                raise SystemExit(143)
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)
            if pool is not None:
                pool.close()
                pool.join()
            fsum.close()
        log(f"fin de la invocación: {n_done0 + n_new}/{total} hechas, {n_fail} fallas, "
            f"{(time.perf_counter() - t_start) / 60:.1f} min")
        return dict(total=total, done=n_done0 + n_new, new=n_new, failed=n_fail)
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def collect_series(out) -> Path | None:
    """Concatena series/p*_r*.parquet en series.parquet (formato de run_ensemble / pipeline)."""
    out = Path(out)
    files = sorted((out / "series").glob("p*_r*.parquet"))
    if not files:
        return None
    S = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cols = ["point", "rep"] + [c for c in S.columns if c not in ("point", "rep")]
    S = S[cols].sort_values(["point", "rep", "t"])
    tmp = out / "series.parquet.tmp"
    S.to_parquet(tmp, index=False)
    os.replace(tmp, out / "series.parquet")
    return out / "series.parquet"


def load_summary(out) -> pd.DataFrame:
    """summary.csv ordenado por (point, rep), sin duplicados (por si una corrida se guardó dos veces)."""
    df = pd.read_csv(Path(out) / "summary.csv")
    return df.drop_duplicates(["point", "rep"], keep="last").sort_values(["point", "rep"]).reset_index(drop=True)
