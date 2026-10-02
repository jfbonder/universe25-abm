"""Re-evaluación de barridos con el evaluador vigente.

    python -m u25 reeval results/f2b_base --out results_v2/reeval_v1/f2b_base [--n-jobs 4]
    python -m u25 reeval DIR --out OUT --from-series          # sin simular, desde las series guardadas
    python -m u25 reeval DIR --out OUT --peak-time plateau_start --crowd-class-threshold 8

Nunca modifica el directorio de origen: escribe un barrido nuevo en `--out` con el mismo formato
(`summary.csv`, `params.json`, `meta.json`, ...), más `reeval.json` (origen, opciones) y `reeval_check.json`
(comparación con el `summary.csv` de origen: columnas no-objetivo que difieren y cambios de estado por objetivo).

Modo por defecto (re-simulación): el modelo es determinista, así que cada (punto, réplica) se vuelve a correr con
la misma semilla (`SeedSequence([base_seed, punto, rep])` o `[base_seed, rep]` con CRN, leídos de sweep.json) y se
evalúa con `u25.objectives` vigente. Es reanudable (usa `u25.sweep.run_sweep`).

R7 en barridos de v1: en v1 `refuge_hard=True` era la capacidad de una sola pasada, que en v2 se llama 'legacy'
(en v2 True = 'strict'). Con `--legacy-r7 auto` (default) los barridos sin `u25_version >= 2` en meta.json se
re-corren con refuge_hard='legacy', y así reproducen v1 bit a bit; `--legacy-r7 no` los corre con la regla v2.

Modo `--from-series`: lee `series.parquet` o `series/p*_r*.parquet` (deben estar a resolución completa,
series_every <= 1, con las columnas que usa el evaluador) y re-evalúa sin simular; las columnas no-objetivo del
summary se copian del origen.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Params

OBJ_PREFIX = "obj_"


def _load_json(path: Path, default=None):
    return json.load(open(path)) if path.exists() else default


def _import_cls(spec: str | None):
    """'paquete.modulo.Clase' o 'modulo:Clase' -> clase; None/'' -> None (Universe)."""
    if not spec or spec in ("u25.model.Universe", "none", "Universe"):
        return None
    mod, _, name = spec.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(mod), name)


def source_config(src) -> dict:
    """Configuración del barrido de origen: T, base_seed, crn, reps, model_cls, u25_version, design_columns."""
    src = Path(src)
    sj = _load_json(src / "sweep.json", {})
    meta = _load_json(src / "meta.json", {})
    if not sj and not meta:
        raise FileNotFoundError(f"{src}: falta sweep.json/meta.json (no es un barrido de u25.sweep)")
    return dict(T=int(sj.get("T", meta.get("T"))), base_seed=int(sj.get("base_seed", meta.get("base_seed", 0))),
                crn=bool(sj.get("crn", meta.get("crn", False))), reps=int(sj.get("reps", meta.get("n_reps", 0))),
                model_cls=sj.get("model_cls", meta.get("model_cls", "u25.model.Universe")),
                u25_version=int(meta.get("u25_version", sj.get("u25_version", 1))),
                design_columns=list(meta.get("design_columns") or []),
                series_every=int(sj.get("series_every", meta.get("series_every", 0)) or 0))


def source_params(src, legacy_r7: str = "auto", version: int = 1) -> dict:
    """{point: Params} desde params.json, con la traducción de refuge_hard de v1 si corresponde."""
    plist = json.load(open(Path(src) / "params.json"))
    legacy = legacy_r7 == "yes" or (legacy_r7 == "auto" and version < 2)
    out = {}
    for i, d in enumerate(plist):
        if d is None:
            continue
        if legacy and d.get("refuge_hard") in (True, "true", "True", 1):
            d = {**d, "refuge_hard": "legacy"}
        out[i] = Params.from_dict(d)
    return out


def compare(src_summary: pd.DataFrame, new_summary: pd.DataFrame) -> dict:
    """Columnas no-objetivo que difieren (por fila emparejada en (point, rep)) y cambios de estado por obj_*."""
    a = src_summary.drop_duplicates(["point", "rep"], keep="last").set_index(["point", "rep"])
    b = new_summary.drop_duplicates(["point", "rep"], keep="last").set_index(["point", "rep"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    skip = {"wall_time_s", "base_seed"}
    diff_cols, obj_changes = {}, {}
    for c in a.columns.intersection(b.columns):
        if c in skip:
            continue
        x, y = a[c], b[c]
        if c.startswith(OBJ_PREFIX):
            xs, ys = x.astype(object).fillna("nan").astype(str), y.astype(object).fillna("nan").astype(str)
            ch = xs != ys
            if ch.any():
                obj_changes[c] = {f"{u}->{v}": int(n) for (u, v), n in
                                  pd.DataFrame({"u": xs[ch], "v": ys[ch]}).value_counts().items()}
            continue
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if xn.notna().any() or yn.notna().any():
            same = (np.isclose(xn, yn, rtol=1e-12, atol=0, equal_nan=True)) | (xn.isna() & yn.isna())
        else:
            same = (x.isna() & y.isna()) | (x.astype(object).fillna("").astype(str) ==
                                             y.astype(object).fillna("").astype(str))
        if not same.all():
            diff_cols[c] = int((~same).sum())
    return dict(n_rows=int(len(idx)), n_src=int(len(src_summary)), n_new=int(len(new_summary)),
                differing_non_objective_columns=diff_cols,
                new_columns=sorted(set(b.columns) - set(a.columns)),
                objective_status_changes=obj_changes)


def _series_iter(src: Path):
    if (src / "series.parquet").exists():
        S = pd.read_parquet(src / "series.parquet")
        for key, g in S.groupby(["point", "rep"], sort=True):
            yield key, g
    else:
        for f in sorted((src / "series").glob("p*_r*.parquet")):
            g = pd.read_parquet(f)
            yield (int(g["point"].iloc[0]), int(g["rep"].iloc[0])), g


def reeval_from_series(src, out, legacy_r7="auto", eval_kw=None) -> dict:
    from .objectives import evaluate_run
    src, out = Path(src), Path(out)
    cfg = source_config(src)
    if cfg["series_every"] > 1:
        raise ValueError(f"{src}: series compactas (series_every={cfg['series_every']}); usar la re-simulación")
    pmap = source_params(src, legacy_r7, cfg["u25_version"])
    summ = pd.read_csv(src / "summary.csv", low_memory=False, keep_default_na=False, na_values=[""])
    summ = summ.drop_duplicates(["point", "rep"], keep="last")
    rows = {(int(r.point), int(r.rep)): r for r in summ.itertuples(index=False)}
    cols = list(summ.columns)
    out.mkdir(parents=True, exist_ok=True)
    new = []
    for (pt, rp), g in _series_iter(src):
        base = rows.get((int(pt), int(rp)))
        if base is None:
            continue
        d = dict(zip(cols, base))
        res = evaluate_run(g, pmap[int(pt)], d, **(eval_kw or {}))
        for k, v in res.items():
            if k == "_times":
                continue
            name = "obj_O17_run" if k == "calhoun_like" else f"obj_{k}"
            d[name] = v.value if k == "G_B" else v.status
            if k == "R_B":
                d["obj_R_B_val"] = v.value
        new.append(d)
    df = pd.DataFrame(new)
    df.to_csv(out / "summary.csv", index=False)
    for fn in ("params.json", "meta.json", "design.csv", "labels.json", "stage.json", "sweep.json"):
        if (src / fn).exists():
            (out / fn).write_text((src / fn).read_text())
    return dict(total=len(summ), done=len(df), new=len(df), failed=0)


def reeval_dir(src, out, n_jobs: int = 4, legacy_r7: str = "auto", model_cls: str | None = "auto",
               from_series: bool = False, eval_kw: dict | None = None, quiet: bool = True,
               max_runs: int | None = None, series_every: int = 0) -> dict:
    """Re-evalúa el barrido `src` en `out` (ver docstring del módulo). Devuelve el resumen de la comparación."""
    from .sweep import run_sweep
    src, out = Path(src).resolve(), Path(out).resolve()
    if src == out:
        raise ValueError("--out debe ser distinto del origen (los barridos de origen no se modifican)")
    cfg = source_config(src)
    legacy = legacy_r7 == "yes" or (legacy_r7 == "auto" and cfg["u25_version"] < 2)
    if from_series:
        r = reeval_from_series(src, out, legacy_r7, eval_kw)
    else:
        pmap = source_params(src, legacy_r7, cfg["u25_version"])
        if model_cls == "auto":
            try:
                mc = _import_cls(cfg["model_cls"])
            except Exception as e:  # noqa: BLE001
                print(f"aviso: no se pudo importar {cfg['model_cls']} ({e}); se usa u25.model.Universe "
                      f"(las columnas x_* de f2b_model no se regeneran)")
                mc = None
        else:
            mc = _import_cls(model_cls)
        summ = pd.read_csv(src / "summary.csv", low_memory=False)
        dcols = cfg["design_columns"] or []
        if (src / "design.csv").exists():
            ddf = pd.read_csv(src / "design.csv")
        else:
            ddf = summ.drop_duplicates("point")[["point", *dcols]].reset_index(drop=True)
        ddf = ddf[ddf["point"].isin(list(pmap))]
        reps = cfg["reps"] or int(summ["rep"].max()) + 1
        r = run_sweep(ddf, reps, out, T=cfg["T"], base_seed=cfg["base_seed"], n_jobs=n_jobs, crn=cfg["crn"],
                      model_cls=mc, quiet=quiet, max_runs=max_runs, params_map=pmap, eval_kw=eval_kw,
                      series_every=series_every)
        for fn in ("labels.json", "stage.json"):
            if (src / fn).exists():
                (out / fn).write_text((src / fn).read_text())
    info = dict(source=str(src), legacy_r7=bool(legacy), source_u25_version=cfg["u25_version"],
                from_series=bool(from_series), eval_kw=eval_kw or {}, result=r)
    json.dump(info, open(out / "reeval.json", "w"), indent=1)
    src_s = pd.read_csv(src / "summary.csv", low_memory=False, keep_default_na=False, na_values=[""])
    new_s = pd.read_csv(out / "summary.csv", low_memory=False, keep_default_na=False, na_values=[""])
    chk = compare(src_s, new_s)
    json.dump(chk, open(out / "reeval_check.json", "w"), indent=1)
    return dict(**info, check=chk)


def find_sweeps(root="results", pattern="f2b_*") -> list[Path]:
    """Directorios de barrido (con sweep.json y summary.csv) bajo root/pattern, ordenados."""
    return sorted(p.parent for p in Path(root).glob(f"{pattern}/sweep.json") if (p.parent / "summary.csv").exists())
