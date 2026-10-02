"""Lectura de la salida de u25.run_ensemble."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

PARAM_COLS_SKIP = {"point", "rep", "base_seed"}


def load_run(run_dir, series=False):
    """Devuelve dict(summary, params, meta, series|None, dir). `series=True` carga series.parquet (o csv.gz)."""
    d = Path(run_dir)
    out = {"dir": d, "summary": pd.read_csv(d / "summary.csv")}
    out["params"] = json.load(open(d / "params.json")) if (d / "params.json").exists() else None
    out["meta"] = json.load(open(d / "meta.json")) if (d / "meta.json").exists() else {}
    out["series"] = load_series(d) if series else None
    s = out["summary"]
    if "T" not in s:
        s["T"] = out["meta"].get("T", np.nan)
    if "extinct" in s and s["extinct"].dtype != bool:
        s["extinct"] = s["extinct"].astype(str).str.lower().isin(["true", "1"])
    return out


def load_series(run_dir):
    d = Path(run_dir)
    if (d / "series.parquet").exists():
        return pd.read_parquet(d / "series.parquet")
    if (d / "series.csv.gz").exists():
        return pd.read_csv(d / "series.csv.gz")
    return None


def event_table(summary: pd.DataFrame):
    """(tiempo, evento) para análisis de supervivencia: t_ext si extinta, si no t_end (censura a la derecha)."""
    s = summary
    time = np.where(s["extinct"], s["t_ext"], s["t_end"] if "t_end" in s else s["T"]).astype(float)
    event = s["extinct"].to_numpy().astype(bool)
    return time, event


def design_columns(summary: pd.DataFrame):
    """Columnas de parámetros que varían entre puntos (las que agrega run_ensemble antes de las métricas)."""
    cols = []
    for c in summary.columns:
        if c == "N_max":
            break
        if c not in PARAM_COLS_SKIP:
            cols.append(c)
    return cols
