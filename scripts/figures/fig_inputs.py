"""Insumos de `make figs`: verifica que estén y restituye los que se versionan comprimidos o reducidos.

    python3 scripts/figures/fig_inputs.py                              # verifica y, si falta, restituye
    python3 scripts/figures/fig_inputs.py --make results_v2/f2b_base   # crea series_figs.parquet desde series.parquet
    python3 scripts/figures/fig_inputs.py --pack                       # comprime de nuevo summary.csv / params.json

* `results_v2/**/summary.csv` y `results_v2/**/params.json` se versionan comprimidos (`*.gz`, gzip con mtime = 0).
  Si falta la copia sin comprimir, se descomprime al lado (byte a byte idéntica; .gitignore ignora las copias).
  Después de re-correr una etapa en results_v2/, `--pack` vuelve a escribir los `.gz` a partir de las copias.
* `results_v2/f2b_base/series_figs.parquet` (~3,5 MB, versionado): serie reducida a las columnas que leen
  fig_trajectories, fig_cohorts y numbers_extra de `scripts/figures/figs_results.py` (point, rep, t, N, births,
  S_mean, S_adult, mean_age). Si falta `series.parquet` (la serie completa, 28 MB, no se versiona), se copia desde
  aquí. Para regenerar la serie completa desde cero (determinista, bit a bit; ≈ 3–8 min con 4 núcleos):
      python3 scripts/production/u25x.py sweep \
          --design scripts/production/designs_v2/calhoun_C1/base --reps 200 --T 600 --preset calhoun_C1 \
          --full-series --series-every 1 --base-seed 25301 --model v2 --out <dir>
      python3 -m u25 collect <dir>
* `results/meanfield/kin_sweep_*.log` y `ens_rw_40x2000.pkl` (campo medio cinético y ABM rápido, de
  scripts/analysis/mf_kin.py y ensamble_rapido.py), `results/baseline_alpha14/analisis/{km,ciclos}.csv`
  (`python -m pipeline baseline`), `results_v2/f2b_*/progress.log`, `results/fisico_v3/fig2c_series.csv` (medias
  de s de C1 sin dilución, de scripts/analysis/analiza_gamma.py), `results/fisico_v3/gamma_v3.parquet` (Γ, de
  scripts/analysis/gamma_v3.py) y `results_v2/o11_aw/<etapa>/summary.csv` (O11 a la edad a_w, de `make o11-aw`):
  versionados.
Sale con código 1 si falta algún insumo que no se puede restituir (y entonces `make figs` se detiene antes de
escribir nada).
"""
from __future__ import annotations

import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_COLS = ["point", "rep", "t", "N", "births", "S_mean", "S_adult", "mean_age"]
SERIES_STAGES = [("results_v2", "f2b_base")]   # (raíz, etapa) cuya serie leen las figuras
PACKED = ("summary.csv", "params.json")         # se versionan como <nombre>.gz
LOGS = ["results/meanfield/kin_sweep_05_8.log", "results/meanfield/kin_sweep_1_5.log",
        "results/meanfield/ens_rw_40x2000.pkl",
        "results/baseline_alpha14/analisis/km.csv", "results/baseline_alpha14/analisis/ciclos.csv",
        "results/fisico_v3/fig2c_series.csv",      # panel (c) de fig_res_trajectories
        "results/fisico_v3/gamma_v3.parquet",      # Γ (numbers_v3 de figs_results.py)
        *(f"results_v2/o11_aw/{st}/{f}" for st in ("f2b_base", "f2b_nulos", "f2b_rob")
          for f in ("summary.csv", "labels.json"))]   # O11 a la edad a_w (numbers_v3; make o11-aw)


def make(stage_dir: Path) -> Path:
    import pandas as pd
    stage_dir = Path(stage_dir).resolve()
    S = pd.read_parquet(stage_dir / "series.parquet", columns=FIG_COLS)
    out = stage_dir / "series_figs.parquet"
    S.to_parquet(out, index=False, compression="zstd")
    return out


def _gzip(src: Path, dst: Path) -> None:
    with open(dst, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=9) as g:
        g.write(src.read_bytes())


def unpack() -> int:
    """Descomprime results_v2/**/{summary.csv,params.json}.gz si falta la copia sin comprimir."""
    n = 0
    for gz in sorted((ROOT / "results_v2").rglob("*.gz")):
        plain = gz.with_suffix("")
        if plain.name in PACKED and not plain.exists():
            plain.write_bytes(gzip.decompress(gz.read_bytes()))
            n += 1
    if n:
        print(f"results_v2: {n} archivos descomprimidos (summary.csv / params.json)")
    return n


def pack() -> int:
    """Escribe <archivo>.gz para cada summary.csv / params.json de results_v2 cuyo .gz falte o difiera."""
    n = 0
    for name in PACKED:
        for plain in sorted((ROOT / "results_v2").rglob(name)):
            gz = plain.with_name(plain.name + ".gz")
            if not gz.exists() or gzip.decompress(gz.read_bytes()) != plain.read_bytes():
                _gzip(plain, gz)
                n += 1
                print("comprimido:", gz.relative_to(ROOT))
    return n


def check() -> int:
    unpack()
    missing = []
    for root, st in SERIES_STAGES:
        d = ROOT / root / st
        full, red = d / "series.parquet", d / "series_figs.parquet"
        if not full.exists():
            if red.exists():
                shutil.copy2(red, full)
                print(f"{full.relative_to(ROOT)}: copiada desde {red.name} (columnas de las figuras)")
            else:
                missing.append(f"{full.relative_to(ROOT)} (ver la receta en scripts/figures/fig_inputs.py)")
    for f in LOGS:
        if not (ROOT / f).exists():
            missing.append(f)
    for f in missing:
        print("FALTA:", f)
    if not missing:
        print("insumos de make figs: completos")
    return 1 if missing else 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--make":
        print(make(Path(sys.argv[2])))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--pack":
        pack()
        sys.exit(0)
    sys.exit(check())
