"""Pipeline estadístico reutilizable para los ensambles de `u25` (scripts/analysis/pipeline).

Módulos
  io           lectura de results/<dir>/{summary.csv, series.parquet, params.json, meta.json}
  design       diseños Morris / LHS / Saltelli (SALib) -> listas de Params para u25.run_ensemble
  metrics      métricas por corrida (desde series) y por punto de diseño (agregación de réplicas)
  sensitivity  análisis de Morris (mu*, sigma) y Sobol (S1, ST con IC bootstrap)
  survival     Kaplan-Meier con censura, Nelson-Aalen, log-rank, MLE Weibull/lognormal/exponencial, GOF
  peaks        picos/valles, Mann-Kendall, pendiente de Sen, test de drift del ensamble
  multiple     Holm y Benjamini-Hochberg
  plots        figuras
  report       informes listos (línea base, Morris, Sobol)

Uso rápido
  # desde la raíz del repositorio
  export PYTHONPATH=$PWD:$PWD/scripts/production:$PWD/scripts/analysis
  python3 -m pipeline baseline results/baseline_alpha14 --out results/baseline_alpha14/analisis
  python3 -m pipeline design morris --out /tmp/morris --r 6 --problem scripts/analysis/pipeline/problems/chico.json
  python3 -m pipeline run /tmp/morris --n-reps 3 --T 600
  python3 -m pipeline analyze morris /tmp/morris --out /tmp/morris/analisis
"""
from . import io, design, metrics, sensitivity, survival, peaks, multiple, plots, report  # noqa: F401
