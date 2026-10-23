# Density movies and maps (supplementary material)

Spatial density movies of single runs of the agent-based model (Universe 25 analogue), for the applied
article (Supplementary videos S1-S3) and the public repository. One run per model, 30 x 30 grid, **seed 1**
(`rng = numpy.random.default_rng(SeedSequence([20261017, 1]))`, the same seed and generator as the single colony of
manuscript Fig. 3f; peak withdrawn counts reproduce the figure: 416 spatial, 6 well-mixed).

## Regenerate

    python scripts/figures/density_movies.py [--seed 1] [--fps 12] [--every 2] [--only C1,null] [--nosim]

Simulates the 8 models in 4 processes (~2 min) and renders (~2 min). `--nosim` reuses `cache/*.npz`
(weekly density map, competent count per cell and withdrawn non-breeders per cell). Requires numpy, matplotlib, ffmpeg.
The package `u25/` is not modified.

## Files

| File | Content |
|---|---|
| `video_S1_comparison.mp4` | Supplementary video S1: mosaic of C1 (spatial), C1 well-mixed, short-lived control and null model, synchronised, weeks 0-600 |
| `video_S2_C1_spatial.mp4` | Supplementary video S2: C1 spatial |
| `video_S3_C1_wellmixed.mp4` | Supplementary video S3: C1 well-mixed |
| `movie_C0.mp4`, `movie_C1p.mp4`, `movie_C2.mp4`, `movie_short_lived.mp4`, `movie_long_lived.mp4`, `movie_null.mp4` | Same movie for the other models |
| `snapshots_<model>.png` | Six snapshots per model: founding; growth (N first >= 50% of peak); peak; stagnation (midpoint between peak and decline); decline (N <= 50% of peak after the peak); last live week |
| `cache/<model>.npz` | Simulated maps and series (inputs of all figures) |

Movies: 1280 x 720, 12 fps, one frame every 2 weeks, ending 10 weeks after extinction (runs that survive end at T).

Panels: left, animals per cell (grey, square-root scale, common to all models and saturated at the 99.5th percentile of
the occupied cells of all models, vmax = 81); right, fraction of competent animals per cell (dark grey = empty cell);
bottom, population N(t) with a cursor, the phases A-D shaded, and the number of withdrawn non-breeders (right axis,
magenta). The time axis ends at extinction plus a 10-week margin.

## Definitions of the marks

- **Withdrawn non-breeders** (magenta dots; dot size ~ sqrt of the number in the cell): adults (age >= `repro_age_min`)
  with competence s < 0.2 (`obs_s_low`) that have been isolated (at most 2 animals in the cell, `obs_iso_m`) for at
  least 4 consecutive weeks (`obs_iso_weeks`); the package condition phi_BO, as in the manuscript. A lone last
  survivor (N <= 1) is not counted.
- **Retreat sites** (blue squares): cells with a refuge, capacity 2 animals. In well-mixed C1 they act only as capacity.
- **Competent** (right panel): s >= 0.2.
- **Phases A-D** (as in manuscript Figs. 1-3): A to B, first
  week with births; B to C, growth rate of ln N (centred over 4 weeks) falls below half its maximum (first 60 weeks);
  C to D, start of the plateau, N >= 0.98 max N; D ends at extinction. A phase that does not exist in a run is omitted.

## Models

- C0, C1' (`C1p`), C1, C2: presets `calhoun_C0`, `calhoun_C1p`, `calhoun_C1`, `calhoun_C2` (T = 600; C2: 700).
- C1 well-mixed: preset `calhoun_C1` with `jump_prob = 1` and `jump_mode = "uniform"` only (point 0 of
  `em_wm_run.py`); the retreat sites of the preset (30% of cells, capacity 2) remain and act only as capacity.
- Short-lived control: preset `notebook` with `a_max = 60`, `repro_age_max = 50`.
- Long-lived control: preset `longevo` (alpha = 1.4). Null model: `longevo` with alpha = 4 (no rules).
- Controls run to T = 600 (the manuscript uses T = 1000). A single run per model is illustrative; ensemble statistics
  are in the manuscript.
