# Universe 25 ABM

Code, simulation outputs and manuscripts of

> **An Agent-Based Model for the Social Collapse of Calhoun's Universe 25 Experiment**
> Julián Fernández Bonder (Departamento de Matemática, FCEN, Universidad de Buenos Aires, and Instituto de
> Cálculo, UBA–CONICET, Buenos Aires, Argentina)

In Universe 25, J. B. Calhoun (1973) let a colony of mice grow in an enclosure with unlimited food, water and
nesting material. The population grew, stagnated, lost its social and reproductive behaviour (the "beautiful
ones") and went extinct. This repository contains a minimal agent-based model that reproduces those
observations *qualitatively*: mice on a lattice with a social state that is damaged by crowding, a critical
window for social learning (R1), learning from neighbours (R2), crowd aversion (AV) and refuges of hard capacity
(R7). The paper calibrates the model, analyses a mean-field reduction, and tests seven preregistered hypotheses on
about 51 800 production runs.

The paper comes in two versions, both built from the same data:

| Version | Files |
|---|---|
| Extended version (all analyses: calibration, mean field, complete results, full ODD description) | `manuscript/extended/main.pdf` |
| Reduced version for *Physical Review E* | `manuscript/pre/main.pdf` and its Supplemental Material `manuscript/pre/supplement.pdf` |

## Repository layout

| Path | Contents |
|---|---|
| `u25/` | The model, as a Python package: `params.py` (`Params` and the presets `notebook`, `longevo`, `calhoun_C0`, `calhoun_C1p`, `calhoun_C1`, `calhoun_C2`, ...), `model.py` (`Universe`; agents in struct-of-arrays, sparse affinities), `kernels.py` (numba), `stats.py` (time series and run summaries), `objectives.py` (automatic evaluator of the patterns O1–O17, O13p, O17p and the anti-patterns A1–A4), `experiments.py` (transplants), `ensemble.py`, `sweep.py` (resumable sweeps), `reeval.py` (re-scoring of sweeps) and the command line `python -m u25` |
| `tests/` | pytest suite (127 tests): bit-for-bit fingerprints of the presets, exactness of the sparse affinities against a slow port of the original code, hard refuge capacity, the evaluator on synthetic series, determinism and resumption of sweeps |
| `scripts/production/` | Production runs: orchestrator `run_2b_v2.sh`, runner `u25x.py`, model classes with the extra per-run observables (`f2b_model.py`, `f2b_model_v2.py`), transplant protocol `trans_v2.py`, design generators `gen_designs.py` / `gen_designs_v2.py`, evaluator settings `v2_api.json` and the designs of every stage (`designs_v2/calhoun_C1/`) |
| `scripts/analysis/` | Statistical pipeline (`pipeline/`, run as `python -m pipeline`: survival, Holm/BH families, Morris/Sobol, null models, re-scoring), mean-field models (`mf_ode.py`, `mf_edad.py`, `mf_kin.py`, `mf_lineage_v2.py`, fast ABM `abm_rapido.py`, `ensamble_rapido.py`), generation register and Γ (`generaciones.py`, `generaciones_base.py`, `analiza_generaciones.py`, `gamma_v3.py`, `analiza_gamma.py`) and O11 at age a_w (`o11_aw.py`) |
| `scripts/figures/` | Figures and tables: `figs_results.py` (results figures, all `tab_res_*.tex` tables and `results_numbers.json`), `figs_paper.py` (mean-field figures), `figs_pre.py` (figures of the PRE version), `fig_model_schematic.py` / `fig_model_schematic_pre.py` (model schematic) and `fig_inputs.py` (checks and restores the inputs of `make figs`) |
| `results_v2/` | Outputs of the production runs, from which every number, table and results figure of the paper is computed: 26 stages `f2b_<stage>/` (per stage: `summary.csv.gz` with one row per run, `params.json.gz`, `meta.json` with the base seed, `labels.json`, `stage.json`, `design.csv`, `progress.log` and the analysis files read by the figures), the confirmatory family and consolidated reports `f2b_final/`, O11 at age a_w `o11_aw/`, and `results_numbers.json` (every number quoted in the text, with its ensemble) |
| `results/` | Auxiliary inputs: `fisico_v3/` (Γ re-runs: `gamma_v3.parquet`, `gamma_v3_series.parquet`, `fig2c_series.csv`), `fisico_v2/` (generation register of the reference ensemble), `meanfield/` (fast-ABM ensemble and kinetic mean-field sweeps) and `baseline_alpha14/` (baseline of the original model and its survival analysis) |
| `manuscript/extended/` | LaTeX sources of the extended version (`main.tex`, `sections/`, `tables/`, `figs/`, `refs.bib`, `build.sh`) and `main.pdf` |
| `manuscript/pre/` | LaTeX sources of the PRE version (`main.tex`, `supplement.tex`, `sections/`, `tables/`, `figs/`, `refs.bib`, `build.sh`), `main.pdf` and `supplement.pdf` |
| `docs/` | `preregistered_plan.md`: the preregistered statistical plan (1 October 2026, 22:02 UTC); `production_plan.md`: the stages, decision rules and primary evaluator fixed before the production runs (2 October 2026, 05:34 UTC). Both verbatim, with a header giving their date and the equivalence of paths |

Code comments, log messages and some column labels of the outputs are in Spanish. The preset `notebook` and
`u25/reference.py` refer to the original (unpublished) notebook implementation of the model that `u25`
re-implements; references to commits such as `253b3fc` (v1) are to the development repository, which is not public.

## Installation

Tested with Python 3.11 on Linux x86-64.

```bash
python3 -m pip install -r requirements.txt     # exact versions used for the results (or: make install)
python3 -m pip install -e .                    # optional: installs the u25 package and the `u25` command
```

The first run compiles the numba kernels (10–20 s); they are cached in `u25/__pycache__/`. Without numba the code
still runs (pure-Python fallback), but about 100 times slower. Building the PDFs needs **pdflatex and bibtex**
(TeX Live with REVTeX 4.2, e.g. `apt install texlive-latex-extra texlive-publishers texlive-bibtex-extra`), which
pip does not install; `make check-latex` checks for them.

The scripts import `u25` and each other from `scripts/production`, `scripts/analysis` and `scripts/figures`. The
Makefile sets `PYTHONPATH` accordingly; to run a script by hand, from the repository root:

```bash
export PYTHONPATH=$PWD:$PWD/scripts/production:$PWD/scripts/analysis:$PWD/scripts/figures
```

## Tests

```bash
make test          # = python3 -m pytest -q tests   (127 tests, about 1-1.5 min with 4 free cores)
```

## Quick use of the model

```bash
python3 -m u25 run --preset calhoun_C1 --T 600 --seed 1
python3 -m u25 ensemble --preset calhoun_C1 --n-reps 30 --T 600 --out /tmp/c1        # 30 replicates
python3 -m u25 objectives /tmp/c1                                                    # O1–O17 / A1–A4, Wilson CIs
python3 -m u25 transplant --preset calhoun_C1 --n-reps 30 --out /tmp/c1_transplant   # O12 / O12b
```

Every parameter is documented in `u25/params.py`.

## Reproducing the figures, tables and PDFs

```bash
make figs          # every figure, table and results_v2/results_numbers.json, from the versioned outputs (~3 min)
make paper         # manuscript/extended/main.pdf, manuscript/pre/main.pdf and manuscript/pre/supplement.pdf (<1 min)
make figs-clean-check   # clones HEAD into a temporary directory, runs make figs there and fails if any file changes
```

`make figs` does not simulate, except for two short runs of the fast mean-field ABM inside `figs_paper.py`. It
runs, in order:

1. `scripts/figures/fig_inputs.py` (also `make data`): decompresses `results_v2/**/summary.csv.gz` and
   `params.json.gz` next to the compressed files (they are stored compressed to keep the repository small; the
   uncompressed copies are ignored by git), restores the series of `results_v2/f2b_base` from its reduced copy
   `series_figs.parquet`, and stops with an error if any input is missing;
2. `figs_paper.py`: the four mean-field figures of the extended version;
3. `fig_model_schematic.py`: the model schematic;
4. `figs_results.py`: the results figures `fig_res_*.pdf`, the 19 tables `tab_res_*.tex` of the extended version and
   `results_v2/results_numbers.json`; if any output fails, the JSON is not rewritten and the script exits with 1;
5. `fig_model_schematic_pre.py` and `figs_pre.py`: the figures of the PRE version and the copies of the tables
   that its supplement includes.

The PDFs are reproducible byte for byte: the Makefile fixes `SOURCE_DATE_EPOCH`, which matplotlib uses as the
creation date. Run on a clean clone, `make figs` leaves the working tree unchanged.

## Running the production simulations from scratch

All the ensembles of the paper (26 stages, about 51 800 runs: reference ensemble, null models, single and
multiple ablations, transplants, robustness variants, phase planes, lattice sizes, Morris and Sobol' global
sensitivity) are generated by one resumable orchestrator with fixed base seeds:

```bash
make sweep                     # RES=results_new by default; about 4.5-5 h of wall time with 4 cores
make sweep RES=/path/to/out NJOBS=8
make sweep-smoke               # smoke test: every stage with 2 replicates per point and T <= 250 (a few minutes)
```

`make sweep` calls `scripts/production/run_2b_v2.sh` (variables: `RES`, `N_JOBS`, `STAGES` for a subset in
manifest order, `DEADLINE_H`, `SMOKE`). For each stage it reads the design in
`scripts/production/designs_v2/calhoun_C1/<stage>/`, runs it with the production model class
(`f2b_model_v2.U2v2`) and the primary evaluator (`v2_api.json`), and analyses it with `python -m pipeline`; at the
end it writes the confirmatory family and the consolidated reports in `<RES>/f2b_final/`. Re-running the same
command resumes every stage where it stopped, and a second orchestrator on the same `RES` exits (file lock). The
orchestrator shortens or skips low-priority stages if they do not fit before its deadline; `make sweep` sets
`DEADLINE_H=12`, so that on a slower machine every stage still runs with its full number of replicates.

Every run uses `numpy.random.SeedSequence([base_seed, point, rep])` (or `[base_seed, rep]` with common random
numbers), so the results do not depend on the number of processes, the order of execution or interruptions; the
seeds of every stage are in `results_v2/f2b_<stage>/meta.json`. A re-run reproduces the versioned `summary.csv`
files. The figures read `results_v2/`; to use a new run, replace the stage directories there (or run with
`RES=results_v2` after `make data`, which skips the stages that are already complete) and recompress with
`python3 scripts/figures/fig_inputs.py --pack`.

Other runs behind specific numbers of the paper:

```bash
make o11-aw        # O11 at age a_w: re-runs f2b_base, f2b_nulos and the s0 arms of f2b_rob with the same seeds (~10 min)
make gamma         # Γ without dilution and the series of Fig. 2c: re-runs 710 runs into results/fisico_v3/ (~10-20 min)
make meanfield     # fast-ABM ensemble and kinetic mean-field sweeps of the mean-field figures (~6 min)
make baseline      # baseline of the original model (alpha = 1.4, 100 replicates, T = 2000) and its KM analysis (~2 min)
python3 scripts/analysis/generaciones_base.py /tmp/gen_base.pkl 0,1,2,3 200   # generation register (tables of the calibration appendix)
python3 scripts/analysis/analiza_generaciones.py /tmp/gen_base.pkl
python3 scripts/analysis/mf_lineage_v2.py                                   # lineage criterion of the mean-field appendix
```

The full series of the reference ensemble (28 MB) is not versioned; it is regenerated, bit for bit, by

```bash
python3 scripts/production/u25x.py sweep --design scripts/production/designs_v2/calhoun_C1/base --reps 200 --T 600 \
    --preset calhoun_C1 --full-series --series-every 1 --base-seed 25301 --model v2 --out /tmp/f2b_base
python3 -m u25 collect /tmp/f2b_base
```

## Costs (4 cores of a cloud container)

| Task | Cost |
|---|---|
| One run of `calhoun_C1` (T = 600) | 0.75–1.5 s of CPU |
| Test suite (127 tests) | about 1–1.5 min |
| `make figs` | about 3 min |
| `make paper` (three PDFs) | under 1 min |
| Reference ensemble `f2b_base` (800 runs, T = 600) | about 3 min of wall time |
| `make o11-aw` (2 400 runs) | about 10 min |
| `make sweep`: all production runs (about 51 800 runs) | about 4.4 h of wall time (recorded in the `progress.log` files); allow 5 h |

## Preregistration

The seven confirmatory hypotheses (Q1–Q7), their predictions, designs and decision rules were fixed before the
production runs. The plan is reproduced verbatim, with its date, in
[`docs/preregistered_plan.md`](docs/preregistered_plan.md); the stages, decision rules and primary evaluator that
were added afterwards, also before the production runs, are in [`docs/production_plan.md`](docs/production_plan.md).
Every deviation from the preregistered plan is listed in the manuscripts.

## License

- The **code** (`u25/`, `tests/`, `scripts/`, `Makefile` and the build scripts) is released under the MIT License:
  see [`LICENSE`](LICENSE).
- The **texts, figures and data** (`manuscript/`, `docs/`, `results/`, `results_v2/`) are released under the
  Creative Commons Attribution 4.0 International License (CC BY 4.0): see
  [`LICENSE-CC-BY-4.0.txt`](LICENSE-CC-BY-4.0.txt).

© 2026 Julián Fernández Bonder. If you use this work, please cite it as in [`CITATION.cff`](CITATION.cff).
