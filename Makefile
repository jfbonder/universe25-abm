# Universe 25 ABM: common tasks. See README.md.
PY ?= python3
NJOBS ?= 4
export PYTHONPATH := $(CURDIR):$(CURDIR)/scripts/production:$(CURDIR)/scripts/analysis:$(CURDIR)/scripts/figures
# Reproducible PDFs: matplotlib uses SOURCE_DATE_EPOCH as the creation date in the PDF metadata
export SOURCE_DATE_EPOCH ?= 1790899200

.PHONY: help install test test-fast data fig-inputs figs figs-clean-check paper paper-extended paper-pre check-latex \
        sweep sweep-smoke o11-aw gamma meanfield baseline all clean-pyc clean-tex

help:
	@echo "make install          install the pinned dependencies (requirements.txt)"
	@echo "make test             full test suite (pytest, ~1-2 min with 4 free cores)"
	@echo "make figs             regenerate every figure, table and results_v2/results_numbers.json from the"
	@echo "                      versioned outputs (no simulation; ~3 min)"
	@echo "make figs-clean-check clone HEAD into a temporary directory, run make figs there and check that nothing changes"
	@echo "make paper            build manuscript/extended/main.pdf, manuscript/pre/main.pdf and supplement.pdf (pdflatex + bibtex)"
	@echo "make sweep            production runs from scratch into RES (default results_new/; ~51 800 runs, ~4.5-5 h with 4 cores)"
	@echo "make sweep-smoke      smoke test of the production pipeline (2 replicates per point, T <= 250; a few minutes)"
	@echo "make o11-aw           O11 at age a_w: re-runs f2b_base, f2b_nulos and the s0 arms of f2b_rob (~10 min)"
	@echo "make gamma            Gamma without dilution: re-runs 710 runs into results/fisico_v3/ (~10-20 min) and rewrites fig2c_series.csv"
	@echo "make meanfield        fast ABM ensemble and kinetic mean-field sweeps read by the mean-field figures (~6 min)"
	@echo "make baseline         notebook baseline (alpha = 1.4, 100 replicates, T = 2000; ~2 min) and its survival report"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q tests

test-fast:
	$(PY) -m pytest -q tests -k "not sigterm and not statistically"

# Inputs of make figs: decompresses results_v2/**/{summary.csv,params.json}.gz and restores the reduced series
data fig-inputs:
	$(PY) scripts/figures/fig_inputs.py

figs: fig-inputs
	$(PY) scripts/figures/figs_paper.py
	$(PY) scripts/figures/fig_model_schematic.py
	$(PY) scripts/figures/figs_results.py
	$(PY) scripts/figures/fig_model_schematic_pre.py
	$(PY) scripts/figures/figs_pre.py

# make figs from a clean clone of HEAD; fails if any versioned file changes (figures, tables, results_numbers.json)
CLEAN_CLONE ?= $(or $(TMPDIR),/tmp)/u25_figs_clean_check
figs-clean-check:
	rm -rf $(CLEAN_CLONE) && git clone -q $(CURDIR) $(CLEAN_CLONE)
	$(MAKE) -C $(CLEAN_CLONE) figs
	cd $(CLEAN_CLONE) && git status --porcelain && test -z "$$(git status --porcelain)"
	@echo "figs-clean-check: figures, tables and results_numbers.json identical to HEAD"

check-latex:
	@command -v pdflatex >/dev/null && command -v bibtex >/dev/null || \
	  { echo "make paper needs pdflatex and bibtex (e.g. apt install texlive-latex-extra texlive-publishers texlive-bibtex-extra)"; exit 1; }

paper: paper-extended paper-pre

paper-extended: check-latex
	bash manuscript/extended/build.sh

paper-pre: check-latex
	bash manuscript/pre/build.sh

# Production runs (all results_v2/f2b_* stages) from scratch. Resumable: re-running the same command continues.
# RES=results_v2 re-runs into the versioned directory (run `make data` first so that finished stages are skipped).
# DEADLINE_H: wall-clock limit of the orchestrator; with a generous limit no stage is shortened or skipped.
RES ?= results_new
DEADLINE_H ?= 12
sweep:
	RES=$(RES) N_JOBS=$(NJOBS) DEADLINE_H=$(DEADLINE_H) FG=1 bash scripts/production/run_2b_v2.sh

sweep-smoke:
	RES=$(RES) N_JOBS=$(NJOBS) SMOKE=1 FG=1 bash scripts/production/run_2b_v2.sh

o11-aw: data
	$(PY) scripts/analysis/o11_aw.py run
	$(PY) scripts/analysis/o11_aw.py report

gamma: data
	nice -n 10 $(PY) scripts/analysis/gamma_v3.py results/fisico_v3/gamma_v3.parquet base,nulos,rob,barrido $(NJOBS)
	$(PY) scripts/analysis/analiza_gamma.py results/fisico_v3/gamma_v3.parquet results/fisico_v3/fig2c_series.csv

meanfield:
	cd results/meanfield && $(PY) $(CURDIR)/scripts/analysis/ensamble_rapido.py rw 40 2000
	$(PY) scripts/analysis/mf_kin.py sweep 0.5 8 > results/meanfield/kin_sweep_05_8.log
	$(PY) scripts/analysis/mf_kin.py sweep 1 5 > results/meanfield/kin_sweep_1_5.log

baseline:
	$(PY) -m u25 ensemble --n-reps 100 --T 2000 --base-seed 2026 --n-jobs $(NJOBS) --out results/baseline_alpha14 --quiet
	$(PY) -m pipeline baseline results/baseline_alpha14 --out results/baseline_alpha14/analisis

all: test figs paper

clean-pyc:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

clean-tex:
	cd manuscript/extended && rm -f *.aux *.bbl *.blg *.log *.out *.toc
	cd manuscript/pre && rm -f *.aux *.bbl *.blg *.log *.out *Notes.bib
