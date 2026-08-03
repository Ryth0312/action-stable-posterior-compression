.POSIX:
.PHONY: help smoke synthetic synthetic-quick synthetic-run paper-from-artifacts regen-postprocess regen-calibration manifest clean-twin

PYTHON ?= python3
export OMP_NUM_THREADS ?= 4
export PYTHONPATH := src

help:
	@echo "make smoke                 fast check: imports, the twin loads, its truth file parses"
	@echo "make synthetic-quick       T1: the whole chain on the twin at reduced resolution (minutes)"
	@echo "make synthetic             T1: the same chain at the article's resolution (hours)"
	@echo "make paper-from-artifacts  T0: rebuild the article figures and table source from artifacts"
	@echo "make regen-postprocess     T0: re-derive the post-processed artifacts (overwrites them)"
	@echo "make regen-calibration     T0: re-run the 300-replicate calibration study (hours; overwrites)"
	@echo "make manifest              rewrite MANIFEST.csv from what is in results/bayes"
	@echo
	@echo "Every non-default flag a committed artifact needs is passed explicitly by these targets"
	@echo "and is listed per artifact in MANIFEST.csv. Do not reconstruct commands from the PDF."

# --------------------------------------------------------------------------- T1: the twin
smoke:
	$(PYTHON) -c "import numpy, scipy, yaml; print('numpy', numpy.__version__)"
	$(PYTHON) -c "from cex_model import app_support as A; b = A.load_product('HLXSYN'); \
	              print('twin:', b.label, len(b.experiments), 'experiments')"
	$(PYTHON) -c "import json; d = json.load(open('results/bayes/synthetic_twin_truth.json')); \
	              print('twin truth: rmse_at_truth', d['expected_fit']['rmse_at_truth'])"
	@echo "smoke OK"

# The full-resolution run: the same n-steps and iteration budget the article's fits use. Hours of
# CPU. Use synthetic-quick first to check the chain end to end in minutes.
synthetic: NSTEPS = 300
synthetic: MAPITERS = 150
synthetic: synthetic-run

synthetic-quick: NSTEPS = 120
synthetic-quick: MAPITERS = 40
synthetic-quick: synthetic-run

synthetic-run:
	$(PYTHON) scripts/make_synthetic_twin.py
	$(PYTHON) scripts/bayes_calibrate.py --product HLXSYN --n-steps $(NSTEPS) --init config \
	    --map-iters $(MAPITERS) --engine laplace --predict --predict-samples 100 \
	    --out-dir results/bayes
	$(PYTHON) scripts/bayes_decision.py --product HLXSYN --n-steps $(NSTEPS) \
	    --map-iters $(MAPITERS) --mc-samples 200
	$(PYTHON) scripts/bayes_loeo.py --product HLXSYN --n-steps $(NSTEPS) --loeo --decision
	$(PYTHON) scripts/bayes_correlated_refit.py --product HLXSYN --kernel ou --n-steps $(NSTEPS) \
	    --map-iters $(MAPITERS) --optimizer lbfgs --rho-max 0.9 \
	    --c-param-out results/bayes/HLXSYN_c_param_correlated.json
	$(PYTHON) scripts/bayes_decision_window.py --products HLXSYN --n-steps $(NSTEPS) \
	    --n-samples 200 --n-candidates 24 --seed 0 --gate posterior_action
	@echo "synthetic OK -- results/bayes/HLXSYN_*"

# --------------------------------------------------------------------------- T0: from artifacts
# Torch-free. Needs the committed artifacts, which ship only in the confidential referee bundle.
# This target only READS them: it writes figures and table source, and overwrites no artifact.
paper-from-artifacts:
	mkdir -p docs
	$(PYTHON) scripts/make_aoas_figures.py --results results/bayes --out-dir docs
	$(PYTHON) scripts/make_supplement_tables.py > docs/_supplement_tables.tex
	@echo "paper-from-artifacts OK -- docs/fig_*.pdf and docs/_supplement_tables.tex"

# Deterministic re-derivations. Each OVERWRITES its own committed artifact in place, so run these
# to check reproduction, not as a prerequisite of paper-from-artifacts.
regen-postprocess:
	$(PYTHON) scripts/step2b_real_ladder.py
	$(PYTHON) scripts/step3_voi_nullity.py
	$(PYTHON) scripts/step3b_voi_prior_whitened.py
	$(PYTHON) scripts/step4b_r2_reverdict.py --in 'results/bayes/r2_paired_coupling_*.json' \
	    --spec 0.70 0.50
	@echo "regen-postprocess OK -- compare against MANIFEST.csv checksums"

# A 300-replicate simulation study, not a post-processing step: hours of CPU, and it overwrites the
# committed r6_* files that article Figure 2 and Supplement A Tables 5 and 6 are drawn from.
regen-calibration:
	$(PYTHON) scripts/step6_fullpipeline_calibration.py --n-replicates 300 --shape iso \
	    --disc-wdec 1.271 --bias-wdec 0.6 --n-post 200 --pool-size 8 --gibbs-iter 3000 \
	    --gibbs-burn 600 --gibbs-thin 6 --seed 0
	$(PYTHON) scripts/step6_fullpipeline_calibration.py --n-replicates 300 --shape aniso \
	    --aniso-ratio 6.0 --disc-wdec 1.271 --bias-wdec 0.6 --n-post 200 --pool-size 8 \
	    --gibbs-iter 3000 --gibbs-burn 600 --gibbs-thin 6 --seed 0

manifest:
	$(PYTHON) scripts/make_reproduction_manifest.py --target referee \
	    --results results/bayes --out MANIFEST.csv

# The twin's refit is pointed at its own c_param file so the shared, committed
# c_param_correlated.json is never modified by a twin run.
clean-twin:
	rm -f results/bayes/HLXSYN_*
