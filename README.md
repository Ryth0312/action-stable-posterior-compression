# Supplement B: code and reproduction archive

Companion to *Action-stable posterior compression for mechanistic calibration*.

## What is here, and what is not

This archive contains the mechanistic model, the differentiable solver, the calibration and
decision pipeline, and a **synthetic twin** the whole pipeline runs on end to end.

It does **not** contain the five antibody products behind the article and its supplements -- the
three applications and the two stress tests. Their elution and chromatographic purity traces are
proprietary, and so are their identities, so they are absent from
this archive rather than anonymised in it: no file name, no result file and no array key here names
a real product. Every number the article reports is derived from those traces; the artifacts behind
them are supplied to the editorial office confidentially during review, and afterwards under a
data-use agreement (see the article's data-availability statement).

## The synthetic twin

`HLXSYN` is a fictitious five-component cation-exchange product, generated forward from parameters
written down in `scripts/make_synthetic_twin.py` -- chosen, not fitted to anything. It has the same
column class, the same three-experiment design and the same fractionated observation format as the
real products, and it reproduces the identifiability structure the article's certificates address:
at its known truth a 10% move in the main component's characteristic charge multiplies the fit
residual by 9.8, while a 50% move in a basic component's steric factor multiplies it by 1.2.

    pip install numpy scipy pyyaml pandas openpyxl        # torch additionally for the fitting stages
    python scripts/make_synthetic_twin.py                 # regenerate the twin from its parameters
    python -c "import sys; sys.path.insert(0,'src'); \
               from cex_model import app_support as A; print(A.load_product('HLXSYN').label)"

`results/bayes/synthetic_twin_truth.json` holds the ground truth a fit should recover, the
observation model, and the residual a correct fit lands at -- 0.144, against a noise level of 0.080
on the scored points. The two differ because the observations are fraction means while the
likelihood compares to point values, exactly as for the real pooled fractions; a fit that reaches
0.080 is fitting the noise.

## Licence

MIT, see `LICENSE`.
