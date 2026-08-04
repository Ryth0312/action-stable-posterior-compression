#!/usr/bin/env python3
"""Build the Supplement B archive: the public deposit, and the confidential referee bundle.

The repository this is built from holds the three products' proprietary elution and chromatographic
purity traces, and their identifiers are confidential too, so the archive cannot be the repository,
a fork of it, or a mirror -- git history carries the data past any later deletion. It has to be a
curated export, and the export has to be checked rather than trusted.

Two targets:

  --target public     code, the synthetic twin, and nothing product-specific. This is the Zenodo
                      deposit. The five real products are DELETED from the product registry rather
                      than renamed, so no filename, JSON body or .npz key carrying an identifier
                      ever enters the tree. Deletion is what makes the guard below decidable.

  --target referee    the above plus every committed artifact behind a reported number, for the
                      editorial office. Not for publication. Still excludes the raw traces, which
                      are supplied separately by the data owner.

THE GUARD. After staging, the tree is scanned for the product identifiers -- in file names, in text
files, and in the key lists of .npz archives. Any hit fails the build. A rename that is merely
believed to be complete is not a control; a build that refuses to finish is.

  python scripts/make_supplement_b_archive.py --target public
  python scripts/make_supplement_b_archive.py --target referee --out /tmp/referee
"""
from __future__ import annotations
import argparse, ast, gzip, hashlib, json, re, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Identifiers that must not appear in the public archive. Longest first: a shorter code is a
# prefix of a longer one and a naive order would leave "HLXSYN" behind inside "HLXSYN".
CODES = ['HLXSYN']
CODE_RE = re.compile("|".join(CODES) + "|" + "|".join(c.lower() for c in CODES))

# The bare family prefix, which the codes above do not cover. It survives in identifiers such as
# _LAB_FRACTION_ROWS and in prose such as "the real LAB posteriors", and it names the owner's
# product line as surely as a full code does. Matched case-insensitively and excluding the twin.
BARE_RE = re.compile(r"(?i)lab(?!syn)")

TWIN = "HLXSYN"

# --- what ships ---------------------------------------------------------------------------------
# The whole package. A curated subset was tried and abandoned: the import closure of
# app_support + bayes + diffsolver reaches 50 modules and pulls in the surrogate, diffpeak and
# inverse subpackages anyway, so a subset is fragile without being smaller in any way that
# matters (1.7 MB). The scripts and the result artifacts are where curation earns its keep.
SRC_KEEP = ["cex_model"]

# Third-party dependencies the archive is allowed to import. Anything outside this list, the
# standard library and the staged tree is a module we failed to ship. Keeping it explicit means
# the list is also the archive's dependency manifest, and a new dependency cannot appear silently.
THIRD_PARTY = {
    "numpy", "scipy", "pandas", "matplotlib", "yaml", "openpyxl", "torch", "pyro", "requests",
    "tqdm", "sklearn", "statsmodels", "seaborn", "pytest",
}
# The scripts that write an artifact the article reports, plus the twin and the figures/tables.
SCRIPTS_KEEP = [
    "make_reproduction_manifest.py",
    "_calibration_metrics.py",      # path-loaded by step5 and step6
    "bayes_nuts_real.py",           # imported by bayes_correlated_nuts.py
    "bayes_decision_discrepancy_hier_audit.py",   # named by MANIFEST.csv stage S9

    "bayes_decision_discrepancy.py",
    "bayes_decision_gain.py",
    "bayes_discrepancy_prior_fold_audit.py",
    "bayes_predictive_closure.py",
    "bayes_restricted_fisher_certificate.py",
    "bayes_restricted_sigma_gain.py",
    "bayes_sigma_direction_audit.py",
    "bayes_timing.py",
    "step4b_r2_reverdict.py",
    "bayes_calibrate.py", "bayes_cmc_class.py", "bayes_correlated_nuts.py",
    "bayes_correlated_refit.py", "bayes_decision.py", "bayes_decision_discrepancy_hier.py",
    "bayes_decision_window.py", "bayes_empirical_convolution.py", "bayes_fisher_ablation.py",
    "bayes_loeo.py", "bayes_prior_sensitivity.py", "bayes_sigma_ablation.py",
    "bayes_solver_agnostic.py", "make_aoas_figures.py", "make_supplement_tables.py",
    "make_synthetic_twin.py", "step2_pmeet_alignment.py", "step2b_real_ladder.py",
    "step2d_meet_margin_certificate.py", "step2e_nonlinear_meet_certificate.py",
    "step3_voi_nullity.py", "step3b_voi_prior_whitened.py", "step4_r2_paired_coupling.py",
    "step5_decision_sbc.py", "step6_fullpipeline_calibration.py",
    "make_supplement_b_archive.py",
]
CONFIGS_TWIN = ["column_hlxsyn.yaml", "components_hlxsyn.yaml", "experiments_hlxsyn.yaml"]

MIT = """MIT License

Copyright (c) 2026 Zheyu Shi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


README = """# Supplement B: code and reproduction archive

Companion to *Action-stable posterior compression under practical nonidentifiability, with an
application to chromatography process design*.

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
    make smoke                                            # imports, twin loads, truth file parses
    make synthetic                                        # regenerate the twin and fit it (needs torch)
    python -c "import sys; sys.path.insert(0,'src'); \\
               from cex_model import app_support as A; print(A.load_product('HLXSYN').label)"

`results/bayes/synthetic_twin_truth.json` holds the ground truth a fit should recover, the
observation model, and the residual a correct fit lands at -- 0.144, against a noise level of 0.080
on the scored points. The two differ because the observations are fraction means while the
likelihood compares to point values, exactly as for the real pooled fractions; a fit that reaches
0.080 is fitting the noise.

## What produced what

`MANIFEST.csv` lists the twin pipeline: for each stage, the reproduction tier, the producing script,
**the flags that must be passed explicitly** (many defaults silently write a different file) and the
outputs that stage produces. It carries no checksums, because no product artifact ships here; the
confidential bundle's manifest binds each shipped artifact to an article object and hashes it. The
`Makefile` passes those flags for you -- `make help` lists the targets. Do not reconstruct commands
by hand.

## Licence

MIT, see `LICENSE`.
"""

GITIGNORE = """\
# Python bytecode. Importing the package writes these; they must not be committed.
__pycache__/
*.py[cod]

# Synthetic-twin fit outputs. `make synthetic` and `make synthetic-quick` write these and
# `make clean-twin` removes them. Only the twin's inputs and its ground truth are archived.
results/bayes/HLXSYN_*

# Local build and environment noise
.venv/
build/
dist/
.ipynb_checkpoints/
.DS_Store
"""


ZENODO = {
    "upload_type": "software",
    "title": "Supplement B to \"Action-stable posterior compression under practical "
             "nonidentifiability, with an application to chromatography process design\": "
             "code and reproduction archive",
    "description":
        "<p>Code and reproduction archive accompanying the article. Contains the mechanistic "
        "steric-mass-action model, the differentiable solver, the Bayesian calibration and "
        "decision pipeline, and a <b>synthetic twin</b> the whole pipeline runs on end to end "
        "on CPU.</p>"
        "<p>The five antibody products behind the article and its supplements---three applications and two stress tests---are <b>not</b> included. Their "
        "elution and chromatographic purity traces are proprietary, and so are their identities, "
        "so they are absent from this archive rather than anonymised within it: no file name, no "
        "result file and no array key names a real product. Artifacts derived from those traces "
        "are available from the data owner under a data-use agreement; see the article's data "
        "availability statement.</p>"
        "<p>The twin reproduces the identifiability structure the article's certificates address: "
        "at its known truth a 10% move in the main component's characteristic charge multiplies "
        "the fit residual by 9.8, while a 50% move in a basic component's steric factor "
        "multiplies it by 1.2.</p>",
    "creators": [
        {"name": "Shi, Zheyu", "affiliation": "Department of Computer Science, Brown University; "
                                              "Department of Technical Operations, Shanghai Henlius Biotech, Inc.",
         "orcid": "0009-0002-8843-8172"},
        {"name": "Ji, Dongni", "affiliation": "Department of Technical Operations, Shanghai Henlius Biotech, Inc."},
        {"name": "Xu, Weijie", "affiliation": "Department of Technical Operations, Shanghai Henlius Biotech, Inc."},
        {"name": "Han, Dongmei", "affiliation": "Department of Technical Operations, Shanghai Henlius Biotech, Inc."},
        {"name": "Hsu, Simon", "affiliation": "Department of Technical Operations, Shanghai Henlius Biotech, Inc."},
        {"name": "Chen, Ran", "affiliation": "Department of Technical Operations, Shanghai Henlius Biotech, Inc."},
    ],
    "license": "mit",
    "access_right": "open",
    "language": "eng",
    # No "version" key on purpose. Zenodo takes the version from the GitHub release tag; a
    # literal here would override the tag and silently ship a stale version on the next release.
    "keywords": [
        "Bayesian calibration", "computer model calibration", "practical nonidentifiability",
        "decision-theoretic model reduction", "posterior compression",
        "steric mass action", "ion-exchange chromatography", "differentiable simulation",
        "reproducible research",
    ],
    # No related_identifiers yet: the article has no DOI. Once it does, add
    #   {"relation": "isSupplementTo", "identifier": "<article DOI>", "scheme": "doi"}
    # and re-upload this file.
}


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        shutil.copy2(src, dst)


def stage_public(out: Path) -> list[str]:
    """Code, the twin, and nothing that names a product."""
    notes = []
    for rel in SRC_KEEP:
        s = ROOT / "src" / rel
        if not s.exists():
            notes.append(f"MISSING src/{rel}")
            continue
        _copy(s, out / "src" / rel)
    for name in SCRIPTS_KEEP:
        s = ROOT / "scripts" / name
        if not s.exists():
            notes.append(f"MISSING scripts/{name}")
            continue
        _copy(s, out / "scripts" / name)
    for name in CONFIGS_TWIN:
        _copy(ROOT / "configs" / name, out / "configs" / name)
    _copy(ROOT / "data" / "synthetic_twin", out / "data" / "synthetic_twin")
    _copy(ROOT / "results" / "bayes" / "synthetic_twin_truth.json",
          out / "results" / "bayes" / "synthetic_twin_truth.json")
    # The root Makefile is the referee tree's. The public archive has no product artifacts, so its
    # manifest target must generate the public manifest; --target referee would fail the coverage
    # check and leave MANIFEST.csv untouched.
    mk = (ROOT / "Makefile").read_text(encoding="utf-8").replace(
        "make_reproduction_manifest.py --target referee",
        "make_reproduction_manifest.py --target public")
    (out / "Makefile").write_text(mk, encoding="utf-8")
    (out / "LICENSE").write_text(MIT, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".zenodo.json").write_text(json.dumps(ZENODO, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    return notes


def stage_referee(out: Path) -> list[str]:
    """The public tree plus every committed artifact behind a reported number."""
    notes = stage_public(out)
    src_res = ROOT / "results" / "bayes"
    n = 0
    for f in sorted(src_res.iterdir()):
        if f.is_dir() or f.suffix not in (".json", ".npz"):
            continue
        _copy(f, out / "results" / "bayes" / f.name)
        n += 1
    for name in ("column", "components", "experiments"):
        for f in sorted((ROOT / "configs").glob(f"{name}_*.yaml")):
            _copy(f, out / "configs" / f.name)
    notes.append(f"referee bundle: staged {n} artifacts from results/bayes")
    return notes


def _strip_registry(text: str) -> tuple[str, int]:
    """Drop the real products from ``_BUILTIN_PRODUCTS``, leaving only the twin.

    Each entry is one ``"CODE": {...},`` block at four-space indent, so the block can be found by
    its opening line and closed on the first line that ends the brace at that indent. Removing the
    entries -- rather than renaming them -- is what makes the identifier guard decidable.
    """
    lines, out, dropped, i = text.splitlines(keepends=True), [], 0, 0
    entry = re.compile(r'^ {4}"(' + "|".join(CODES) + r')":\s*\{')
    while i < len(lines):
        if entry.match(lines[i]):
            depth = 0
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0:
                    break
            dropped += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out), dropped


def _strip_dead_readers(text: str) -> tuple[str, int]:
    """Reduce the raw-data reader chain to the twin's branch alone.

    The whole ``if/elif/.../else`` on ``spec["data_source"]`` is replaced by the ``synthetic``
    branch promoted to a plain ``if``. Deleting branches one at a time would leave an orphan
    ``elif`` and a dead ``else`` whose comments name the products, so the chain is rewritten as a
    unit and the result is parsed before it is accepted.
    """
    lines = text.splitlines(keepends=True)
    head = re.compile(r'^(\s*)if spec\["data_source"\] == ')
    branch = re.compile(r'^\s*(?:el)?if spec\["data_source"\] == "([a-z0-9_]+)":')

    start = next((i for i, l in enumerate(lines) if head.match(l)), None)
    if start is None:
        return text, 0
    indent = len(head.match(lines[start]).group(1))

    end = start + 1
    while end < len(lines):                              # the chain's own elif/else sit at the
        s = lines[end]                                   # same indent, so only a line at that
        if not s.strip():                                # indent that continues neither ends it
            end += 1
            continue
        ind = len(s) - len(s.lstrip())
        if ind > indent:
            end += 1
            continue
        if ind == indent and re.match(r'^\s*(elif |else:)', s):
            end += 1
            continue
        break

    keep, cur, n = [], None, 0
    for i in range(start, end):
        m = branch.match(lines[i])
        if m or re.match(r'^\s*else:', lines[i]):
            cur = m.group(1) if m else None
            n += 1
            if cur == "synthetic":
                keep.append(re.sub(r'^(\s*)(?:el)?if ', r'\1if ', lines[i]))
            continue
        if cur == "synthetic":
            keep.append(lines[i])
    if not keep:
        raise SystemExit("the synthetic reader branch was not found; the archive would ship no reader")
    return "".join(lines[:start] + keep + lines[end:]), n - 1


def deidentify(tree: Path) -> list[str]:
    """Remove every product identifier from the staged public tree.

    Two mechanisms, in this order. The registry entries and the readers that serve them are
    REMOVED, because a public archive that names five products it does not ship is worse than one
    that does not mention them. Everything left -- docstring examples, argparse defaults, product
    lists in driver scripts -- is a reference to a product the archive no longer has, so it is
    repointed at the twin.
    """
    notes = []
    app = tree / "src" / "cex_model" / "app_support.py"
    if app.exists():
        t = app.read_text(encoding="utf-8")
        t, n_reg = _strip_registry(t)
        t, n_rd = _strip_dead_readers(t)
        app.write_text(t, encoding="utf-8")
        notes.append(f"app_support.py: dropped {n_reg} product registry entries, {n_rd} raw-data readers")

    n_files = n_subs = 0
    for p in sorted(tree.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".md", ".yaml", ".yml", ".json", ".csv"):
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        new = CODE_RE.sub(lambda m: TWIN if m.group(0)[0].isupper() else TWIN.lower(), t)
        if new != t:
            p.write_text(new, encoding="utf-8")
            n_files += 1
            n_subs += len(CODE_RE.findall(t))
    notes.append(f"repointed {n_subs} remaining references to {TWIN} across {n_files} files")

    # Three sites the code substitution above cannot repair, because the right edit is not a
    # rename. Each is applied by exact match so that a change upstream fails loudly here rather
    # than silently leaving the original text in the archive.
    repairs = [
        # A comment describing the redaction itself, which tells a reader both that codes were
        # removed and what they looked like.
        ("src/cex_model/bayes/plots.py",
         "# Double-blind label map for figure TITLES (gated by env BLIND_LABELS=1; default OFF ->\n"
         "# original output; filenames are unaffected). LAB0X are real, searchable product codes.\n",
         "# Double-blind label map for figure TITLES (gated by env BLIND_LABELS=1; default OFF ->\n"
         "# original output; filenames are unaffected).\n"),
        ("scripts/step3_voi_nullity.py",
         "Torch-free. Verifies, on the real LAB posteriors + committed decision Jacobians:",
         "Torch-free. Verifies, on the committed posteriors + decision Jacobians:"),
        # Per-component-count starting templates. The blanket substitution collapsed three distinct
        # product configs onto the twin's, so a three- or four-component fit would silently get a
        # five-component template. Only the twin ships, so only its size has a template.
        ("src/cex_model/app_support.py",
         '_TEMPLATES = {\n    5: "configs/components_hlxsyn.yaml",\n'
         '    4: "configs/components_hlxsyn.yaml",\n    3: "configs/components_hlxsyn.yaml",',
         '_TEMPLATES = {\n    5: "configs/components_hlxsyn.yaml",'),
    ]
    for rel, old, new in repairs:
        p = tree / rel
        t = p.read_text(encoding="utf-8")
        if old not in t:
            raise SystemExit(f"de-identification repair no longer applies in {rel}; the source "
                             f"changed and the archive would ship the original text")
        p.write_text(t.replace(old, new, 1), encoding="utf-8")
    notes.append(f"applied {len(repairs)} targeted repairs a rename cannot make")

    # Whatever bare prefix is left is in an identifier, where a consistent rename is safe.
    n_bare = 0
    for p in sorted(tree.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".md", ".yaml", ".yml", ".json", ".csv"):
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        new = BARE_RE.sub(lambda m: "LAB" if m.group(0).isupper() else "lab", t)
        if new != t:
            n_bare += len(BARE_RE.findall(t))
            p.write_text(new, encoding="utf-8")
    notes.append(f"renamed {n_bare} bare family-prefix tokens")

    n_col = _dedupe_collapsed(tree)
    notes.append(f"de-duplicated {n_col} literals the rename collapsed")

    bad = []                                   # a transform that breaks the source must not ship
    for p in sorted(tree.rglob("*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError as e:
            bad.append(f"{p.relative_to(tree)}:{e.lineno}: {e.msg}")
    if bad:
        raise SystemExit("de-identification broke the source:\n  " + "\n  ".join(bad))
    notes.append(f"parsed {len(list(tree.rglob('*.py')))} staged python files, all clean")
    return notes


def write_manifest(out: Path, target: str) -> list[str]:
    """MANIFEST.csv, generated from the STAGED tree so its checksums are of the shipped bytes."""
    gen = ROOT / "scripts" / "make_reproduction_manifest.py"
    cmd = [sys.executable, str(gen), "--target", target, "--out", str(out / "MANIFEST.csv")]
    if target == "referee":
        cmd += ["--results", str(out / "results" / "bayes")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        sys.exit(f"manifest generation failed (exit {r.returncode})")
    return [r.stdout.strip()]



def _literal_spans(text):
    """Offending literals as ``(start, end, replacement)``, outermost first.

    The bare-prefix rename maps several product codes onto one, so any literal that keyed or
    listed products becomes a dict with duplicate keys (Python silently keeps the last) or a
    sequence of identical strings. Both are legal Python, so the syntax gate cannot see them.
    Only the twin ships, so one entry is the correct content.
    """
    tree = ast.parse(text)
    off, acc = [0], 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        off.append(acc)
    def span(n):
        return off[n.lineno - 1] + n.col_offset, off[n.end_lineno - 1] + n.end_col_offset

    edits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict) and len(n.keys) >= 2:
            if any(k is None or not isinstance(k, ast.Constant) for k in n.keys):
                continue
            seen, keep = set(), []
            for k, v in zip(n.keys, n.values):
                if k.value in seen:
                    continue
                seen.add(k.value)
                keep.append((k, v))
            if len(keep) < len(n.keys):
                body = ", ".join(f"{ast.unparse(k)}: {ast.unparse(v)}" for k, v in keep)
                edits.append((*span(n), "{" + body + "}"))
        elif isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) >= 2:
            vals = [e.value for e in n.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(vals) == len(n.elts) and len(set(vals)) == 1:
                one = ast.unparse(n.elts[0])
                edits.append((*span(n), f"[{one}]" if isinstance(n, ast.List) else f"({one},)"))

    # Drop anything nested inside another edit; the outer rewrite subsumes it.
    edits.sort(key=lambda e: (e[0], -e[1]))
    out = []
    for e in edits:
        if out and e[0] >= out[-1][0] and e[1] <= out[-1][1]:
            continue
        out.append(e)
    return out


def _dedupe_collapsed(tree: Path) -> int:
    n = 0
    for p in sorted(tree.rglob("*.py")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for _ in range(8):                      # a rewrite can expose an enclosing literal
            edits = _literal_spans(text)
            if not edits:
                break
            for a, b, r in sorted(edits, reverse=True):
                text = text[:a] + r + text[b:]
            n += len(edits)
        p.write_text(text, encoding="utf-8")
    return n



def check_imports(tree: Path) -> list[str]:
    """Every first-party module a staged script needs must be staged too.

    ``ast.parse`` proves a file is syntactically valid, which is orthogonal: a script importing a
    sibling that was never added to SCRIPTS_KEEP parses perfectly and fails at run time. Two ways
    a dependency hides -- a normal import, and a path load through
    ``spec_from_file_location(..., parent / "x.py")`` -- so both are checked. Third-party and
    standard-library names are resolved against the building environment and are the referee's
    problem, not the archive's.
    """
    import importlib.util
    scripts = {p.stem for p in (tree / "scripts").glob("*.py")}
    src = tree / "src"
    missing = []
    for p in sorted(tree.rglob("*.py")):
        rel = p.relative_to(tree)
        text = p.read_text(encoding="utf-8", errors="ignore")
        for n in ast.walk(ast.parse(text)):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module.split(".")[0]]
            for name in names:
                if name in sys.stdlib_module_names or name in scripts or name in THIRD_PARTY:
                    continue
                if (src / name).is_dir() or (src / f"{name}.py").is_file():
                    continue
                try:
                    if importlib.util.find_spec(name) is not None:
                        continue
                except (ImportError, ValueError):
                    pass
                missing.append(f"{rel}:{n.lineno}: imports '{name}', which is not in the archive")
        # A path load hides in a plain string. Read them off the AST so that prose in a docstring
        # or a comment cannot trip the gate.
        staged = {q.name for q in tree.rglob("*.py")}
        for n in ast.walk(ast.parse(text)):
            if not (isinstance(n, ast.Constant) and isinstance(n.value, str)):
                continue
            v = n.value
            if (v.endswith(".py") and len(v) > 3 and "*" not in v and "/" not in v
                    and " " not in v and v not in staged):
                missing.append(f"{rel}:{n.lineno}: names '{v}', which is not in the archive")
    return missing


def check_collapsed(tree: Path) -> list[str]:
    """No literal may survive with duplicate keys or an all-identical element run."""
    hits = []
    for p in sorted(tree.rglob("*.py")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for a, _b, _r in _literal_spans(text):
            line = text.count("\n", 0, a) + 1
            hits.append(f"{p.relative_to(tree)}:{line}: literal collapsed by the rename")
    return hits


def _write_zip(path: Path, out: Path, files: list[Path]):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, Path(out.name) / p.relative_to(out))


def _write_tar(path: Path, out: Path, files: list[Path]):
    """Deterministic tar.gz: sorted entries, no mtimes, no uid/gid, one top-level directory.

    The gzip header carries an mtime of its own, which ``tarfile.open(..., "w:gz")`` fills with the
    build time -- so scrubbing the tar members alone leaves the archive differing byte for byte
    between builds of an identical tree. Compress through an explicit GzipFile at mtime zero.
    """
    def scrub(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 0
        return ti
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as t:
                for p in sorted(files):
                    t.add(p, arcname=str(Path(out.name) / p.relative_to(out)), filter=scrub)


def scan_for_codes(tree: Path) -> list[str]:
    """Every place a product identifier survives. Empty list == the tree is clean."""
    hits = []
    for p in sorted(tree.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(tree)
        if CODE_RE.search(p.name) or BARE_RE.search(p.name):
            hits.append(f"filename: {rel}")
        if p.suffix == ".npz":
            try:
                with np.load(p, allow_pickle=False) as z:
                    for k in z.files:
                        if CODE_RE.search(k) or BARE_RE.search(k):
                            hits.append(f"npz key: {rel} :: {k}")
            except Exception as e:                       # a bad archive is itself a build failure
                hits.append(f"unreadable npz: {rel} ({e})")
            continue
        if p.suffix in (".py", ".json", ".yaml", ".yml", ".txt", ".md", ".csv", ".tex", ""):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for rx in (CODE_RE, BARE_RE):
                for m in rx.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"text: {rel}:{line} :: {m.group(0)}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["public", "referee"], default="public")
    ap.add_argument("--out", default=None, help="staging directory; default build/supplement_b_<target>")
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    ap.add_argument("--tar", action="store_true", help="also write <out>.tar.gz (the Zenodo upload)")
    ap.add_argument("--allow-codes", action="store_true",
                    help="referee target only: skip the identifier guard, which that bundle is "
                         "exempt from because it is confidential and not for publication")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / "build" / f"supplement_b_{a.target}"
    if out.exists():
        # The staging directory is wiped on every build. If someone has made it the working tree of
        # a clone -- the public archive is also a git repository -- that would delete the clone.
        if (out / ".git").exists():
            sys.exit(f"refusing to build into {out}: it is a git working tree, and this build "
                     f"would delete it.\n"
                     f"Build somewhere else with --out, then sync into the clone, e.g.\n"
                     f"  python scripts/make_supplement_b_archive.py --target {a.target} "
                     f"--out /tmp/archive\n"
                     f"  rsync -a --delete --exclude .git --exclude __pycache__ "
                     f"/tmp/archive/ {out}/")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("=" * 92)
    print(f"SUPPLEMENT B ARCHIVE  --  target={a.target}  ->  {out}")
    print("=" * 92)
    notes = stage_public(out) if a.target == "public" else stage_referee(out)
    if a.target == "public":
        notes += deidentify(out)
    notes += write_manifest(out, a.target)
    for n in notes:
        print(f"  {n}")

    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"  staged {len(files)} files, {size/1e6:.1f} MB")

    for label, fn in (("import", check_imports), ("collapsed-literal", check_collapsed)):
        problems = fn(out)
        if problems:
            print(f"\n  {label.upper()} GATE FAILED: {len(problems)} problem(s)")
            for q in problems[:30]:
                print(f"    {q}")
            if len(problems) > 30:
                print(f"    ... and {len(problems)-30} more")
            sys.exit(1)
        print(f"  {label} gate passed")

    guard = a.target == "public" or not a.allow_codes
    if guard:
        hits = scan_for_codes(out)
        if hits:
            print(f"\n  GUARD FAILED: {len(hits)} product identifiers survive in the staged tree")
            for h in hits[:40]:
                print(f"    {h}")
            if len(hits) > 40:
                print(f"    ... and {len(hits)-40} more")
            print("\n  The archive was NOT written. Remove the offending content, or for the referee")
            print("  bundle pass --allow-codes, which is exempt because it is not for publication.")
            sys.exit(1)
        print("  guard passed: no product identifier in any filename, text file or .npz key")
    else:
        print("  guard SKIPPED (--allow-codes); this bundle must not be published")

    for want, suffix, write in (
        (a.zip, ".zip", _write_zip),
        (a.tar, ".tar.gz", _write_tar),
    ):
        if not want:
            continue
        path = out.parent / (out.name + suffix)
        write(path, out, files)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  wrote {path}  ({path.stat().st_size/1e6:.1f} MB)")
        print(f"  sha256 {digest}")
    print("=" * 92)


if __name__ == "__main__":
    main()
