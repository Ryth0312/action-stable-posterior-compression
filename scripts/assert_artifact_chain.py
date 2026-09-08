"""Refuse to build the article's tables when the artifact chain is inconsistent.

Every check here corresponds to a way the committed artifacts have actually gone out of step:
a product refit left uncapped while its folds were capped, a predictive scan reading one hierarchy
draw set while a certificate read another, a certificate assembled before the artifact it reads its
deployed probabilities out of, or a manifest checksum left behind by a re-run.

Usage:  python scripts/assert_artifact_chain.py [--results results/bayes] [--manifest MANIFEST.csv]
Exit status is 1 on the first failure, so `make` stops before writing a table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np

PRODUCTS = ['HLXSYN']
RHO_MAX = 0.9
CANONICAL_DRAWS = "results/bayes/hier_draws_capB.npz"
DEPLOYED_POSTERIOR = "correlated_posterior"

_failures: list[str] = []


def check(ok, what, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        _failures.append(f"{what}: {detail}")


def _load(p):
    return json.loads(pathlib.Path(p).read_text())


def rho_cap_is_uniform(res):
    """Every canonical refit and every leave-one-experiment-out fold at the same prespecified cap."""
    for pid in PRODUCTS:
        for tag, name in (("refit", f"{pid}_correlated.json"), ("folds", f"{pid}_correlated_loeo.json")):
            rho = _load(res / name)["hyper"]["rho"]
            check(rho == RHO_MAX, f"rho cap  {pid:14} {tag}", f"rho = {rho}, expected {RHO_MAX}")


def one_canonical_draw_set(res):
    """Every object that integrates over the hierarchy names the same committed draw set."""
    named = []
    for pid in PRODUCTS:
        d = _load(res / f"{pid}_decision_window_predictive_hier.json")
        named.append((f"{pid} deployed scan", d["provenance"]["hier_draws"]))
    named.append(("threshold certificate", _load(res / "meet_margin_certificate.json")[0]["hier_draws"]))
    named.append(("linearisation certificate", _load(res / "predictive_linearisation_certificate.json")["hier_draws"]))
    for f in sorted(res.glob("r2_paired_coupling_*correlated*_op*.json")):
        flips = _load(f)[0].get("predictive_flips")
        if flips:
            named.append((f.name[:52], flips["hier_draws"]))
    for what, path in named:
        check(path == CANONICAL_DRAWS, f"draw set  {what}", f"names {path}")


def deployed_posterior_is_the_capped_refit(res):
    """The candidate-level certificates are taken at the refit the deployment reads use."""
    for f in sorted(res.glob("r2_paired_coupling_*correlated*_op*.json")):
        post = _load(f)[0].get("posterior")
        check(post == DEPLOYED_POSTERIOR, f"posterior  {f.name[:52]}", f"is {post}")


def candidate_pool_agrees(res):
    """The pool the certificates score is the pool the scans deploy, candidate by candidate."""
    ref = None
    for pid in PRODUCTS:
        scan = [r["op"] for r in _load(res / f"{pid}_decision_window_predictive_hier.json")["rows"]]
        cert = [r["op"] for r in
                next(b for b in _load(res / "meet_margin_certificate.json") if b["product"] == pid)["rows"]]
        check(np.allclose(np.asarray(scan, float), np.asarray(cert, float)),
              f"pool      {pid:14} scan vs threshold certificate",
              f"{len(scan)} vs {len(cert)} candidates, or coordinates differ")
        ref = ref or len(scan)


def paired_draws_match_their_certificate(res):
    """The stored draws are the ones the conditional certificate was computed from."""
    spec = np.array([0.70, 0.50])
    for f in sorted(res.glob("r2_paired_coupling_*correlated*_op*.json")):
        d = _load(f)[0]
        npz = res / f"paired_draws_{d['product']}_op{d['op'][0]:g}.npz"
        if not npz.exists():
            check(False, f"paired    {npz.name}", "absent")
            continue
        z = np.load(npz)
        X, Y, tol = z["X"], z["Y"], z["tol"]
        pX = float(np.mean(np.all(X / tol >= spec / tol, axis=1)))
        pY = float(np.mean(np.all(Y / tol >= spec / tol, axis=1)))
        ok = (abs(pX - d["P_X_meet_nonlinear"]) < 1e-12
              and abs(pY - d["P_Y_meet_linear"]) < 1e-12
              and np.allclose(z["op"], d["op"]))
        check(ok, f"paired    {npz.name[:52]}", "draws do not reproduce the stored probabilities")


def flip_certificate_is_downstream(res):
    """bayes_predictive_flip_certificate.py reads its deployed probabilities out of the
    linearisation certificate, so running it first silently centres every interval on stale values."""
    lin = {(r["product"], round(r["op"][0], 4)): r
           for r in _load(res / "predictive_linearisation_certificate.json")["rows"]}
    for r in _load(res / "predictive_flip_certificate.json")["rows"]:
        key = (r["product"] if "product" in r else None, round(r["loading"], 4))
        src = lin.get(key)
        if src is None:
            check(False, f"flip      {r['name']} @{r['loading']:.4f}", "no matching linearisation row")
            continue
        check(abs(src["P_meet_deployed"] - r["P_meet_deployed"]) < 1e-12,
              f"flip      {r['name']} @{r['loading']:.4f} deployed probability",
              f"{r['P_meet_deployed']:.6f} against {src['P_meet_deployed']:.6f} upstream; "
              "re-run bayes_predictive_flip_certificate.py after the linearisation certificate")


def regret_shift_is_downstream(res):
    """bayes_pairwise_regret_shift.py reads its reference action out of route_b_predictive_action.json,
    so running it first anchors every regret at the previous fit's minimizer and the paired column
    certifies an action the tables no longer name."""
    rb = {b["product"]: b for b in _load(res / "route_b_predictive_action.json")}
    pr = _load(res / "pairwise_regret_shift.json")
    for b in (pr["products"] if isinstance(pr, dict) and "products" in pr else pr):
        pid = b["product"]
        src = rb.get(pid)
        if src is None:
            check(False, f"regret    {pid:14}", "no matching route-b block")
            continue
        check(b.get("a_star") == src.get("a_star"),
              f"regret    {pid:14} reference action",
              f"a* = {b.get('a_star')} against {src.get('a_star')} upstream; "
              "re-run bayes_pairwise_regret_shift.py after bayes_route_b_predictive_action.py")


def action_stability_is_downstream(res):
    """bayes_multistart_action_stability.py re-scores the deployed pool at each restart's law. Its
    restart 0 is the canonical fit, so its best in-domain probability must be the one the deployed
    scan reports; a drift there means the two were built from different fits."""
    f = res / "multistart_action_stability.json"
    if not f.exists():
        check(True, "action    not present, skipped")
        return
    for b in _load(f):
        pid = b["product"]
        lo, hi = b["deployment_domain"]
        scan = _load(res / f"{pid}_decision_window_predictive_hier.json")["rows"]
        dep = max(r["p_meet"] for r in scan if lo <= r["op"][0] <= hi)
        r0 = b["rows"][0]["p_max_deployment"]
        check(abs(r0 - dep) <= 1e-6 * max(1.0, abs(dep)) + 1e-8,
              f"action    {pid:14} restart 0 against the deployed scan",
              f"{r0:.6g} against {dep:.6g}; re-run bayes_multistart_action_stability.py")
        n_ms = _load(res / f"{pid}_correlated_multistart.json")["n_restarts"]
        check(b["n_restarts"] == n_ms, f"action    {pid:14} restart count",
              f"{b['n_restarts']} scored against {n_ms} run")


def basin_closure_is_downstream(res):
    """bayes_basin_closure.py reads the deployed pool at each restart's law with its own Jacobians. Its
    restart 0 is the canonical fit, so under the full law it must select what the deployed scan and
    the action-stability audit select; and its segment endpoints must reproduce the objectives the
    multistart recorded, or the segment was evaluated on a different objective."""
    for f in sorted(res.glob("*_basin_closure.json")):
        b = _load(f)
        pid = b["product"]
        lo, hi = b["deployment_domain"]
        scan = _load(res / f"{pid}_decision_window_predictive_hier.json")["rows"]
        dep = max(r["p_meet"] for r in scan if lo <= r["op"][0] <= hi)
        r0 = next(r for r in b["rows"] if r["restart"] == 0)["law"]["full"]
        check(abs(r0["p_max_deployment"] - dep) <= 1e-6 * max(1.0, abs(dep)) + 1e-8,
              f"basin     {pid:14} restart 0 against the deployed scan",
              f"{r0['p_max_deployment']:.6g} against {dep:.6g}; re-run bayes_basin_closure.py --stage jacobians")
        act = res / "multistart_action_stability.json"
        if act.exists():
            a0 = next(x for x in _load(act) if x["product"] == pid)["rows"][0]
            check(abs(r0["loading_a_star"] - a0["loading_a_star"]) < 1e-6,
                  f"basin     {pid:14} restart 0 minimizer against the action-stability audit",
                  f"{r0['loading_a_star']:.4f} against {a0['loading_a_star']:.4f}")
        seg = b.get("segment")
        if seg:
            check(max(seg["endpoint_abs_error"]) <= 1e-6,
                  f"basin     {pid:14} segment endpoints reproduce the recorded objectives",
                  f"errors {seg['endpoint_abs_error']}; the segment objective differs from the multistart's")
    if not list(res.glob("*_basin_closure.json")):
        check(True, "basin     not present, skipped")


def manifest_checksums_hold(manifest, root):
    """A checksum left behind by a re-run means the row no longer describes the shipped bytes."""
    if not manifest.exists():
        check(True, "manifest  not present, skipped")
        return
    stale, absent = [], 0
    for row in csv.DictReader(manifest.open()):
        out, sha = row.get("output", ""), row.get("sha256", "")
        if not out or len(sha) != 64:
            continue
        p = root / out
        if not p.exists():
            absent += 1
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            stale.append(out)
    check(not stale, f"manifest  checksums ({absent} rows not present locally)",
          "stale for " + ", ".join(stale[:4]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/bayes")
    ap.add_argument("--manifest", default="MANIFEST.csv")
    a = ap.parse_args()
    res = pathlib.Path(a.results)
    root = pathlib.Path(a.manifest).resolve().parent

    print("artifact chain:")
    rho_cap_is_uniform(res)
    one_canonical_draw_set(res)
    deployed_posterior_is_the_capped_refit(res)
    candidate_pool_agrees(res)
    paired_draws_match_their_certificate(res)
    flip_certificate_is_downstream(res)
    regret_shift_is_downstream(res)
    action_stability_is_downstream(res)
    basin_closure_is_downstream(res)
    manifest_checksums_hold(pathlib.Path(a.manifest), root)

    if _failures:
        print(f"\n{len(_failures)} check(s) failed; the article tables are not rebuilt.", file=sys.stderr)
        return 1
    print("\nartifact chain consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
