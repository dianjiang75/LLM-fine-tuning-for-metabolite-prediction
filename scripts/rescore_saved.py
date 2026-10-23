"""Recompute cached evaluation artifacts without model calls.

Requires the supplied data/test_set.csv and prepared data/processed/dev.jsonl.
Run with --write to refresh the comparison, prior sweep and bootstrap artifacts.
Original trained-arm results in docs/arm_results.csv remain the recorded experiments.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import decoding as dec, evaluation as ev
from src.inference import compile_rules

GRID = [0, .25, .5, .75, 1, 1.5, 2, 3, 4, 8, 1e6]
SEED, REPLICATES = 20261021, 20000


def read_json(path):
    return json.loads((ROOT / path).read_text())


def site_enumerator():
    """Load only the two pure functions from the saved prior notebook."""
    notebook = read_json("SOM/som_prior.ipynb")
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            nodes.extend(n for n in ast.parse("".join(cell["source"])).body
                         if isinstance(n, ast.FunctionDef)
                         and n.name in {"largest_fragment", "enumerate_with_sites"})
    if len(nodes) != 2:
        raise ValueError("Expected the prior notebook's two enumeration functions")
    import collections
    namespace = {"Chem": Chem, "collections": collections,
                 "COMPILED": [(reaction, keep) for _, reaction, keep in compile_rules()]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "site_functions", "exec"), namespace)
    return namespace["enumerate_with_sites"]


def hits(predictions, pairs, k=5, key=ev.standardize):
    keys = {p: {key(c) for c in cs[:k]} for p, cs in predictions.items()}
    return np.array([key(t) in keys[p] for p, t in pairs], dtype=int)


def paired_interval(frame, a, b, label, indices):
    grouped = frame.groupby("cluster", sort=True).agg(["sum", "size"])
    sizes = grouped[(a, "size")].to_numpy()
    differences = (grouped[(a, "sum")] - grouped[(b, "sum")]).to_numpy()
    values = 100 * differences[indices].sum(1) / sizes[indices].sum(1)
    lo, hi = np.quantile(values, [.025, .975])
    return {"label": label, "a": a, "b": b,
            "delta_pp": float(100 * differences.sum() / sizes.sum()),
            "ci_low": float(lo), "ci_high": float(hi),
            "a_recall": float(frame[a].mean()), "b_recall": float(frame[b].mean()),
            "parents_net_improved": int(sum(differences > 0)),
            "parents_net_worsened": int(sum(differences < 0)),
            "parents_net_tied": int(sum(differences == 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh derived CSV/JSON artifacts")
    args = parser.parse_args()
    required = ["data/test_set.csv", "data/processed/dev.jsonl"]
    missing = [p for p in required if not (ROOT / p).is_file()]
    if missing:
        parser.error("This rescore requires the supplied data and prepared dev split: "
                     + ", ".join(missing) + ". Run scripts/check_offline.py for public replay checks.")

    test = pd.read_csv(ROOT / required[0], keep_default_na=False)
    test_pairs = [(ev.canonical(p), ev.canonical(t))
                  for p, t in zip(test.parent_smiles, test.metabolite_smiles)]
    cache = read_json("SOM/som_candidate_scores.json")
    baseline = {p: dec.rank_candidates(list(s), list(s.values()), p, 5).predictions
                for p, s in cache["test"].items()}
    endpoint_hits = {name: int(hits(baseline, test_pairs, key=key).sum()) for name, key in
                     [("exact", ev.canonical),
                      ("nostereo", lambda s: ev.canonical(s, False)),
                      ("standardized", ev.standardize)]}
    print("Fixed cached top-five hits:", endpoint_hits, flush=True)

    fps = {}
    def fingerprint_key(smiles):
        if smiles not in fps:
            fps[smiles] = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)).ToBitString()
        return fps[smiles]
    known = defaultdict(set)
    for parent, target in test_pairs:
        known[parent].add(fingerprint_key(target))
    published = pd.read_csv(ROOT / "docs/published_comparison.csv", index_col="system")
    label = "this pipeline (+GOSTAR mammalian ranker)"
    comparison = {
        "recall@5": float(hits(baseline, test_pairs, key=fingerprint_key).mean()),
        "precision@5": float(np.mean([sum(fingerprint_key(q) in known[p] for q in baseline[p]) / 5
                                      for p in known])),
        "any_hit@5": float(np.mean([any(fingerprint_key(q) in known[p] for q in baseline[p])
                                    for p in known]))}
    for key, value in comparison.items():
        published.loc[label, key] = value
    published.loc[label, "matching"] = "RDKit RDKFingerprint equality"
    published.loc[label, "source"] = "scripts/rescore_saved.py; SOM/som_candidate_scores.json"
    print("RDKFingerprint comparison:", comparison, flush=True)

    enum = site_enumerator()
    candidates = {p: enum(p) for split in cache.values() for p in split}
    if not all(set(candidates[p]) == set(s) for split in cache.values() for p, s in split.items()):
        raise ValueError("Cached candidate lists differ from current enumeration")
    test_sites = {p: set(s) for p, s in read_json("SOM/som_predicted_sites.json")["sites"].items()}
    dev_sites = {p: set(s) for p, s in read_json("SOM/som_predicted_sites_dev.json").items()}
    dev = defaultdict(set)
    for line in (ROOT / required[1]).read_text().splitlines():
        row = json.loads(line)
        dev[row["parent"]].add(row["target"])
    dev_pairs = [(p, t) for p in sorted(cache["dev"]) for t in sorted(dev[p])]
    if len(dev_pairs) != 98:
        raise ValueError("The saved dev cache expects the original 98-pair selection probe")

    def rank(split, sites, weight, gate=False):
        result = {}
        for p, scores in cache[split].items():
            cs = [c for c in scores if not gate or candidates[p][c] & sites[p]]
            values = [scores[c] + weight * bool(candidates[p][c] & sites[p]) for c in cs]
            result[p] = dec.rank_candidates(cs, values, p, 5).predictions
        return result

    sweep, test_predictions = [], {}
    for weight in GRID:
        dp = rank("dev", dev_sites, weight)
        tp = rank("test", test_sites, weight)
        test_predictions[weight] = tp
        sweep.append({"lambda": float(weight), "holdout_recall1": float(hits(tp, test_pairs, 1).mean()),
                      "holdout_recall5": float(hits(tp, test_pairs).mean()),
                      "dev_recall5": float(hits(dp, dev_pairs).mean())})
    sweep = pd.DataFrame(sweep)
    chosen = float(sweep.loc[sweep.dev_recall5.idxmax(), "lambda"])
    sweep["chosen"] = sweep["lambda"] == chosen
    frame = pd.DataFrame({"cluster": [ev.standardize(p) for p, _ in test_pairs],
                          "baseline_hit": hits(baseline, test_pairs),
                          "prior_hit": hits(test_predictions[chosen], test_pairs),
                          "saturated_hit": hits(test_predictions[1e6], test_pairs),
                          "gate_hit": hits(rank("test", test_sites, 0, gate=True), test_pairs)})
    clusters = frame.cluster.nunique()
    indices = np.random.default_rng(SEED).integers(clusters, size=(REPLICATES, clusters))
    bootstrap = {"seed": SEED, "replicates": REPLICATES,
                 "n_parent_clusters": clusters, "n_canonical_parents": len(known),
                 "n_pairs": len(test_pairs),
                 "method": "Paired percentile bootstrap, resampling parent clusters with replacement, retaining pair-weighted recall",
                 "interpretation": "Conditional on fixed predictions; excludes training-run variance and test-informed rule selection",
                 "source": "scripts/rescore_saved.py; current decoder and saved candidate scores",
                 "comparisons": [paired_interval(frame, a, b, title, indices) for a, b, title in
                                 [("prior_hit", "baseline_hit", "Dev-selected prior vs. no-site baseline"),
                                  ("saturated_hit", "gate_hit", "Retain candidates vs. delete candidates")]]}
    reachable = defaultdict(Counter)
    for p, t in test_pairs:
        key = ev.standardize(t)
        if key in {ev.standardize(c) for c in candidates[p]}:
            reachable[p][key] += 1
    coverage = sum(sum(c.values()) for c in reachable.values())
    ceiling = sum(sum(sorted(c.values(), reverse=True)[:5]) for c in reachable.values())
    audit = {"source": "scripts/rescore_saved.py", "rdkit": rdBase.rdkitVersion,
             "n_pairs": len(test_pairs), "n_parent_strings": len(known),
             "n_standardized_parents": clusters, "endpoint_hits": endpoint_hits,
             "rule_coverage_hits": coverage, "rule_top5_ceiling_hits": ceiling,
             "fingerprint_comparison": comparison, "selected_prior_weight": chosen,
             "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                              [*required, "SOM/som_candidate_scores.json", "rules/metabolism_rules.csv",
                               "SOM/som_predicted_sites.json", "SOM/som_predicted_sites_dev.json",
                               "SOM/som_prior.ipynb", "src/evaluation.py", "src/decoding.py"]}}
    print(sweep.to_string(index=False), flush=True)
    print("Rule coverage / top-five ceiling:", coverage, ceiling, "of", len(test_pairs), flush=True)
    print(json.dumps(bootstrap, indent=2), flush=True)
    if args.write:
        published.to_csv(ROOT / "docs/published_comparison.csv")
        sweep.to_csv(ROOT / "SOM/som_prior_results.csv", index=False)
        (ROOT / "docs/paired_bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
        (ROOT / "docs/evaluation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print("Updated derived artifacts. Original training-run tables are unchanged.")


if __name__ == "__main__":
    main()
