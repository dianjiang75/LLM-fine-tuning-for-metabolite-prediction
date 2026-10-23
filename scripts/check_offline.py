"""Check the public repository's replay workflow without data files or API calls.

Run from any directory:
    python scripts/check_offline.py
    python scripts/check_offline.py --all-saved

The default checks four replay examples. ``--all-saved`` checks every cached
parent, including agreement between the saved and freshly enumerated candidates.
Notebook cells are checked for syntax. The pure fingerprint comparison function
is also exercised in isolation; no notebook cell or API workflow is run.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem, DataStructs

from src.config import load_local_env, require_env
from src.decoding import collapse, rank_candidates
from src.evaluation import canonical, match_key
from src.inference import (
    InvalidInput,
    LiveRanker,
    Pipeline,
    RankerUnavailable,
    clean_parent,
    enumerate_candidates,
)
from src.smiles import get_smiles_variants


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_sources() -> None:
    paths = [ROOT / "app.py"]
    for folder in ("src", "docs", "scripts"):
        paths.extend(sorted((ROOT / folder).glob("*.py")))
    for path in paths:
        compile(path.read_text(), str(path), "exec")

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    notebooks += sorted((ROOT / "SOM").glob("*.ipynb"))
    cells = 0
    fingerprint_equal = None
    for path in notebooks:
        notebook = json.loads(path.read_text())
        require(notebook["nbformat"] == 4, f"Unsupported notebook format: {path}")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}:cell {index}", "exec")
            require("import molecule_utils" not in source,
                    f"Notebook still depends on the unbundled helper: {path}")
            cells += 1
            if path.name == "05_reranker.ipynb":
                for node in ast.parse(source).body:
                    if isinstance(node, ast.FunctionDef) and node.name == "fingerprint_equal":
                        namespace = {"Chem": Chem, "DataStructs": DataStructs}
                        module = ast.Module(body=[node], type_ignores=[])
                        exec(compile(module, str(path), "exec"), namespace)
                        fingerprint_equal = namespace[node.name]
    require(fingerprint_equal is not None, "Published comparison function is missing")
    require(fingerprint_equal("CC(=O)O", "CC(=O)[O-]"),
            "Published comparison must use RDKit path fingerprints")
    require(not fingerprint_equal("CCCCCCC", "CCCCCCCC"),
            "Published comparison still behaves like Morgan equality")
    print(f"PASS: {len(paths)} Python files and {cells} code cells in {len(notebooks)} notebooks compile; comparison fingerprint checked", flush=True)


def check_input_and_decoding() -> None:
    for value in ("", "I cannot determine this.", "CCO.[Na+]", "[CH3]", "[CH3:1]O"):
        try:
            clean_parent(value)
        except InvalidInput:
            continue
        raise AssertionError(f"Input validation accepted {value!r}")
    require(canonical("I cannot determine this.") is None,
            "SMILES parser accepted prose")
    decoded = collapse([["CCO", "OCC", "CC=O", "O=CC", "invalid"]], "CCO")
    require(decoded.n_parent_echo == 2 and decoded.n_unparseable == 1,
            "Decoder did not account for parent echoes and invalid output")
    require(decoded.predictions == ["CC=O"], "Decoder did not merge equivalent molecules")
    a = rank_candidates(["CC=O", "CC(=O)O"], [1.0, 1.0], "CCO")
    b = rank_candidates(["CC(=O)O", "CC=O"], [1.0, 1.0], "CCO")
    require(a.predictions == b.predictions, "Equal-score ranks depend on input order")

    for parent in ("CCO", "C[C@H](O)C(=O)O", "F/C=C/F", "c1ccccc1"):
        random.seed(37)
        first = get_smiles_variants(parent, n=4)
        random.seed(37)
        second = get_smiles_variants(parent, n=4)
        require(first == second, "SMILES augmentation is not reproducible from its seed")
        require(first and first[0] == canonical(parent), "Canonical spelling must come first")
        require(len(first) == len(set(first)) <= 4, "Repeated or excess augmentation variants")
        require(all(canonical(s) == canonical(parent) for s in first),
                "Augmentation changed the molecule or specified stereochemistry")
    print("PASS: invalid input, parent-echo removal, duplicate collapse, score ties and seeded stereo-preserving augmentation", flush=True)


def check_credentials() -> None:
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        path = Path(directory) / ".env"
        load_local_env(path)
        try:
            require_env("OFFLINE_CHECK_MISSING")
        except RuntimeError as error:
            require("OFFLINE_CHECK_MISSING" in str(error), "Missing-key error does not identify its setting")
        else:
            raise AssertionError("Missing credentials were accepted")
        path.write_text(
            "# Test values only\nOFFLINE_CHECK_EXISTING=file-value\n"
            'OFFLINE_CHECK_QUOTED="quoted value"\nOFFLINE_CHECK_EQUALS=abc=def\n'
        )
        os.environ["OFFLINE_CHECK_EXISTING"] = "exported-value"
        load_local_env(path)
        require(require_env("OFFLINE_CHECK_EXISTING") == "exported-value",
                "Local settings replaced an exported environment variable")
        require(require_env("OFFLINE_CHECK_QUOTED") == "quoted value",
                "Quoted local settings were parsed incorrectly")
        require(require_env("OFFLINE_CHECK_EQUALS") == "abc=def",
                "An equals sign inside a setting was lost")
    print("PASS: optional credential file, exported settings, quoted values and missing-key explanation", flush=True)


def check_replay(all_saved: bool) -> None:
    pipe = Pipeline()
    parents = pipe.saved.parents() if all_saved else [
        "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(=O)Nc1ccc(O)cc1",
        "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    ]
    require(len(pipe.rules) == 42, "Expected the 42 rules used by the saved experiment")
    for index, parent in enumerate(parents, 1):
        parent = clean_parent(parent)
        generated = enumerate_candidates(parent, pipe.rules)
        saved = pipe.saved.scores[parent]
        require(set(generated) == set(saved), f"Saved candidate set differs from the rules for {parent}")
        require(all(math.isfinite(s) for s in saved.values()), "Non-finite saved score")
        result = pipe.predict(parent, source="saved", k=10)
        keys = [match_key(p.smiles) for p in result.predictions]
        require(result.source == "saved" and result.checkpoint, "Missing replay provenance")
        require(len(keys) == len(set(keys)) <= 10, "Replay returned repeated or excess structures")
        require(None not in keys and match_key(parent) not in keys, "Replay returned invalid input or a parent echo")
        require(all(p.rule != "unattributed" for p in result.predictions), "A replay product has no rule attribution")
        require(all(math.isfinite(p.score) for p in result.predictions), "A replay score is non-finite")
        if index % 25 == 0:
            print(f"Replay progress: {index}/{len(parents)} parents", flush=True)
    try:
        pipe.predict("C", source="saved")
    except RankerUnavailable:
        pass
    else:
        raise AssertionError("An uncached molecule unexpectedly had saved scores")
    print(f"PASS: full replay pipeline and fresh candidate agreement for {len(parents)}/{len(pipe.saved.parents())} cached parents; uncached input reports its limit", flush=True)


def check_app() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()
    require(not app.exception and not app.error, "App failed during startup")
    require(app.radio[0].value == "saved", "Offline app check must select saved scoring")

    def submit() -> None:
        next(button for button in app.button if button.label == "Predict metabolites").click().run()
        require(not app.exception, "App raised an exception")

    submit()
    require(not app.error, "Saved example failed in the app")
    require(any(x.value == "Top 5" for x in app.subheader), "App did not display five predictions")
    require(len(app.get("download_button")) == 1, "Prediction download is unavailable")
    app.slider[0].set_value(3).run()
    require(any(x.value == "Top 3" for x in app.subheader), "Changing the prediction budget failed")
    app.text_input[0].set_value("not a molecule")
    submit()
    require(bool(app.error), "App did not explain invalid input")
    app.text_input[0].set_value("C")
    submit()
    require(bool(app.error) and "No saved scores" in app.error[0].value,
            "App did not explain unavailable saved scores")
    print("PASS: Streamlit startup, saved prediction, top-k control, CSV download and input errors", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-saved", action="store_true", help="Check every cached parent")
    parser.add_argument("--skip-app", action="store_true", help="Skip Streamlit interaction checks")
    args = parser.parse_args()
    # These guards also apply when a developer has a real key in the environment or .env.
    blocked = AssertionError("An offline check attempted live scoring")
    with patch.object(LiveRanker, "configured", return_value=False), \
         patch.object(LiveRanker, "api_key", return_value=None), \
         patch.object(LiveRanker, "score", side_effect=blocked), \
         patch.object(LiveRanker, "_connect", side_effect=blocked), \
         patch.object(LiveRanker, "expires_at", side_effect=blocked):
        check_sources()
        check_input_and_decoding()
        check_credentials()
        check_replay(args.all_saved)
        if not args.skip_app:
            check_app()
    print("Offline checks passed. Training, live inference and external publishing were not run.", flush=True)


if __name__ == "__main__":
    main()
