# Submission validation

Validated on 23 October 2026 against the current workspace, including uncommitted changes.

The free/replay workflow passed in a fresh Python environment and a separate copy containing
only tracked and non-ignored project files. That copy had no `.env`, original molecule helper,
licensed source datasets or prepared training/development splits. A second isolated copy
received the five supplied CSV files for the two free preparation notebooks.

## Environment and public replay

The fresh environment used Python 3.12.2, RDKit 2024.03.5, NumPy 2.0.2, pandas 3.0.1 and
Streamlit 1.56.0. Installing `requirements.txt` installed 159 packages; `uv pip check` reported
that all installed packages were compatible.

From the public-file copy, using that environment's interpreter:

```text
python -B -u scripts/check_offline.py --all-saved

PASS: 11 Python files and 76 code cells in 9 notebooks compile; comparison fingerprint checked
PASS: invalid input, parent-echo removal, duplicate collapse, score ties and seeded stereo-preserving augmentation
PASS: optional credential file, exported settings, quoted values and missing-key explanation
PASS: full replay pipeline and fresh candidate agreement for 215/215 cached parents; uncached input reports its limit
PASS: Streamlit startup, saved prediction, top-k control, CSV download and input errors
Offline checks passed. Training, live inference and external publishing were not run.
```

The process exited with status 0. Streamlit emitted its bare-mode `ScriptRunContext` warning
during the AppTest check. No application exception occurred. The download check verifies that
the prediction CSV control is available.

The app was also started as a local server with API credentials removed from its environment:

```text
python -B -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 18501 --browser.gatherUsageStats false

/_stcore/health 200 text/html; charset=UTF-8
/ 200 text/html
```

In the browser, submitting the README's imipramine example displayed 24 scored rule candidates
and five ranked structures, with scores, reaction labels and a CSV download button. The page
correctly reported that training membership and reference metabolites were unavailable in the
public copy.

## Free data preparation

Both notebooks ran sequentially through `nbclient.NotebookClient.execute()` in fresh kernels
using the same installed environment. Executed notebooks and regenerated data stayed in the
isolated rehearsal copy.

```text
PASS 01_enumeration_rules.ipynb seconds 237
PASS 02_data_and_evaluation.ipynb seconds 71
Both free notebooks completed in fresh kernels. Paid notebooks were not executed.
```

All 18 code cells completed with zero error outputs. Notebook 02 reported:

```text
train : 4,276 pairs over 2,067 molecules
dev   :   437 pairs over   230 molecules
test  :   309 pairs over   155 molecules
molecules in both train and dev : 0
molecules shared with the test  : 0
training examples written : 17,079
violations: 0
```

The included augmentation helper generates fresh SMILES spellings. Historical outputs remain
in the source notebooks, and fresh training can differ from the recorded runs.

## Evaluation artifacts and report

`python -B -u scripts/rescore_saved.py --write` completed using the supplied test data and the
original prepared development split. It recovered 104/309 exact, 138/309 stereo-stripped and
145/309 standardized matches from the later score cache. The corrected RDKit path-fingerprint
comparison produced 46.0% recall, 18.2% precision and 61.3% of parents with a hit. The
development-selected site-prior weight remained 0.75; its paired parent-cluster interval still
included zero.

`docs/evaluation_audit.json` records these results and input hashes. All nine hashes were checked
against the final inputs. The original per-arm experiment table remains a separate historical
record, including the expanded ranker's original 144/309 result.

Both HTML reports rebuilt successfully. Their local links and assets resolved, and
`git diff --check` passed. The README, writeup, findings and all nine notebooks' markdown were
checked for the requested prose cleanup.

## Scope

After submission cleanup on 23 October, the project contains 56 public source and artifact files.
Credentials, licensed CSV inputs, generated training splits, retired helpers, old output, editor
settings and caches were moved to a separate local archive. All moved files were verified by
SHA-256, and all tracked project files were retained. A second `scripts/check_offline.py` run
passed on the cleaned folder, covering the four documented replay examples and Streamlit
interactions. The earlier full replay check covered all 215 cached parents. Git metadata remains
in the working repository and is excluded from the submission ZIP.

This validates installation, data preparation, cached rescoring and the complete public replay
workflow. Paid training, live model inference, checkpoint availability and W&B publishing were
excluded as requested. Syntax checks of those notebooks do not establish that their external
services will execute successfully.

The scientific limitations remain: test transformations informed rule development, each ranker
training mixture has one run, and the reference set and normalized matching criterion limit the
interpretation of recall. The submission presents an exploratory work sample. A fresh reserved
cohort and repeated training are still required for stronger conclusions.
