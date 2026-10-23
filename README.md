# Metabolite prediction with a fine-tuned LLM

Given a parent drug molecule as SMILES, predict the structures of its metabolites.

## Layout

```
data/                        the five source datasets, not included (see Data below)
  processed/                 each run's history and checkpoint record; notebook 02 writes the prepared splits here
rules/metabolism_rules.csv   42 SMIRKS transformation rules, editable without touching code
notebooks/
  01_enumeration_rules.ipynb how the rules were built and validated, with outputs
  02_data_and_evaluation.ipynb data preparation and the metric suite, with outputs
  03_finetune.ipynb          the LoRA fine-tune, its diagnostics and the checkpoint it keeps
  04_arms.ipynb              arms 1-5, scored on the same test set
  05_reranker.ipynb          trains a second adapter to RANK, scores it as arm 6
  06_experiment_tracking.ipynb publishes the saved run histories to W&B
WRITEUP.html                the writeup: summary, results, what failed, limitations, plan
WRITEUP.md                   a pointer to it, and an index of the rest of the record
docs/
  FINDINGS.md                every measured result, and where to reproduce it
  finetune_curves.png        where dev loss says to stop, against where recall peaks
  arm_comparison.png         recall@5 for each way of using the model, untuned against fine-tuned
  som_comparison.png         the site of metabolism given vs predicted, and why they differ
  misses_by_chemistry.png    which reactions the ranker misses, and where in them
  examples.html              the prompt and every arm's top five, drawn as structures
  reranker_curves.png        the two ranking runs, plotted against epochs
  arm_results.csv            the full per-arm table at k = 1, 3, 5
  published_comparison.csv   contextual results from this cohort and a different published cohort
  baselines.json             the untuned numbers, written out so nothing hard-codes them
  model_probe_results.csv    the seven-model comparison behind the choice of Qwen3-8B
SOM/
  som_oracle.ipynb           what knowing the site of metabolism is worth, measured four ways
  som_predictor.ipynb        a second adapter trained to mark the atom the body attacks
  som_results.csv            the four conditions, side by side
src/evaluation.py            the metric suite: every scored number comes from here
src/decoding.py              how raw model samples become the k reported predictions
src/inference.py             the best arm as a callable pipeline: parent SMILES in, metabolites out
app.py                       a browser interface over that pipeline
requirements.txt             pinned analysis packages and compatible interface dependencies
```

## Data

The datasets are not in this repository. Their owners license them, and one of them is commercial.
The notebooks expect five CSV files in `data/`, each with the columns `parent_smiles`,
`metabolite_smiles`, `som_smiles`, `enzyme`, `species`, `source`, `reaction`,
`parent_inchi_nostereo` and `parent_scaffold_smiles`.

| file | rows | contents | source |
|---|---|---|---|
| `drugbank.csv` | 3,088 | human parent-metabolite pairs | [DrugBank](https://go.drugbank.com), free for academic, non-commercial use under its licence |
| `gloryx.csv` | 2,364 | human parent-metabolite pairs | the GLORYx reference set, [de Bruyn Kops et al., 2021](https://doi.org/10.1021/acs.chemrestox.0c00224) |
| `metxbiodb.csv` | 2,035 | human parent-metabolite pairs | MetXBioDB, released with BioTransformer, [Djoumbou-Feunang et al., 2019](https://doi.org/10.1186/s13321-018-0324-5) |
| `gostar.csv` | 18,093 | pairs across many species, less curated | GOSTAR, a commercial database licensed by Excelra |
| `test_set.csv` | 309 | 155 held-out parents, none shared with the training files | not publicly available |

The first three are the sources MetaTrans drew on ([Litsa et al., 2020](https://doi.org/10.1039/D0SC02639E)).
These particular files were prepared from those sources and are not public, so freshly downloaded
data will differ in coverage and the reported numbers will not reproduce exactly. Any set of
parents disjoint from the training parents can stand in for the test set.

Notebooks 02 to 04 also import `get_smiles_variants` from `utils/molecule_utils.py`, a helper that
came with the datasets and is not included either. It returns a molecule's canonical SMILES plus
random SMILES spellings of the same molecule, so any RDKit implementation of that behaviour can
replace it.

Without the data, every notebook still shows its saved outputs, `data/processed/` still holds each
run's record and training history, and the interface still runs in replay mode.

## Running it

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
```

### The interface

```bash
.venv/bin/python -m streamlit run app.py
```

Opens on <http://localhost:8501>. Enter a parent drug as SMILES, or pick one of the six examples,
and the top k metabolites come back drawn, each with the rule that produced it and the score the
ranker gave it.

Recorded metabolites appear underneath the predictions when references are available, with hits
outlined. Training-membership messages use the active ranker's candidate-label training file;
reference availability is a separate property. A recorded hit on a parent used for training does
not by itself establish generalization. The supplied test parents were excluded from weight
fitting and checkpoint selection, while their missed transformations informed rule development.

Scores come from one of two places, chosen in the sidebar. **Call the model** scores the
candidates against the checkpoint in `data/processed/reranker_run.json`, which needs
`TINKER_API_KEY` in `.env`; a 35-candidate molecule takes about 7 seconds. **Replay saved scores**
reads `SOM/som_candidate_scores.json`, which that same checkpoint produced for the 215 holdout and
dev molecules, so the interface runs with no key and no network, for those molecules only.

The notebooks save every checkpoint with a seven-day lifetime, which is enough to carry one
notebook's result into the next. The chosen checkpoint of each of the four runs, the generator,
the two rankers and the site-of-metabolism predictor, was extended to 30 days afterwards, to
21 November 2026, so the notebooks and this interface run without retraining until then. The
other checkpoints of each run still expire on their original schedule.

The sidebar shows the date live scoring stops, read from the service rather than from a file, so
it stays right when a lifetime changes. A lifetime can be set or removed at any time:

```python
import tinker
rest = tinker.ServiceClient(api_key=KEY).create_rest_client()
rest.set_checkpoint_ttl_from_tinker_path(path, 30 * 24 * 3600).result()   # 30 days
rest.set_checkpoint_ttl_from_tinker_path(path, None).result()             # no expiry at all
```

`src/inference.py` holds the pipeline and can be called without the interface:

```python
from src.inference import Pipeline
result = Pipeline().predict("CC(C)Cc1ccc(C(C)C(=O)O)cc1", source="live", k=5)
```

### The notebooks

```bash
.venv/bin/python -m notebook notebooks/
```

Run the notebooks in order. 01 and 02 reproduce their numbers from the provided data alone and
cost nothing; 02 writes the prepared training files into `data/processed/`. 03, 04 and 05 call
the Tinker API and need `TINKER_API_KEY` in `.env`.

03 trains the generator and records the chosen checkpoint's `tinker://` path in
`data/processed/finetune_run.json`; 04 reads that file, so it runs from a cold kernel without
retraining. 05 trains a separate ranking adapter and appends arm 6 to `docs/arm_results.csv`, so
it runs after 04. It is run once per training mix, and names its records after the mix:
`TRAINING_MIX = "base"` writes `reranker_base_history.csv` and `reranker_base_run.json`, and
`"base+gostar"` writes `reranker_history.csv` and `reranker_run.json`. Notebook 06 reads both, so
rebuilding the two-run comparison means running 05 twice, about 6.5 hours each. A notebook saves
its checkpoint with a seven-day TTL and records that lifetime in its run file, and 04 checks the
age against it and fails with a clear message rather than calling a checkpoint that has gone. The
four current checkpoints were extended after the fact, so their run files record the extended
lifetime instead and that check follows the real date.

**Reproducibility limits.** Sampling is unseeded and LoRA initialisation happens remotely, so
new runs can change individual predictions, curves, and measured differences. SamplingParams
accepts a seed; these runs did not set one. Repeating the generator fine-tune with an identical
recipe moved recall@5 by about four points. Ranker training variance remains unmeasured because
each mixture was trained once. Generator variability cannot supply a ranker error bar. Saved
prediction artifacts support offline scoring of the recorded runs. Rule coverage and metric
ceilings reproduce under the pinned chemistry and evaluation environment.

**What cannot be re-checked at all.** Two sets of artifacts were not kept. The arms saved their
decoded top-five lists and not the raw generations, so the effect of the `standardize()` spelling
fix on the generating arms cannot be measured after the fact (`docs/FINDINGS.md` §15). The first
replicate run's artifacts were not kept either, so every *(replicate: x)* figure in
`docs/FINDINGS.md` rests on what was recorded at the time and cannot be re-derived. Neither
affects the headline numbers, which come from the current run and are reproducible from the saved
records.

Experiment record: [https://wandb.ai/dianjiang75/metabolite-prediction](https://wandb.ai/dianjiang75/metabolite-prediction)

## Result

**Fine-tuned to rank**, recall@5: 46.0% (17.2% at recall@1), or 46.6% (18.1%) with GOSTAR
mammalian data added. The best **fine-tuned generative** arm reaches 21.4%. The full arm table is in `WRITEUP.html`.

Measured on the holdout test set, 309 pairs over 155 parents, standardised-SMILES matching.

The base train set contains 4,276 pairs over 2,312 canonical parent strings, and dev contains
437 pairs over 250 strings. Splits are disjoint under standardised parent identity. Dev selects
checkpoints; supplied test parents are excluded from weight fitting and checkpoint selection.
Rule development inspected test misses, so the final pipeline’s benchmark is exploratory and
needs validation on a newly reserved cohort (`docs/FINDINGS.md` §10).

Supplied answer-derived site annotations raise recall@5 from 46.6% to 61.2% in the oracle filter.
Predicted-site filtering scores 32.4%. A dominating site bonus retains every candidate and scores
37.5%, recovering some of the gate's losses. The dev-selected bonus (`lambda=0.75`) scores 47.9%
against its matched cached-score baseline of 46.9%, with a paired 95% interval for the difference
of [−0.65, +2.70] points. Its incremental benefit remains uncertain.

Known limitations include incorrect product branches in three chemistry rules and conflicting
candidate labels across equivalent molecular representations. Repairs require regenerated
candidates or training data and reevaluation; the current recorded experiments retain those
limitations. See `docs/FINDINGS.md` §21.

The controls, the split of recall into coverage and ranking, the comparison with published
baselines and the next steps are in `WRITEUP.html`.

## Measured without a model

| | |
|---|---|
| Metric ceiling | top-1 recall cannot exceed **0.502** under exact canonical SMILES, or 0.515 once tautomers and stereochemistry are standardised (309 rows, 155 parents) |
| Baseline correction | MetaTrans reports no top-1. Its 42.5% is top-5; the 30.2% sometimes paired with it is GLORYx's |
| No-model floor | echoing the parent back scores **0.634 Tanimoto** while getting 0% correct; a wrong-site prediction still scores 0.72 |
| Leakage | 0% exact parent overlap with training, but 44% scaffold overlap |
| Rule enumeration | **72.2% coverage** of the holdout test set (standardised; 68.6% stereo-stripped, 53.7% exact) at a median of **32 candidates** per parent |

The rules reach 223 of 309 reference rows. A flawless top-five ranking of those candidates can
recover at most 221/309 (71.5%) because some parents have more than five reachable answers.
Published systems were measured on different cohorts and output policies, so their reported
values are context only.

## Decisions

- **USPTO is not used.** It is a reaction corpus, not a metabolism one, and it lists the
  conjugating co-substrate as a reactant, so transferase metabolites cannot be learned from it
  in the form this task needs.
- **SyGMa and GLORYx rule sets are not imported.** 90.6% of test transformations already appear in
  the training data, but a rule must also fire at the right *site*. Of the 86 test metabolites the rules cannot reach, **66 belong to
  transformation types the training data already contains** (21.4% of all test pairs), so a
  broader rule library could recover familiar chemistry at sites these 42 rules miss. The benefit
  of an external library is untested; the reason for not importing one is the CDK-to-RDKit
  translation cost, not a coverage bound.
- **Every molecule passes through one `standardize()` function before comparison**: normalised,
  neutral, canonical tautomer, stereochemistry dropped. The source data is internally inconsistent
  about tautomers, with 58% of parents changing under canonicalisation, and 22.6% of training
  skeletons recorded with more than one stereo form.
