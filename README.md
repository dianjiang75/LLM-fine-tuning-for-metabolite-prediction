# Metabolite prediction by fine-tuning Qwen3-8B

Given a parent drug as a SMILES string, this pipeline proposes up to five metabolite structures.
Forty-two SMIRKS rules generate candidates, a median of 32 per drug on the supplied benchmark.
Qwen3-8B, fine-tuned with LoRA, scores each candidate with log P(yes) − log P(no) and returns the
highest-ranked distinct structures.

The full writeup is [`WRITEUP.html`](WRITEUP.html). The measured results and their provenance are
in [`docs/FINDINGS.md`](docs/FINDINGS.md), and the training runs are on
[W&B](https://wandb.ai/dianjiang75/metabolite-prediction).

## Results

The main comparison uses 309 drug–metabolite pairs. Test drugs were excluded from model training
and checkpoint selection, but test transformations informed rule development, so these results
are exploratory. Recall@5 measures recovery of recorded metabolites after normalization, charge
neutralization, tautomer canonicalization and removal of stereochemistry.

Three experiments guided the model design:

1. **Generation and ranking.** Direct ranker training reached 46.0% recall@5. Reusing the
   generative fine-tune as a ranker scored 30.7%, and fine-tuned generation scored 21.4%. The
   untuned model scored 16.2% in the same yes/no format and 4.9% when generating with examples.
   These results compare complete recipes; the training data, scoring and checkpoint-selection
   procedures differ between the generator and ranker.
2. **Additional training data.** Adding mammalian GOSTAR records increased ranking examples
   from 24,025 to 60,520 and moved recall@5 from 46.0% to 46.6%, a gain of two reference rows.
   Each mixture was trained once. The change also increased optimizer updates and changed data
   composition, so it does not establish data saturation.
3. **Site-of-metabolism information.** Answer-derived site labels raised recall@5 from 46.6%
   to 61.2% in an oracle experiment. The learned adapter matched at least one labelled site for
   62.6% of drugs, but filtering candidates on its predictions reduced recall to 32.4%. A score
   adjustment chosen on development data retained all candidates and reached 47.9% against its
   46.9% baseline. Its paired parent-cluster 95% interval, −0.7 to +2.7 points, includes zero.

| # | Configuration | recall@5 | precision@5 | drugs with ≥1 hit |
|---|---|---|---|---|
| 1 | Untuned, generative (control) | 4.9% | 1.8% | 8.4% |
| 2 | Fine-tuned, generative | 21.4% | 8.5% | 36.8% |
| 3 | Fine-tuned generative used as ranker | 30.7% | 12.0% | 45.8% |
| 4 | Untuned, ranking by likelihood | 20.4% | 8.0% | 34.2% |
| 5 | Untuned, ranking (control) | 16.2% | 6.3% | 26.5% |
| 6 | Fine-tuned to rank | **46.0%** | 17.9% | 61.3% |
| 7 | Fine-tuned to rank, +GOSTAR | **46.6%** | 18.5% | 63.2% |

The test set has 309 pairs over 155 parent strings, representing 154 standardized drug identities.
Precision@5 counts recorded metabolites and can undercount genuine metabolites absent from the
reference set. Results at k = 1, 3 and 5 are in [`docs/arm_results.csv`](docs/arm_results.csv).

![Recall@5 for each prediction approach](docs/arm_comparison.png)

*Metabolite recovery across prediction approaches. The 71.5% ceiling applies to ranking the fixed
rule-generated candidate lists.*

### Published context

MetaTrans's released evaluator uses RDKit path-fingerprint equality. The published top-five
figures below come from Table 4 in Litsa, Das & Kavraki, *Chem. Sci.* 2020, on 65 drugs with
179 recorded metabolites. This project's cohort, reference set and output policy differ, so
these figures provide context for the task.

| Top-5, fingerprint equality | recall@5 | precision@5 | drugs with ≥1 hit |
|---|---|---|---|
| MetaTrans | 42.5% | 23.5% | 80.0% |
| SyGMa (rule-based, 2008) | 42.4% | 23.4% | 72.3% |
| GLORYx | 30.2% | 16.6% | 64.6% |

The cached predictions from this project are rescored in
[`docs/published_comparison.csv`](docs/published_comparison.csv). The matching implementation and
cohort differences are described in [`docs/FINDINGS.md`](docs/FINDINGS.md), section 1.

### Training

![Generator training and model selection](docs/finetune_curves.png)

*Generator training and model selection. Development loss was lowest after one epoch, when the
model recovered 19.4% of the development metabolites within five predictions. Recovery reached
28.6% after four epochs, and that checkpoint was kept.*

![Ranking-model training](docs/reranker_curves.png)

*Ranking-model training. Development recall@5 on 40 drugs carrying 68 recorded pairs. The base
set has 24,025 examples and the GOSTAR-expanded set 60,520. Each curve is one training run.*

### Where the remaining error is

A ranker can only return what the rules produce, so its score factors exactly: 46.0% overall
equals 72.2% rule coverage times 63.7% recall when the answer is reachable. The rules fail to
propose 86 of the 309 recorded metabolites, and a flawless top-five ranker of the existing lists
would reach 71.5%.

Recovery varies by reaction. The ranker recovers 28 of 29 O-glucuronidation products and 18 of 42
aliphatic hydroxylations. For 13 of the 24 missed hydroxylations, another product of the same rule
made the top five. Those cases identify reaction-site discrimination as a source of error.

![Recovery by reaction type](docs/misses_by_chemistry.png)

*Recovery by reaction type: recovered metabolites, candidates the ranking missed, and metabolites
absent from the candidate lists. This uses the later GOSTAR scoring pass, which recovered 145 of
309 metabolites (46.9%).*

### Other results

- **The model probe did not establish a size advantage.** Seven untuned models from 4B to 397B
  parameters spanned 8.3 points on the 120-molecule probe. The models also differ in family,
  architecture and training. This comparison supported choosing Qwen3-8B for the experiments;
  it does not establish equivalent performance across model sizes.
- **Pooling candidates reduced recall in this experiment.** Adding generated structures to
  the rule candidates raised coverage from 72.2% to 75.1% and reduced recall@5 from 46.9% to
  29.8%. Training negatives all came from rules, which may explain the poorer ranking of the
  expanded lists. That explanation has not been tested.

### How it was measured

Generator training used 4,276 pairs and development used 437, with splits separated by
standardized drug identity. Development data selected the saved checkpoints. The SOM adapter
used a broader training pool that included mammalian GOSTAR, with development and test drugs
excluded. Exact drug separation still permits close analogues across splits: 44 of 101 distinct
test scaffolds also appear in the combined DrugBank, GLORYx and MetXBioDB training sources.
Rule development inspected the supplied test transformations; an independent evaluation needs a new cohort
reserved before further tuning.

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
  05_reranker.ipynb          trains a ranking adapter and scores it as arm 6
  06_experiment_tracking.ipynb publishes the saved run histories to W&B
WRITEUP.html                the writeup: summary, results, what failed, limitations, plan
WRITEUP.md                   a pointer to it, and an index of the rest of the record
docs/
  FINDINGS.md                measured results, provenance and limitations
  finetune_curves.png        generator loss and development recall over training
  arm_comparison.png         recall@5 for each way of using the model, untuned against fine-tuned
  som_comparison.png         the site of metabolism given vs predicted, and why they differ
  misses_by_chemistry.png    which reactions the ranker misses, and where in them
  examples.html              prompts and top-five structures for four configurations
  reranker_curves.png        the two ranking runs, plotted against epochs
  arm_results.csv            the full per-arm table at k = 1, 3, 5
  published_comparison.csv   contextual results from this cohort and a different published cohort
  baselines.json             recorded untuned baseline metrics
  model_probe_results.csv    the seven-model comparison behind the choice of Qwen3-8B
SOM/
  som_oracle.ipynb           oracle and predicted-site filtering experiments
  som_predictor.ipynb        a second adapter trained to mark reacting atoms
  som_results.csv            the four conditions, side by side
src/evaluation.py            the metric suite: every scored number comes from here
src/decoding.py              how raw model samples become the k reported predictions
src/smiles.py                seeded SMILES spelling augmentation used by the notebooks
src/inference.py             the best arm as a callable pipeline: parent SMILES in, metabolites out
app.py                       a browser interface over that pipeline
scripts/check_offline.py      checks the public replay workflow without credentials or paid calls
requirements.txt             pinned analysis packages and compatible interface dependencies
```

## Data

The datasets are not in this repository. Their owners license them, and one of them is commercial.
The submission package is the one exception: it carries back `data/test_set.csv`, the 309-pair
benchmark that came with the task, so the interface can show recorded metabolites beside its
predictions. It stays out of the public repository.
The notebooks expect five CSV files in `data/`, each with the columns `parent_smiles`,
`metabolite_smiles`, `som_smiles`, `enzyme`, `species`, `source`, `reaction`,
`parent_inchi_nostereo` and `parent_scaffold_smiles`.

| file | rows | contents | source |
|---|---|---|---|
| `drugbank.csv` | 3,088 | human parent-metabolite pairs | [DrugBank](https://go.drugbank.com), access governed by its licence |
| `gloryx.csv` | 2,364 | human parent-metabolite pairs | the GLORYx reference set, [de Bruyn Kops et al., 2021](https://doi.org/10.1021/acs.chemrestox.0c00224) |
| `metxbiodb.csv` | 2,035 | human parent-metabolite pairs | MetXBioDB, released with BioTransformer, [Djoumbou-Feunang et al., 2019](https://doi.org/10.1186/s13321-018-0324-5) |
| `gostar.csv` | 18,093 | pairs across many species, less curated | GOSTAR, a commercial database licensed by Excelra |
| `test_set.csv` | 309 | 155 held-out parents, none shared with the training files | not publicly available |

These particular files were prepared from the listed sources and are not public. Freshly downloaded
data will differ in coverage, so the original numbers require the original input files. A new
evaluation cohort must have the same column schema and be reserved before model and rule tuning.
It should also report structural similarity to the training drugs.

Notebooks 02 to 04 use the included `src/smiles.py` to generate canonical and randomized SMILES
spellings. The original dataset helper is not required.

Without the source data, the saved notebook outputs and training histories remain readable, and
the interface runs in replay mode. Prepared training splits are excluded from the repository and
are regenerated by notebook 02 when the source data is available.

## Running it

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From a new checkout:

```bash
git clone https://github.com/dianjiang75/LLM-fine-tuning-for-metabolite-prediction.git
cd LLM-fine-tuning-for-metabolite-prediction
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_offline.py
```

The offline check uses committed artifacts. It requires no source datasets, API key or model
checkpoint access.
The final rehearsal and its scope are recorded in [`docs/VALIDATION.md`](docs/VALIDATION.md).

### Replay the saved predictions

```bash
.venv/bin/python -m streamlit run app.py
```

Open <http://localhost:8501> and choose **Replay saved scores** in the sidebar. Select
**Imipramine**, then submit the prediction form. The app displays the top-ranked metabolite
structures, their scores and the rules that produced them. Phenobarbital, caffeine and
paracetamol also have saved scores. The ibuprofen and propranolol examples require live scoring.

The saved cache covers 215 test and development parent strings. It supports predictions for
those molecules only. To use the same workflow from Python:

```python
from src.inference import Pipeline

result = Pipeline().predict("CN(C)CCCN1c2ccccc2CCc2ccccc21", source="saved", k=5)
for prediction in result.predictions:
    print(prediction.rank, prediction.smiles, prediction.score)
```

Recorded metabolites appear underneath the predictions when references are available, with hits
outlined. Training-membership messages use the active ranker's candidate-label training file;
reference availability is a separate property. A recorded hit on a parent used for training does
not by itself establish generalization. The supplied test parents were excluded from weight
fitting and checkpoint selection, while their missed transformations informed rule development.

### Live scoring

Live scoring sends candidates to Tinker and incurs API charges. Set `TINKER_API_KEY` in the
environment or a local `.env` file, then select **Call the model**. The app reads the ranker checkpoint from
`data/processed/reranker_run.json` and queries the service for its expiry date. Saved checkpoints
have limited lifetimes; the run file alone does not guarantee that one is still available.
An expired checkpoint requires retraining or updating the run file to a compatible active one.

From Python, select `source="live"`:

```python
from src.inference import Pipeline
result = Pipeline().predict("CC(C)Cc1ccc(C(C)C(=O)O)cc1", source="live", k=5)
```

### Reproduce the experiments

Full experiment runs require the five licensed source files described above. Notebooks 03 to 05
and the SOM experiments also require `TINKER_API_KEY`, set in the environment or a local `.env`
file, and a funded Tinker account. Notebook 06 uploads results to W&B and requires `WANDB_API_KEY`.

```bash
.venv/bin/python -m notebook notebooks/
```

Run 01 and 02 first. They make no API calls; 02 writes the prepared splits into
`data/processed/`. Review the settings and cost estimates before running the paid notebooks.

With `data/test_set.csv` and the original prepared `data/processed/dev.jsonl` available, the
saved candidate scores support a separate offline rescore:

```bash
.venv/bin/python scripts/rescore_saved.py
```

This recomputes the fingerprint comparison, site-prior sweep and paired bootstrap intervals
without model calls. Add `--write` to update their derived CSV and JSON artifacts. The original
training-arm results remain in `docs/arm_results.csv`.

03 trains the generator and records the chosen checkpoint's `tinker://` path in
`data/processed/finetune_run.json`; 04 reads that file and uses the saved checkpoint while it
remains available. 05 trains a separate ranking adapter and appends arm 6 to `docs/arm_results.csv`, so
it runs after 04. It is run once per training mix, and names its records after the mix:
`TRAINING_MIX = "base"` writes `reranker_base_history.csv` and `reranker_base_run.json`, and
`"base+gostar"` writes `reranker_history.csv` and `reranker_run.json`. Rebuilding the two-run
comparison requires both settings; each recorded ranking run took about 6.5 hours. Notebook 06
publishes the resulting histories. Run the SOM predictor after data preparation and the expanded
ranker, then run the oracle and prior notebooks to evaluate its sites. These notebooks write new
results, so use a separate working copy to preserve the recorded experiments.

**Reproducibility limits.** Sampling is unseeded and LoRA initialisation happens remotely, so
new runs can change individual predictions, curves, and measured differences. SamplingParams
accepts a seed; these runs did not set one. Repeating the generator fine-tune with an identical
recipe moved recall@5 by about four points. Ranker training variance remains unmeasured because
each mixture was trained once. Generator variability cannot supply a ranker error bar. Saved
ranker scores support offline rescoring for the cached drugs. Rule coverage and metric ceilings
can be reproduced when the source data is available under the pinned chemistry environment.
Changes to decoding or data preparation can also change a rerun, so retain the source version and
input files with each new experiment.

**Limits of the saved record.** Full raw generations and complete per-parent predictions for
every original arm were not retained. The repository contains aggregate metrics, worked examples
and the later ranker score cache. The effects of parser and SMILES-standardization fixes on the
generating arms therefore cannot be reconstructed without new sampling. The first replicate's
artifacts were also not retained; its figures in `docs/FINDINGS.md` are historical records.

Experiment record: [W&B training runs and scored configurations](https://wandb.ai/dianjiang75/metabolite-prediction).
