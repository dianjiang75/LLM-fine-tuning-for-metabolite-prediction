"""Build WRITEUP.html from saved experiment results.
Figures are embedded so the file opens anywhere on its own.

    python3 docs/build_writeup.py
"""
import base64, json, pathlib, sys, html as H
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import evaluation as ev
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- numbers, read not typed
arms = pd.read_csv(ROOT / "docs/arm_results.csv", index_col=0)
def at(k, arm, col="recall_pair"):
    return float(arms[(arms.index == k) & (arms["arm"] == arm)][col].iloc[0])
A_GEN_UNT, A_GEN, A_REUSE = "1  untuned, direct (few-shot)", "3  tuned, direct", "4  tuned, ranking"
A_RANK, A_GOSTAR = "6  tuned ranker (base mix)", "6  tuned ranker (+GOSTAR mammalian)"
DELTA = (at(5, A_RANK) - at(5, A_REUSE)) * 100

som = pd.read_csv(ROOT / "SOM/som_results.csv").set_index("condition")["recall@5"]
prior = pd.read_csv(ROOT / "SOM/som_prior_results.csv")
P_BASE = float(prior[prior["lambda"] == 0]["holdout_recall5"].iloc[0])
P_HARD = float(prior[prior["lambda"] == 1e6]["holdout_recall5"].iloc[0])
P_PICK = prior[prior["chosen"]].iloc[0]
history = pd.read_csv(ROOT / "data/processed/finetune_history.csv")
FT_LOW = history.loc[history["dev_loss"].idxmin()]
FT_PICK = history.loc[history["dev_recall5"].idxmax()]
bootstrap = json.loads((ROOT / "docs/paired_bootstrap.json").read_text())
PRIOR_INTERVAL = next(c for c in bootstrap["comparisons"]
                      if c["label"] == "Dev-selected prior vs. no-site baseline")
ft = json.loads((ROOT / "data/processed/finetune_run.json").read_text())
rr = json.loads((ROOT / "data/processed/reranker_run.json").read_text())
rb = json.loads((ROOT / "data/processed/reranker_base_run.json").read_text())

# The generator's selection probe, redrawn here the way notebook 03 drew it, so the pair count in
# section 2.3 and the recall it divides cannot come apart.
_dev = {}
for _r in (json.loads(l) for l in (ROOT / "data/processed/dev.jsonl").read_text().splitlines() if l.strip()):
    _dev.setdefault(_r["parent"], set()).add(_r["target"])
PROBE = np.random.default_rng(0).choice(sorted(_dev), size=ft["probe_molecules"], replace=False)
PROBE_PAIRS = sum(len(_dev[q]) for q in PROBE)
_n = ft["probe_recall5"] * PROBE_PAIRS
assert abs(_n - round(_n)) < 1e-9, "probe recall is not a whole number of these pairs"

# ---------------------------------------------------------------- assets
def png(name):
    b = base64.b64encode((ROOT / "docs" / name).read_bytes()).decode()
    return f"data:image/png;base64,{b}"

def mol(smiles, w=170, h=125):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return ""
    d = rdMolDraw2D.MolDraw2DSVG(w, h); o = d.drawOptions()
    o.clearBackground = False; o.bondLineWidth = 1.4
    rdMolDraw2D.PrepareAndDrawMolecule(d, m); d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")

EX = json.loads((ROOT / "docs/examples.json").read_text())
PRO = "CN1CCN(CCCN2c3ccccc3Sc3ccc(Cl)cc32)CC1"
test = pd.read_csv(ROOT / "data/test_set.csv", dtype=str, keep_default_na=False, na_values=[])
test["parent"] = [ev.canonical(s) for s in test["parent_smiles"]]
test["metabolite"] = [ev.canonical(s) for s in test["metabolite_smiles"]]
PRO_METS = list(test[test["parent"] == PRO]["metabolite"])
PRO_KEYS = {ev.match_key(m, "standardized") for m in PRO_METS}
def pro_hits(key):
    return len({ev.match_key(c, "standardized") for c in EX[PRO][key]} & PRO_KEYS)

# Published figures and this method under one matching rule, computed in 05_reranker.ipynb.
PUB = pd.read_csv(ROOT / "docs/published_comparison.csv", index_col="system")
OURS = "this pipeline (+GOSTAR mammalian ranker)"
PUB_ROWS = "".join(
    f'<tr{" class=head" if s == OURS else ""}>'
    f'<td>{H.escape("This method" if s == OURS else s)}</td>'
    f'<td class=n>{r["recall@5"]:.1%}</td><td class=n>{r["precision@5"]:.1%}</td>'
    f'<td class=n>{r["any_hit@5"]:.1%}</td></tr>'
    for s, r in PUB.iterrows())

# The summary table: recall@5, precision and the per-drug hit rate, which are the columns the
# published figures also report. Every arm's row at k = 1, 3 and 5 is in docs/arm_results.csv.
A_CTL = "6c untuned control (base mix run)"
# One name and one order for every arm, matching the table in section 2.2 row for row.
SUMMARY_ARMS = [
    ("Untuned, generative (control)", A_GEN_UNT),
    ("Fine-tuned, generative", A_GEN),
    ("Fine-tuned generative used as ranker", A_REUSE),
    ("Untuned, ranking by likelihood", "2  untuned, ranking"),
    ("Untuned, ranking (control)", A_CTL),
    ("Fine-tuned to rank", A_RANK),
    ("Fine-tuned to rank, +GOSTAR", A_GOSTAR),
]
SUMMARY_ROWS = "".join(
    f'<tr{" class=head" if arm == A_RANK else ""}>'
    f'<td class=idx>{i}</td><td>{H.escape(label)}</td>'
    f'<td class=n>{at(5, arm):.1%}</td>'
    f'<td class=n>{at(5, arm, "precision_known"):.1%}</td>'
    f'<td class=n>{at(5, arm, "parent_hit_rate"):.1%}</td></tr>'
    for i, (label, arm) in enumerate(SUMMARY_ARMS, 1))

# Section 2.2 takes its names from the same list, so the two tables cannot disagree.
PURPOSE = [
    "Establish a generation baseline, with examples in the prompt",
    "Evaluate generation after training",
    "Evaluate candidate ranking with the same trained weights",
    "Establish a ranking baseline",
    "Measure performance of the yes/no format before training",
    "Evaluate training specifically for candidate ranking",
    "Evaluate additional training data",
]
assert len(PURPOSE) == len(SUMMARY_ARMS)
CONFIG_ROWS = "".join(
    f"<tr><td class=idx>{i}</td><td>{H.escape(label)}</td><td class=q>{H.escape(why)}</td></tr>"
    for i, ((label, _), why) in enumerate(zip(SUMMARY_ARMS, PURPOSE), 1))

pool = pd.read_csv(ROOT / "docs/pooling_retest.csv")
POOLED = float(pool[pool["arm"].str.contains("pooled")]["recall@5"].iloc[0])
POOL_RULES = pool[pool["arm"] == "trained ranker, rules only"].iloc[0]
POOL_UNION = pool[pool["arm"].str.contains("pooled")].iloc[0]

PRO_ROWS = "".join(
    f'<tr><td>{H.escape(lab)}</td><td class=n><b>{pro_hits(k)}</b> of {len(PRO_METS)}</td></tr>'
    for k, lab in (("arm1", "Untuned, generative"), ("arm3", "Fine-tuned, generative"),
                   ("arm4", "Fine-tuned generative used as ranker"), ("arm6", "Fine-tuned to rank")))
PRO_MOLS = "".join(f'<figure>{mol(m)}<figcaption>metabolite {i}</figcaption></figure>'
                   for i, m in enumerate(PRO_METS, 1))

# ---------------------------------------------------------------- the page
CSS = """
:root{--bg:#fcfcfb;--panel:#fff;--ink:#121212;--muted:#5c5b57;--line:#e6e5e1;
 --blue:#2a78d6;--orange:#eb6834;--warn:#b4521f}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16161a;--panel:#1e1e23;
 --ink:#f1f1ee;--muted:#a5a49f;--line:#2f2f35;--warn:#f0a068}}
:root[data-theme=dark]{--bg:#16161a;--panel:#1e1e23;--ink:#f1f1ee;--muted:#a5a49f;--line:#2f2f35;--warn:#f0a068}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,
 "Segoe UI",Helvetica,Arial,sans-serif;padding:0 16px 80px}
.wrap{max-width:860px;margin:0 auto}
header{padding:54px 0 10px}
h1{font-size:30px;line-height:1.2;margin:0 0 10px;letter-spacing:-.01em}
.byline{font-size:15px;color:var(--muted);margin:0 0 16px}
.links{display:flex;flex-wrap:wrap;gap:8px;margin:0}
.btn{display:inline-block;border:1px solid var(--line);background:var(--panel);border-radius:8px;
 padding:8px 14px;color:var(--blue);text-decoration:none;font-size:13.5px;font-weight:500;
 transition:border-color .12s,transform .12s}
.btn:hover{border-color:var(--blue);transform:translateY(-1px)}
/* Keep navigation readable as the page scrolls. */
.nav{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:4px;flex-wrap:wrap;
 margin:18px -16px 0;padding:9px 16px;background:var(--panel);
 border-top:1px solid var(--line);border-bottom:1px solid var(--line);
 box-shadow:0 2px 10px rgba(0,0,0,.08)}
.navlabel{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;font-weight:600;
 color:var(--muted);margin-right:8px}
.nav a{color:var(--muted);text-decoration:none;font-size:13px;font-weight:500;padding:6px 13px;
 border:1px solid transparent;border-radius:999px}
.nav a:hover{background:var(--bg);color:var(--ink);border-color:var(--line)}
section{margin:0;padding:34px 0 6px;border-top:2px solid var(--line);scroll-margin-top:52px}
section:first-of-type{border-top:none}
h2{font-size:22px;margin:0 0 18px;letter-spacing:-.01em;color:var(--ink)}
h3{font-size:17px;margin:28px 0 8px}
h4{font-size:15px;margin:22px 0 6px;color:var(--muted)}
p{margin:0 0 12px}
.big{font-size:21px;line-height:1.45;margin:0 0 16px}
/* Give tables a clear header and alternating row backgrounds. */
table{width:100%;border-collapse:separate;border-spacing:0;margin:20px 0;font-size:14.5px;
 background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
 padding:10px 12px;background:var(--bg);border-bottom:1px solid var(--line);font-weight:600}
td{padding:9px 12px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:none}
tbody tr:nth-child(even) td{background:rgba(128,128,128,.05)}
th.n,td.n{text-align:right}
td.n{font-variant-numeric:tabular-nums}
tr.head td{font-weight:700;color:var(--blue);background:rgba(90,130,255,.07)}
figure{margin:0;text-align:center}
figcaption{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
.fig{margin:18px 0;padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px}
.fig img{width:100%;display:block;border-radius:6px}
.fig figcaption{font-size:13px;line-height:1.5;text-align:left;text-transform:none;
 letter-spacing:normal;margin:10px 2px 0}
.table-scroll{overflow-x:auto;margin:20px 0}
.table-scroll table{margin:0;min-width:560px}
h3[id],h4[id],figure[id],div[id]{scroll-margin-top:100px}
.cap{font-size:13px;color:var(--muted);margin:10px 2px 0}
.mols{display:grid;grid-template-columns:minmax(0,1fr) 24px repeat(3,minmax(0,1fr));
 gap:12px;align-items:center;padding:20px 14px;background:var(--panel);
 border:1px solid var(--line);border-radius:12px}
.mols figure{min-width:0}
.mols svg{display:block;width:100%;height:auto;max-width:190px;margin:0 auto}
.arrow{font-size:22px;color:var(--muted)}
.note{background:var(--panel);border-left:3px solid var(--warn);border-radius:0 8px 8px 0;
 padding:12px 14px;font-size:14px;color:var(--muted);margin:16px 0}
.chain{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
 font-size:15px;line-height:1.75;margin:16px 0}
.chain b{color:var(--ink)}
th.idx,td.idx{width:26px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
td.q{color:var(--muted)}
td i{color:var(--muted);font-style:normal;font-size:13px}
.months{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(230px,1fr))}
.month{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.month h4{margin:0 0 8px;font-size:14px;color:var(--blue)}
.month p{font-size:14px;color:var(--muted);margin:0}
code{font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
footer{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
@media(max-width:620px){h1{font-size:24px}.big{font-size:18px}}
"""

PAGE = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Metabolite prediction by fine-tuning Qwen3-8B</title><style>{CSS}</style></head><body><div class=wrap>

<header>
<h1>Metabolite prediction by fine-tuning Qwen3-8B</h1>
<p class=byline>by Rencong (Dian) Jiang</p>
<p class=links>
<a class=btn href="https://wandb.ai/dianjiang75/metabolite-prediction">Experiment record (W&amp;B) &#8599;</a>
<a class=btn href="docs/FINDINGS.md">Full measured record &#8599;</a>
<a class=btn href="docs/examples.html">Worked examples &#8599;</a></p>
</header>

<nav class=nav>
<span class=navlabel>Jump to</span>
<a href="#summary">Summary</a><a href="#methods">Methods</a><a href="#results">Results</a>
<a href="#discussion">Discussion</a><a href="#next">Next steps</a>
</nav>

<section id=summary>
<h2>1. Summary</h2>
<p>The task of this project is to predict the metabolites produced when the body transforms a
parent drug. The input is a SMILES string, a text format for the parent molecule, and the
fine-tuned LLM model outputs 5 metabolite predictions also in SMILES string format based on
likelihood scoring.</p>
<p>There are three hypotheses I want to investigate as I decide how to fine-tune the model:</p>
<ol>
<li>Can the model write metabolites from scratch, or does it do better picking from a list of
rule-generated candidates?</li>
<li>How much data does it take to reach the ceiling? Does more data always help?</li>
<li>Does a separate adapter predicting the site of metabolism (SOM) make it better?</li>
</ol>
<p><b>Methods:</b> for the ranking model in hypothesis #1, we use 42 SMIRKS rules
(<code>rules/metabolism_rules.csv</code>) to enumerate every metabolite candidate those rules can
produce for a given parent (median of 32 candidates per parent). Qwen3-8B, fine-tuned with LoRA,
answers yes/no on each and returns the five highest on <code>logP(yes) &minus; logP(no)</code>.
Training used 4,276 pairs from DrugBank, GLORYx and MetXBioDB; GOSTAR was not included, to test
hypothesis #2. For hypothesis #3, a second LoRA adapter was trained on the same drugs with the
reacting atom marked, so it outputs the parent molecule carrying its SOM as a label.</p>
<p>The main comparison uses the provided test set of 309 pairs. Recall@5 is the share of
recorded metabolites appearing in the top five. Full metrics accompany the results in Section 3.</p>
<p>Those scores use standardised-SMILES matching, while MetaTrans counts a prediction correct
when its fingerprint Tanimoto to a reference is exactly 1.0. Applying their rule to this
method's own top five puts all four systems on one matching rule:</p>
<table><tr><th>Top-5, fingerprint equality</th><th class=n>recall@5</th>
<th class=n>precision@5</th><th class=n>drugs with &ge;1 hit</th></tr>
{PUB_ROWS}</table>
<p class=cap>Published figures are the top-5 window of Table 4 in Litsa, Das &amp; Kavraki,
Chem. Sci. 2020.</p>
<p><b>Results.</b> Answers to my three hypotheses:</p>
<ol>
<li>The ranking model performs best when the model is trained to rank. Reusing the generative
fine-tune as a ranker scores {at(5, A_REUSE):.1%}, but training a ranker directly brings the
recall up to {at(5, A_RANK):.1%}, while the untuned model given the same yes/no format scores
{at(5, A_CTL):.1%} (and only {at(5, A_GEN_UNT):.1%} when writing metabolites from scratch).
Broader SMIRKS coverage will raise the recall ceiling.</li>
<li>Increasing the amount of training data to 2.5&times; did little to help the training outcome.
Adding mammalian GOSTAR records took training examples from 24,025 to 60,520 and moved recall@5
from {at(5, A_RANK):.1%} to {at(5, A_GOSTAR):.1%}. Both runs stopped improving at epoch 1.5,
which means the extra data made each epoch longer, but the plateau began in the same place.</li>
<li>Using site labels derived from the known answers raises recall from
{at(5, A_GOSTAR):.1%} to {som['true sites, union per parent']:.1%}. The SOM adapter matches at
least one labelled site for 62.6% of evaluated drugs. Filtering on its predicted sites reduces
recall to {som['sites predicted by the model']:.1%}. A development-selected score adjustment
that retains all candidates reaches {P_PICK['holdout_recall5']:.1%}, compared with its
{P_BASE:.1%} baseline. This small gain remains uncertain.</li>
</ol>
<p><b>Other results.</b> Model size made no difference. Seven base models were compared without
fine-tuning, from 4B parameters up to 397B. Their scores spanned only 8.4 points, and the
comparison ran on 120 molecules, which is too few to separate anything closer than about 10
points. In fact, the 27-billion model scored exactly the same as the 4-billion one.</p>
<p>Pooling the model's own generated structures into the candidate list from SMIRKS reactions
dropped recall from {P_BASE:.1%} to {POOLED:.1%}, most likely because every negative the ranker
trained on was generated from a SMIRKS rule, so model-generated molecules are unlike anything it
was trained to judge.</p>
</section>

<section id=methods>
<h2>2. Methods</h2>
<p><b>Data and evaluation.</b> We used 4,276 training, 437 development and 309 test
drug&ndash;metabolite pairs, with splits separated by standardised drug identity. Training fitted
the model weights; development data selected the saved models. Test drugs were excluded from
both steps. Rule development included inspection of test transformations, so the evaluation is
exploratory.</p>
<p>All configurations used Qwen3-8B and shared molecular matching and output-cleaning
procedures. Recall@k measures the percentage of recorded metabolites recovered among the top k
predictions. We report recall@5 throughout, with results at k = 1, 3 and 5 in the
<a href="docs/arm_results.csv">accompanying results table</a>.</p>

<h3>2.1 Prediction pipeline</h3>
<p>We compared generating metabolites directly with ranking candidates produced by 42 editable
chemical transformation rules. With five predictions allowed per drug, the highest achievable
recall was <b>71.5%</b>.</p>
<p>The trained ranker scored each drug&ndash;candidate pair independently using
<code>log P(yes) &minus; log P(no)</code> and returned the five highest-scoring candidates.</p>
<p>Separate LoRA adapters were trained for generation and ranking using the literature
training corpus. These tasks produced {ft['train_examples']:,} generation examples and
{rb['train_examples']:,} ranking examples. The expanded ranking dataset included GOSTAR and
contained {rr['train_examples']:,} examples, approximately
{rr['train_examples'] / rb['train_examples']:.1f} times as many. Each recorded metabolite was
paired with up to eight candidates that the rules produce but the source tables do not record for
that parent, half of them the closest wrong candidates by fingerprint similarity and half drawn at
random. Some of those negatives may be real metabolites missing from the dataset.</p>

<h3>2.2 Experimental comparisons</h3>
<p>Seven configurations examined the effects of task-specific training, prediction method and
additional data.</p>
<table><thead><tr><th class=idx>#</th><th>Configuration</th><th>Purpose</th></tr></thead><tbody>
{CONFIG_ROWS}
</tbody></table>
<p>The generator-to-ranker comparison changed the training examples, scoring procedure and
model-selection task. Its improvement therefore reflects the combined approach. Additional
experiments evaluated a prompt without examples and pooling rule-generated and model-generated
candidates.</p>

<h3>2.3 Choosing the saved model</h3>
<p>We selected generation and ranking checkpoints with the highest recall@5 on fixed
development subsets. Recall@5 measures the fraction of recorded metabolites recovered within
five predictions. Cross-entropy loss measures how well the model predicts the tokens in the
recorded answer strings.</p>
<p>The figure shows how the generator changed during training. Development loss was measured on
160 examples, and recall@5 on {ft['probe_molecules']} parent structures carrying {PROBE_PAIRS}
recorded drug&ndash;metabolite pairs. Development loss was lowest after one epoch, where the model
recovered <b>{FT_LOW['dev_recall5']:.1%}</b> of the recorded metabolites within its five
predictions. Recovery reached <b>{FT_PICK['dev_recall5']:.1%} after four epochs</b>, an improvement
of <b>{(FT_PICK['dev_recall5'] - FT_LOW['dev_recall5']) * 100:.1f} percentage points</b>.</p>
<p>We selected the four-epoch checkpoint because it recovered the most recorded metabolites on the
development subset. Training loss continued to fall over those epochs while development loss rose,
so the model was fitting its training examples more closely and predicting the development answer
strings less well. The improvement in metabolite recovery has to be read alongside that evidence
of possible overfitting.</p>
<figure class=fig id=figure-1>
<img src="{png('finetune_curves.png')}" alt="Generator training loss and development recall at individual checkpoints">
<figcaption><b>Figure 1. Generator training and model selection.</b> Top: training loss, as a
25-step running mean, and development loss on 160 examples. Bottom: development recall@5 on the {PROBE_PAIRS}
recorded pairs of the probe. Dots show individual checkpoint evaluations; the line averages each
checkpoint with its neighbours on either side and leaves out the final one. Open circles mark the
lowest-loss checkpoint, in both panels, and the selected model, in the bottom panel.</figcaption>
</figure>

</section>
"""

PAGE += f"""
<section id=results>
<h2>3. Results</h2>
<h4 id=table-s1>Results across the main configurations</h4>
<div class=table-scroll>
<table><thead><tr><th class=idx>#</th><th>Configuration</th><th class=n>recall@5</th>
<th class=n>precision@5</th><th class=n>drugs with &ge;1 hit</th></tr></thead>
<tbody>{SUMMARY_ROWS}</tbody></table>
</div>
<p class=cap>Test set: 309 pairs over 155 canonical parent strings, representing 154
standardised drug identities. The parent hit rate uses the 155 evaluated parent strings.
Results at k = 1, 3 and 5 are in <a href="docs/arm_results.csv">the full results file</a>.</p>

<h3>3.1 Training for ranking gave the strongest results</h3>
<p>Fine-tuned generation recovered <b>{at(5, A_GEN):.1%}</b> of recorded metabolites. Untuned
generation recovered <b>{at(5, A_GEN_UNT):.1%}</b> with six examples in the prompt and
<b>{at(5, '1b untuned, direct (bare)'):.1%}</b> without examples.</p>
<p>Using the fine-tuned generator to rank rule-generated candidates increased recall to
<b>{at(5, A_REUSE):.1%}</b>, using the same model weights. Untuned likelihood ranking reached
<b>{at(5, '2  untuned, ranking'):.1%}</b>.</p>
<p>Training specifically for yes/no ranking reached <b>{at(5, A_RANK):.1%}</b>, compared with
<b>{at(5, A_CTL):.1%}</b> for the untuned yes/no control. This approach combined changes to
training examples, scoring and model selection.</p>
<figure class=fig id=figure-2>
<img src="{png('arm_comparison.png')}" alt="Recall at five for generation, likelihood ranking and yes/no ranking before and after fine-tuning">
<figcaption><b>Figure 2. Metabolite recovery across prediction approaches.</b> Recall@5
measures the percentage of recorded test metabolites recovered among five predictions. The
71.5% ceiling applies to ranking the fixed rule-generated candidate lists.</figcaption>
</figure>
<h4 id=figure-s2>Example: prochlorperazine</h4>
<p>Three recorded metabolites arise through N-demethylation, aromatic hydroxylation and
S-oxidation. The rules produce {EX[PRO]['n_candidates']} candidates, and each arm returns five.</p>
<div class=mols><figure>{mol(PRO,190,140)}<figcaption>parent</figcaption></figure>
<div class=arrow>&rarr;</div>{PRO_MOLS}</div>
<p class=cap><b>Figure 3. Prochlorperazine and its three recorded metabolites.</b></p>
<table aria-label="Metabolites recovered for prochlorperazine">
<thead><tr><th scope=col>Arm</th><th class=n scope=col>Metabolites found</th></tr></thead>
<tbody>{PRO_ROWS}</tbody></table>
<p>The fine-tuned generator and the generator reused for ranking each found the hydroxylation
and sulfoxide products. The trained ranker also recovered the N-demethylation product.</p>
<p><a href="docs/examples.html">Worked examples with prompts and structures &rarr;</a></p>

<h3>3.2 Additional data and candidates</h3>
<p>Adding GOSTAR increased the ranking dataset approximately 2.5-fold and moved recall from
<b>{at(5, A_RANK):.1%} to {at(5, A_GOSTAR):.1%}</b>, recovering two additional metabolites.
Both runs reached their highest measured development recall around 1.5 training epochs.
Each configuration was trained once, so the reliability of this small gain remains uncertain.</p>
<figure class=fig id=figure-s1>
<img src="{png('reranker_curves.png')}" alt="Development recall during ranking-model training with the base dataset and the GOSTAR-expanded dataset">
<figcaption><b>Figure 4. Ranking-model training.</b> Development recall@5 on 40 drug structures containing 68 recorded pairs.
The base dataset contains {rb['train_examples']:,} examples; the GOSTAR-expanded dataset
contains {rr['train_examples']:,}. Each curve represents one training run.</figcaption>
</figure>
<p>Pooling rule-generated and model-generated candidates together increased metabolite coverage
from {POOL_RULES['reachable']:.1%} to {POOL_UNION['reachable']:.1%}. However, in the evaluation
using the same trained ranker for both lists, the recall@5 fell from
{POOL_RULES['recall@5']:.1%} to {POOLED:.1%}, indicating that the expanded candidate list reduced
the effectiveness of ranking.</p>

<h3>3.3 Errors involved missing candidates and reaction sites</h3>
<p>The rules failed to propose <b>86 of 309 recorded metabolites</b>. These answers were
unavailable to the ranker.</p>
<p>Performance also varied by reaction type. The ranker recovered <b>28 of 29
O-glucuronidation products</b> and <b>18 of 42 aliphatic hydroxylation products</b>. Among the
24 missed aliphatic hydroxylations, 14 had another product of the same rule in the top five.
This suggests difficulty identifying the reaction site, the atom where the chemical change
occurs.</p>
<figure class=fig id=figure-5>
<img src="{png('misses_by_chemistry.png')}" alt="Recorded metabolites recovered, ranked below the top five or absent from the candidate lists, grouped by reaction type">
<figcaption><b>Figure 5. Recovery by reaction type.</b> Bars distinguish recovered
metabolites, candidates missed by ranking, and metabolites absent from the candidate lists.
Pairs are grouped by the first matching rule. This analysis uses the later GOSTAR scoring pass,
which recovered 145 of 309 metabolites ({P_BASE:.1%}).</figcaption>
</figure>

<h3>3.4 Using reaction-site information</h3>
<p>Filtering candidates using recorded reaction sites increased recall from
<b>{som['arm 6, no site information']:.1%} to {som['true sites, union per parent']:.1%}</b>.
These sites came from the known answers, so this experiment assessed the potential value of
site information.</p>
<p>A separate model predicted at least one labelled site for <b>62.6%</b> of evaluated drugs.
Filtering candidates using these predictions reduced recall to
<b>{som['sites predicted by the model']:.1%}</b>. Incorrect site predictions could remove the
recorded metabolite from consideration.</p>
<p>We then used predicted sites to adjust candidate scores while retaining every candidate.
The setting selected on development data reached <b>{P_PICK['holdout_recall5']:.1%}</b>,
compared with its <b>{P_BASE:.1%}</b> baseline, recovering three additional metabolites.
The estimated gain remains uncertain: its 95% confidence interval spans
<b>{PRIOR_INTERVAL['ci_low']:.1f} to +{PRIOR_INTERVAL['ci_high']:.1f} percentage points</b>.</p>
<p class=cap>Re-evaluating the same model recovered one additional metabolite, moving the
baseline from {at(5, A_GOSTAR):.1%} to {P_BASE:.1%}. The uncertainty estimate was calculated
by repeatedly sampling test drugs and reusing their saved predictions.
Source files: <a href="docs/pooling_retest.csv">candidate-pooling results</a>,
<a href="SOM/som_prior_results.csv">site-adjustment results</a> and
<a href="docs/paired_bootstrap.json">uncertainty calculations</a>.</p>
</section>
"""

PAGE += f"""
<section id=discussion>
<h2>4. Discussion</h2>

<h3>4.1 Decomposition of the remaining error</h3>
<p>Because a ranking arm can only return what the rules produced, its score factors exactly:</p>
<div class=chain><b>{at(5, A_RANK):.1%}</b> overall &nbsp;=&nbsp; <b>72.2%</b> rule coverage
&nbsp;&times;&nbsp; <b>63.7%</b> recall given the answer is reachable</div>
<p>Coverage did not move between the two fine-tunes. The ranking component went from 42.6% to
63.7%, closing 37% of the gap to a perfect ranker. A flawless top-five ranker would reach 71.5%
rather than 72.2%, because two parents have six reachable answers and only five slots. That
leaves 25.5 points inside lists that already contain the answer, against 8.4 points available
from the five largest missing transformation families.</p>

<p>The quality of the training labels depends on how thoroughly each drug’s metabolism has been studied. An unrecorded candidate may have been missed experimentally or never investigated. Improving the dataset would therefore require a clearly defined prediction setting, records of experimental conditions and detection limits, and follow-up assessment of promising predictions without a recorded match. This would help us distinguish incorrect predictions from gaps in the available evidence.</p>

<h3>4.2 Validation loss as a selection criterion</h3>
<p>The growing gap between training and development loss suggests overfitting. Selecting by
development recall@5 gave the checkpoint that recovered the most metabolites on that subset.
Repeated training runs and evaluation on an untouched dataset are needed to assess how well
this choice generalises. Multiple valid metabolites and SMILES spellings may contribute to the
difference between loss and recall; we did not test that explanation directly.</p>

<h3>4.3 What we would improve in the engineering</h3>
<p>The next engineering experiments should examine how closely training matches the final prediction task. Our ranker learns a yes/no decision for each candidate, so a useful next experiment is to train it to compare candidates from the same drug, focusing on mistakes that it currently scores highly. When expanding the candidate sources, we would test training the ranker on the combined list, given the drop in recall observed when we pooled them. For precision, the current five-slot metric is limited by incomplete metabolite records, and the ranking scores have not been calibrated as confidence estimates. We would therefore evaluate a confidence threshold for returning shorter lists, reporting their precision and coverage separately. These changes should be compared on a larger development set using the final prediction procedure, with other training settings held fixed and raw outputs saved so that each improvement can be checked.</p>

<h3>4.4 Limitations</h3>
<p>The holdout set was never trained on and never used to choose a checkpoint. The rule set is the
one exception: its stopping point was chosen by inspecting which holdout transformations remained
uncovered, so the rules saw that distribution even though no model did. 44% of holdout parents
share a scaffold with a training parent, and recall was never scored as a function of that
similarity. Sampling is unseeded and <code>compute_logprobs</code> is not bit-reproducible, so a
rerun reproduces the shape of every curve and the ordering of the arms, not the exact figures.
Three independent audits ran against this work; each removed a causal claim, and the measurements
underneath held.</p>
</section>

<section id=next>
<h2>5. Next steps</h2>
<h3>Month 1: widen the candidate list, and train the ranker on its own mistakes</h3>
<p>Two things, both measured. <b>27.8%</b> of holdout pairs are unreachable from the current 42
SMIRKS rules, and a parent the rules miss scores zero however good the ranker is; the five largest
missing transformation families alone account for <b>8.4 points</b> of coverage, and the largest
is plain hydroxylation, which is cheap to write. Separately, the ranker's negatives are drawn by
fingerprint similarity, which is a proxy for what confuses it. Train it on its own top-five
mistakes instead, which is what actually confuses it, and score a drug's whole candidate list
together rather than one candidate at a time, so the model compares candidates instead of judging
each in isolation. <b>25.5 points</b> sit inside lists that already contain the answer.</p>

<h3>Month 2: build a site-of-metabolism model worth using, and give it to the ranker as an input</h3>
<p>Given the true site, the ranker finds <b>90.5%</b> of the answers still reachable against
<b>64.6%</b> with no site information. That is the size of what is available. Our own site
predictor marks a true site for 62.6% of parents, which is not enough to filter on, and the gap
between those two numbers is the work. The fix is architectural as much as it is accuracy: put
the predicted sites in the ranker's prompt and retrain on site-marked drugs, so the model learns
per drug how far to trust them, rather than a filter deleting an answer whenever the guess is
wrong.</p>

<h3>Month 3: use the enzyme, and make the numbers defensible</h3>
<p>Every dataset carries an enzyme column and this work discards it, on the assumption that the
SMIRKS rules already encode most of what the enzyme would tell the model. That assumption is
untested and cheap to test: add the enzyme as an optional input and measure. Extend uncertainty
estimates to the main model comparisons and measure variability across repeated training runs.
Add an abstention path so the model can decline on a scaffold
unlike anything it trained on; in tox screening a confident wrong answer costs more than a
refusal.</p>
</section>



<footer>
<p><b>Time spent:</b> about 30 to 40 hours over three to four days. The two ranking fine-tunes
account for roughly 13 of those on their own and ran unattended.</p>
<p>Built from saved experiment results with <code>docs/build_writeup.py</code>.</p>
</footer>

</div></body></html>"""

out = ROOT / "WRITEUP.html"
out.write_text(PAGE)
import re
body = re.sub(r"<[^>]+>", " ", PAGE)
print(f"wrote {out.relative_to(ROOT)}  ({len(PAGE)/1024:.0f} KB, {len(body.split())} words)")
