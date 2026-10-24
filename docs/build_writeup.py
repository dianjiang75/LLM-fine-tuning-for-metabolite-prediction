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
SOM_PRIOR_EQUATION = (ROOT / "docs/som_prior_equation.svg").read_text().strip()
assert f'data-prior-weight="{P_PICK["lambda"]:g}"' in SOM_PRIOR_EQUATION, \
    "Re-render the SOM equation SVG for the selected prior weight."
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

# Different cohorts; the cached pipeline row uses MetaTrans's RDKFingerprint criterion.
PUB = pd.read_csv(ROOT / "docs/published_comparison.csv", index_col="system")
OURS = "this pipeline (+GOSTAR mammalian ranker)"
PUB_ROWS = "".join(
    f'<tr{" class=head" if s == OURS else ""}>'
    f'<td>{H.escape("This method" if s == OURS else s)}</td>'
    f'<td class=n>{r["recall@5"]:.1%}</td><td class=n>{r["precision@5"]:.1%}</td>'
    f'<td class=n>{r["any_hit@5"]:.1%}</td></tr>'
    for s, r in PUB.iterrows())

# The summary table: recall@5, precision and the per-drug hit rate, which are the columns the
# published figures also report. Every configuration's row at k = 1, 3 and 5 is in docs/arm_results.csv.
A_CTL = "6c untuned control (base mix run)"
# One name and one order for every configuration, matching the table in section 2.2 row for row.
SUMMARY_ARMS = [
    ("Untuned, generative (control)", A_GEN_UNT),
    ("Fine-tuned, generative", A_GEN),
    ("Fine-tuned generator used as ranker", A_REUSE),
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
    'Generates metabolites from six examples. Establishes a prompting-only baseline.',
    'Trains on recorded parent–metabolite pairs. Tests whether fine-tuning improves generation.',
    'Ranks supplied candidates by generation likelihood, without further training. Tests whether providing candidates improves recovery.',
    'Ranks candidates by base-model likelihood. Provides a control for configuration 3.',
    'Ranks candidates using yes/no scores. Establishes the baseline for ranker training.',
    'Trains on labelled parent–candidate pairs. Compares ranking supervision with the untuned scorer (5) and generator used for ranking (3).',
    'Adds mammalian GOSTAR records to ranking training. Tests whether more data improves recall over configuration 6.',
]
assert len(PURPOSE) == len(SUMMARY_ARMS)
CONFIG_ROWS = "".join(
    f"<tr><td class=idx>{i}</td><td>{H.escape(label)}</td><td class=q>{H.escape(why)}</td></tr>"
    for i, ((label, _), why) in enumerate(zip(SUMMARY_ARMS, PURPOSE), 1))

# precision@5 cannot reach 1: the numerator is capped by a drug's recorded metabolite count
# while the denominator is always k, so the attainable maximum is worth stating beside it.
_per_parent = (test.assign(key=[ev.match_key(x, "standardized") for x in test["metabolite_smiles"]],
                           par=[ev.canonical(x) for x in test["parent_smiles"]])
                   .groupby("par")["key"].nunique())
PREC_CEIL = sum(min(n, 5) for n in _per_parent) / (len(_per_parent) * 5)

pool = pd.read_csv(ROOT / "docs/pooling_retest.csv")
POOLED = float(pool[pool["arm"].str.contains("pooled")]["recall@5"].iloc[0])
POOL_RULES = pool[pool["arm"] == "trained ranker, rules only"].iloc[0]
POOL_UNION = pool[pool["arm"].str.contains("pooled")].iloc[0]

PRO_ROWS = "".join(
    f'<tr><td>{number}. {H.escape(SUMMARY_ARMS[number - 1][0])}</td>'
    f'<td class=n><b>{pro_hits(key)}</b> of {len(PRO_METS)}</td></tr>'
    for number, key in ((1, "arm1"), (2, "arm3"), (3, "arm4"), (6, "arm6")))
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
.math-line{font-family:"STIX Two Math","Cambria Math","Times New Roman",serif;
 font-size:17px;font-weight:400;line-height:1.5;margin:12px 0 16px}
.som-equation{margin:10px 0 16px;padding:0;overflow-x:auto}
.som-equation svg{display:block;width:500px;max-width:100%;height:auto;
 margin:0 auto;color:var(--ink)}
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
<a class=btn href="http://127.0.0.1:8501/" target="_blank" rel="noopener" title="Open the locally running prediction app">Prediction webtool &#8599;</a>
<a class=btn href="https://wandb.ai/dianjiang75/metabolite-prediction">Experiment record (W&amp;B) &#8599;</a>
<a class=btn href="docs/FINDINGS.md">Full measured record &#8599;</a>
<a class=btn href="docs/examples.html">Worked examples &#8599;</a></p>
</header>

<nav class=nav>
<span class=navlabel>Jump to</span>
<a href="#summary">Summary</a><a href="#methods">Methods</a><a href="#results">Results</a>
<a href="#discussion">Discussion and next steps</a>
</nav>

<section id=summary>
<h2>1. Summary</h2>
<p>Given a parent drug as a SMILES string, this project returns up to five candidate metabolite
structures. I compared generating structures with Qwen3-8B and using it to rank candidates
produced by <a href="rules/metabolism_rules.csv">chemical reaction rules</a>.</p>
<p>The experiments address three questions:</p>
<ol>
<li>Can the model write metabolites from scratch, or does it do better picking from a list of
rule-generated candidates?</li>
<li>Does adding mammalian GOSTAR records improve the ranker?</li>
<li>Does a separate adapter predicting the site of metabolism (SOM) make it better?</li>
</ol>
<p><b>Methods:</b> for the ranking model in hypothesis #1, we use <a href="rules/metabolism_rules.csv">42 SMIRKS rules</a>
to enumerate every metabolite candidate those rules can
produce for a given parent (median of 32 candidates per parent). Qwen3-8B, fine-tuned with LoRA,
answers yes/no on each and returns the five highest on <code>logP(yes) &minus; logP(no)</code>.
The base training corpus contained 4,276 pairs from DrugBank, GLORYx and MetXBioDB. A second
ranking run added mammalian GOSTAR records. A separate adapter was trained on available site
annotations, including mammalian GOSTAR.
It outputs the parent molecule with the predicted reacting atoms marked.</p>
<p>The main comparison uses the provided test set of 309 pairs. Recall@5 is the share of
recorded metabolites appearing in the top five under standardised matching, which removes
stereochemistry and merges charge and tautomer representations. The test set informed rule
development, so these results are exploratory. Full metrics accompany the results in Section 3.</p>
<p>For context, the table below compares results from different evaluation cohorts. This
pipeline's saved top five were rescored using the <code>RDKFingerprint</code> equality criterion
in MetaTrans's released code. The published systems were evaluated on 65 drugs with 179
metabolites; this pipeline uses 155 parent strings with 309 reference pairs. Reference sets and
output policies also differ, so the table does not establish which system performs better.</p>
<table><tr><th>Top-5, fingerprint equality</th><th class=n>recall@5</th>
<th class=n>precision@5</th><th class=n>drugs with &ge;1 hit</th></tr>
{PUB_ROWS}</table>
<p class=cap>Published figures are the top-5 window of Table 4 in Litsa, Das &amp; Kavraki,
Chem. Sci. 2020. The pipeline row is an offline rescore of the later saved scoring pass;
<a href="docs/FINDINGS.md">the measured record</a> gives its provenance.</p>
<p><b>Results.</b> Answers to my three hypotheses:</p>
<ol>
<li>The trained ranker recovered {at(5, A_RANK):.1%} of recorded metabolites, compared with
{at(5, A_REUSE):.1%} for the generator reused as a ranker and {at(5, A_GEN):.1%} for direct
generation. The untuned yes/no control scored {at(5, A_CTL):.1%}. These compare complete training
and scoring recipes; the objective alone was not isolated.</li>
<li>Adding mammalian GOSTAR records increased ranking examples from 24,025 to 60,520 and moved
recall@5 from {at(5, A_RANK):.1%} to {at(5, A_GOSTAR):.1%}, two additional reference rows.
Each mixture was trained once. Both selected checkpoints were at epoch 1.5. The small
development probe and changing data composition and update count leave data saturation unresolved.</li>
<li>Using site labels derived from the known answers raises recall from
{at(5, A_GOSTAR):.1%} to {som['true sites, union per parent']:.1%}. The SOM adapter matches at
least one labelled site for 62.6% of evaluated drugs. Filtering on its predicted sites reduces
recall to {som['sites predicted by the model']:.1%}. A development-selected score adjustment
that retains all candidates reaches {P_PICK['holdout_recall5']:.1%}, compared with its
{P_BASE:.1%} baseline. This small gain remains uncertain.</li>
</ol>
<p><b>Other findings:</b> This small comparison of untuned models showed no consistent relationship
between model size and recall. Seven models from 4B to 397B parameters were tested on 120
molecules, with recall@5 ranging from 15.8% to 24.2%. Model families and training variants also
differed. Qwen3-8B was chosen for its cost and the baselines already available.</p>
<p>Pooling the model's own generated structures into the candidate list from <a href="rules/metabolism_rules.csv">SMIRKS reactions</a>
dropped recall from {P_BASE:.1%} to {POOLED:.1%}. Every training negative came from a reaction
rule, so unfamiliar generated candidates are one possible explanation. That mechanism was not
tested directly.</p>
<p>We tested an alternative SMILES format by adding spaces between chemical units. A short
development pilot found no clear benefit from spaced SMILES: it recovered 1 of 108 reference
metabolites, compared with 3 for standard SMILES, and used 48% more training tokens.
Other chemistry-based tokenization methods could be tested next.</p>
</section>

<section id=methods>
<h2>2. Methods</h2>
<p><b>Data and evaluation.</b> We combined DrugBank, GLORYx and MetXBioDB, converted structures
to canonical SMILES and removed duplicate pairs. We excluded carbon-free parents, parents with
over 100 heavy atoms, and pairs whose metabolite matched the unchanged parent. We also removed
stereochemistry from metabolite targets and any resulting duplicates.</p>
<p>We split the cleaned data 90:10 by standardized parent identity, giving <b>4,276 training pairs
and 437 development pairs</b>. The provided <b>309 test pairs</b> were excluded from weight fitting
and checkpoint selection. Their missed transformations informed reaction-rule development.
A fresh cohort must be reserved before further tuning to estimate generalisation independently.</p>
<p>For generator training, each parent had up to four SMILES representations, including its
canonical form and randomized atom orderings. These described the same molecule and shared one
fixed metabolite target. This expanded training to <b>{ft['train_examples']:,} examples</b>, with the aim of
reducing sensitivity to how the input SMILES was written. GOSTAR additions were restricted to
mammalian records, with invalid structures, duplicate pairs and development/test parent overlaps
removed.</p>
<p>Our primary metric was <b>recall@5</b>, the fraction of recorded metabolites found among the
five highest-ranked predictions. Matching normalises functional groups, neutralises charges,
selects a canonical tautomer and removes stereochemistry. On identical cached predictions,
recall@5 is 33.7% with exact canonical matching, 44.7% after removing stereo, and 46.9% with full
standardisation. The endpoint measures recovery under that identity policy; it cannot establish
recall over unrecorded metabolites.</p>

<h3>2.1 Prediction pipeline</h3>
<p>We compared generating metabolites directly with ranking candidates produced by 42 editable
<a href="rules/metabolism_rules.csv">chemical transformation rules encoded as SMIRKS</a>. With five predictions allowed per drug, the
highest achievable recall@5 from these candidates was <b>71.5%</b> on the test set.</p>
<p>The trained ranker scored each drug&ndash;candidate pair independently using
<code>log P(yes) &minus; log P(no)</code> and returned the five highest-scoring candidates.</p>
<p>Separate LoRA adapters were trained for generation and ranking using the literature training
corpus. These tasks produced {ft['train_examples']:,} generation examples and {rb['train_examples']:,} ranking examples.
Adding GOSTAR expanded the ranking dataset to {rr['train_examples']:,} examples, approximately {rr['train_examples'] / rb['train_examples']:.1f} times
its original size, to test whether additional training data improved performance.</p>
<p><b>Note:</b> Candidates labelled “no” during training may include real metabolites that are
not recorded for that drug in the datasets.</p>

<h3>2.2 Model configurations and comparison goals</h3>
<p>We compared seven configurations to test whether fine-tuning improves metabolite prediction,
how generation compares with candidate ranking, and whether additional training data helps.
All configurations used Qwen3-8B, the same test drugs and the same evaluation procedure.</p>
<table><thead><tr><th class=idx>#</th><th>Configuration</th><th>Method and purpose</th></tr></thead><tbody>
{CONFIG_ROWS}
</tbody></table>
<p>All ranking configurations used the same rule-generated candidate lists. Configurations 3 and 4
scored candidate structure strings with an adjustment for string length.</p>
<p>An additional untuned control used configuration 2’s prompt and sampling settings to assess the
effect of fine-tuning. Configurations 3 and 6 differed in training, scoring and checkpoint selection,
so their comparison measures the combined effect of these changes. Adding GOSTAR also increased
the number of training updates.</p>

<h3>2.3 Choosing the saved model</h3>
<p>We selected generation and ranking checkpoints with the highest recall@5 on fixed
development subsets. Recall@5 measures the fraction of recorded metabolites recovered within
five predictions. Cross-entropy loss measures how well the model predicts the tokens in the
recorded answer strings.</p>
<p>The figure shows how the generator’s performance changed during training. We measured
development loss on 160 examples and recall@5 on a separate subset of {ft['probe_molecules']} parent structures
containing {PROBE_PAIRS} recorded drug&ndash;metabolite pairs. Development loss was lowest after one
epoch, when the model recovered <b>{FT_LOW['dev_recall5']:.1%}</b> of the recorded metabolites within its five
predictions. Recovery reached <b>{FT_PICK['dev_recall5']:.1%} after four epochs</b>, an improvement of
<b>{(FT_PICK['dev_recall5'] - FT_LOW['dev_recall5']) * 100:.1f} percentage points</b>. We selected this four-epoch checkpoint because it recovered
the most recorded metabolites on the development subset.</p>
<p>During this period, training loss continued to fall and development loss increased. This suggests
that the model was becoming more closely fitted to its training examples, with poorer prediction
of the development answer strings. The improvement in metabolite recovery therefore needs to be
considered alongside this evidence of possible overfitting.</p>

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
Known-reference precision@5 cannot exceed {PREC_CEIL:.1%} on this test set, because most drugs
have fewer than five recorded metabolites. Additional valid metabolites can occupy slots without
receiving credit.
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
S-oxidation. The rules produce {EX[PRO]['n_candidates']} candidates, and each configuration returns five.</p>
<div class=mols><figure>{mol(PRO,190,140)}<figcaption>parent</figcaption></figure>
<div class=arrow>&rarr;</div>{PRO_MOLS}</div>
<p class=cap><b>Figure 3. Prochlorperazine and its three recorded metabolites.</b></p>
<table aria-label="Metabolites recovered for prochlorperazine">
<thead><tr><th scope=col>Configuration</th><th class=n scope=col>Metabolites found</th></tr></thead>
<tbody>{PRO_ROWS}</tbody></table>
<p>The fine-tuned generator and the generator reused for ranking each found the hydroxylation
and sulfoxide products. The trained ranker also recovered the N-demethylation product.</p>
<p><a href="docs/examples.html">Worked examples with prompts and structures &rarr;</a></p>

<h3>3.2 Additional data and candidates</h3>
<p>Adding GOSTAR increased the ranking dataset approximately 2.5-fold and moved recall from
<b>{at(5, A_RANK):.1%} to {at(5, A_GOSTAR):.1%}</b>, recovering two additional metabolites.
Both runs reached their highest measured development recall around 1.5 training epochs.
Each configuration was trained once, so the reliability of this small gain remains uncertain.
The selected checkpoints each recovered 33 of 68 development pairs, the maximum of nine
checkpoint evaluations per run. Total optimizer updates were 2,020 and 5,060, including smoke runs.</p>
<figure class=fig id=figure-s1>
<img src="{png('reranker_curves.png')}" alt="Development recall during ranking-model training with the base dataset and the GOSTAR-expanded dataset">
<figcaption><b>Figure 4. Ranking-model training.</b> Development recall@5 on 40 drug structures containing 68 recorded pairs.
The base dataset contains {rb['train_examples']:,} examples; the GOSTAR-expanded dataset
contains {rr['train_examples']:,}. Each curve represents one training run.</figcaption>
</figure>
<h3>3.3 Metabolite coverage and recall by <a href="rules/metabolism_rules.csv">SMIRKS reaction rule</a></h3>
<p>The <a href="rules/metabolism_rules.csv">42 SMIRKS rules</a> did not cover <b>86 of 309 recorded metabolites</b>, leaving those answers
unavailable to the ranker. For the base ranker, the law of total probability gives:</p>
<p class=math-line>46.0% overall recall@5 = 72.2% rule coverage &times; 63.7% recall among reachable metabolites</p>
<p>Expanding the rules to cover common missing transformations could make more metabolites
available for prediction. Any increase in overall recall would also depend on how accurately the
ranker orders the additional candidates. Among the 24 missed aliphatic hydroxylations, 13 had
another product of the same rule in the top five, with hydroxylation at a different position.
This error slice motivates testing whether site-of-metabolism predictions improve candidate
ordering.</p>

<figure class=fig id=figure-5>
<img src="{png('misses_by_chemistry.png')}" alt="Recorded metabolites recovered, ranked below the top five or absent from the candidate lists, grouped by reaction type">
<figcaption><b>Figure 5. Recovery by reaction type.</b> Bars distinguish recovered
metabolites, candidates missed by ranking, and metabolites absent from the candidate lists.
Pairs are grouped by the first matching rule. This analysis uses the later GOSTAR scoring pass,
which recovered 145 of 309 metabolites ({P_BASE:.1%}).</figcaption>
</figure>

<h3>3.4 Using Site of Metabolism (SOM) Information</h3>
<p>A site of metabolism (SOM) identifies where a chemical change occurs in the parent molecule.
For each drug, we used sites recorded for its known metabolites to keep only candidates generated
by rules acting at those sites. The same ranker then ordered the remaining candidates, increasing
recall@5 from {som['arm 6, no site information']:.1%} to {som['true sites, union per parent']:.1%}.
Knowing the reaction sites therefore substantially improved prediction in this experiment.</p>
<p>We then used a separate model to predict the sites. Filtering with these predictions kept only
105 of 223 reachable reference metabolites and reduced recall@5 to
{som['sites predicted by the model']:.1%}. This suggests that the site predictions were too
incomplete or inaccurate to use as a hard filter.</p>
<p>We then kept all candidates and used the predicted sites as a soft prior. Candidates supported
by a predicted site received a score bonus controlled by the prior weight, λ = {P_PICK['lambda']:g}:</p>
<div class=som-equation>
{SOM_PRIOR_EQUATION}
</div>
<p>The indicator is 1 when a candidate's reacting atoms overlap a predicted site, and 0 otherwise.</p>
<p>The prior increased recall@5 from {P_BASE:.1%} to {P_PICK['holdout_recall5']:.1%}, a gain of
approximately one percentage point. Its 95% confidence interval spans
{format(PRIOR_INTERVAL['ci_low'], '.1f').replace('-', '−')} to +{PRIOR_INTERVAL['ci_high']:.1f}
percentage points and includes zero, so the improvement remains uncertain. The interval resamples
154 standardised parent clusters, retaining all reference rows for each sampled drug. It is
conditional on the saved predictions and excludes variation from retraining or rule selection.</p>
</section>
"""

PAGE += f"""
<section id=discussion>
<h2>4. Discussion and next steps</h2>

<h3>4.1 Chemistry: candidate coverage and reaction sites</h3>
<p>The current rules cannot reach 27.8% of reference pairs. We would repair incorrect rule products
and expand missing transformations, starting with hydroxylation. Each rule should pass atom-mapping
and product checks. We would also review the ranker's top-five errors to build harder training
examples, then test training it to compare candidates from the same drug. Unrecorded products
would remain uncertain labels unless further evidence resolves their status.</p>
<p>Known reaction sites improved recall. Filtering with predicted sites removed many reachable
answers. We would improve site prediction and train the ranker with site-marked parent structures,
keeping all candidates. Training-site predictions should come from models that did not train on
those drugs, so the ranker learns from realistic prediction errors.</p>

<h3>4.2 AI engineering: molecular representation and training</h3>
<p>Our small pilot added spaces between SMILES units using Qwen's existing vocabulary. It showed
no clear recall gain and used more tokens. A short output-budget check would compare 96- and 192-token limits
on the same generator with matched sampling seeds. We would then repeat the representation test
with longer training and several training seeds, checking that training and inference use the
same encoding.</p>
<p>After the data and evaluation checks, experiments could test learned chemical fragments as tokens, as in
<a href="https://doi.org/10.1021/acs.jcim.0c01127">SMILES Pair Encoding</a>, with compatible model
embeddings and output layers. We would also test a short training stage on molecular structure
tasks, such as ring counting, before fine-tuning the ranker, following
<a href="https://aclanthology.org/2025.emnlp-main.791/">CLEANMOL</a>. Their benefit for metabolite
ranking remains untested. Compare each change against the same baseline, tracking recall, valid
outputs, runtime and token cost.</p>

<p id=next><b>Three-month sequence:</b> Month 1 would cover data and rule repairs, a reserved evaluation
set, and reviewed ranking errors. Month 2 would test learned SOM inputs and selected LLM changes,
including a small RL pilot. Month 3 would repeat promising runs and evaluate the fixed pipeline
on the reserved set.</p>
</section>

<footer>
<p><b>Time spent:</b> about 30 to 40 hours over four to five days. The two ranking fine-tunes
account for roughly 13 of those on their own and ran unattended.</p>
</footer>

</div></body></html>"""

out = ROOT / "WRITEUP.html"
out.write_text(PAGE)
import re
body = re.sub(r"<[^>]+>", " ", PAGE)
print(f"wrote {out.relative_to(ROOT)}  ({len(PAGE)/1024:.0f} KB, {len(body.split())} words)")
