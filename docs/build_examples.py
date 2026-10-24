"""Build docs/examples.html from docs/examples.json: prompt, responses, every molecule drawn.\n\n    python3 docs/build_examples.py\n"""
import json, sys, pathlib, html
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from src import evaluation as ev
from src.inference import PROMPT as RANKER_PROMPT
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Draw import rdMolDraw2D

EX=json.loads((ROOT/"docs"/"examples.json").read_text())
test=pd.read_csv(ROOT/'data/test_set.csv', dtype=str, keep_default_na=False, na_values=[])
test["parent"]=[ev.canonical(s) for s in test["parent_smiles"]]
test["metabolite"]=[ev.canonical(s) for s in test["metabolite_smiles"]]
truth={p:[r.metabolite for r in test[test["parent"]==p].itertuples()] for p in EX}

NAMES={"CCC1(c2ccccc2)C(=O)NC(=O)NC1=O":("Phenobarbital","aromatic hydroxylation"),
       "CN[C@@H](C)[C@@H](O)c1ccccc1":("Ephedrine","N-demethylation"),
       "NC(=O)NS(=O)(=O)c1ccc(Cl)cc1":("4-chlorobenzenesulfonylurea","hydrolysis"),
       "CN1CCN(CCCN2c3ccccc3Sc3ccc(Cl)cc32)CC1":("Prochlorperazine","three recorded metabolites")}
# Configuration numbers match the writeup; saved experiment keys retain their original names.
CONFIGURATIONS=[("arm1",1,"Untuned, generative (control)"),
                ("arm3",2,"Fine-tuned, generative"),
                ("arm4",3,"Fine-tuned generative used as ranker"),
                ("arm6",6,"Fine-tuned to rank")]

def scoring_math(key):
    """Explain the recorded scoring procedure with clearly illustrative arithmetic."""
    if key == "arm4":
        return '''<div class="scoremath">
          <h3>How the score is calculated</h3>
          <p>The supplied completion is split into tokens.
          For each token, the model evaluates its probability given the prompt and the
          earlier completion tokens. The API returns the natural logarithm of that probability.</p>
          <pre class="equation">ℓᵢ = ln P(token i | prompt, earlier completion tokens)
score = (ℓ₁ + ℓ₂ + … + ℓₙ) / n</pre>
          <p>Python averages the returned completion log probabilities. The closing
          &lt;/metabolite&gt; tag contributes tokens to this average. Prompt tokens and
          an end-of-sequence token are excluded. A higher score, closer to zero, ranks first.</p>
          <details><summary>Worked arithmetic and the scoring code</summary>
            <p><b>Illustrative values for a hypothetical three-token completion.</b></p>
            <pre>Token probabilities:       0.80,   0.50,   0.25
Natural log probabilities: −0.223, −0.693, −1.386
Mean log probability: (−0.223 − 0.693 − 1.386) / 3 ≈ −0.768</pre>
            <p>The prompt and completion are tokenized separately and concatenated.
            The scoring code then takes the completion entries from the API result:</p>
            <pre>tail = [x for x in fut.result()[-n_completion:] if x is not None]
score = float(np.mean(tail)) if tail else -1e9</pre>
            <p>Here, n is the number of returned completion entries after missing values
            are removed. The fallback score is used if there are no usable entries.</p>
            <p>Source: <a href="../notebooks/04_arms.ipynb">notebook 04</a>,
            score_candidates.</p>
          </details>
        </div>'''
    if key == "arm6":
        return '''<div class="scoremath">
          <h3>How the score is calculated</h3>
          <p>The model evaluates the next-token probabilities of “yes” and “no” after
          the same parent–candidate prompt. Each verdict is one token. The API returns
          their natural log probabilities, and Python subtracts them:</p>
          <pre class="equation">score = ln P(yes | prompt) − ln P(no | prompt)
      = ln [P(yes | prompt) / P(no | prompt)]</pre>
          <p>A positive score means “yes” has greater probability than “no”. A negative
          score means “no” has greater probability. Candidates are sorted by descending
          score. These scores have not been calibrated as biological probabilities.</p>
          <details><summary>Worked arithmetic and the scoring code</summary>
            <p><b>Illustrative values for one hypothetical candidate.</b></p>
            <pre>P(yes | prompt) = 0.80     ln P(yes | prompt) ≈ −0.223
P(no  | prompt) = 0.10     ln P(no  | prompt) ≈ −2.303
score = −0.223 − (−2.303) ≈ 2.08
      = ln(0.80 / 0.10) = ln(8)</pre>
            <p>In this illustration, “yes” is eight times as probable as “no”.
            The probabilities come from the full token vocabulary, so other tokens
            can carry some probability too.</p>
            <p>The scoring code uses only the final token entry from each API result:</p>
            <pre>score = float(f_yes.result()[-1]) - float(f_no.result()[-1])</pre>
            <p>Source: <a href="../notebooks/05_reranker.ipynb">notebook 05</a>,
            score_candidates.</p>
          </details>
        </div>'''
    return ""

def prompt_panel(key, parent, data):
    """Show generation prompts and separate ranking prefixes from scored completions."""
    if key == "arm1":
        prompt = data["arm1_prompt"]
        note = ("Only configuration 1 includes the expert instruction and six worked "
                "examples before the parent molecule.")
        extra = ""
        summary = "Full prompt, including six worked examples"
    elif key == "arm3":
        prompt = data["arm3_prompt"]
        note = ("This is the complete prompt, with no expert instruction or worked examples. "
                "The fine-tuned generator continues it with a metabolite structure.")
        extra = ""
        summary = "Prompt"
    elif key == "arm4":
        prompt = data["arm3_prompt"]
        completion = data[key][0] + "</metabolite>"
        extra = (f'<div class="lbl">scored completion for the first-ranked candidate</div>'
                 f'<pre>{html.escape(completion)}</pre>')
        note = ("Uses the same prompt and trained generator as configuration 2. "
                f'Each of the {data["n_candidates"]} rule-generated candidates is scored by '
                "mean log probability across its completion tokens, including the closing tag. "
                "Higher scores rank first.")
        summary = "Prompt and candidate scoring"
    else:
        prompt = RANKER_PROMPT.format(parent=parent, candidate=data[key][0])
        extra = '<div class="lbl">scored next-token completions</div><pre>yes\nno</pre>'
        note = ("The prompt above uses the first-ranked candidate. "
                f'The ranker evaluates all {data["n_candidates"]} rule-generated candidates '
                "individually using log P(yes) − log P(no). Higher scores rank first.")
        summary = "Prompt and candidate scoring"
    return (f'<details class="promptblock"{" open" if key != "arm1" else ""}>'
            f'<summary>{summary}</summary><div class="promptcontent">'
            f'<div class="lbl">prompt</div><pre>{html.escape(prompt)}</pre>'
            f'{extra}<p class="note">{html.escape(note)}</p>{scoring_math(key)}</div></details>')

def svg(smiles, w=150, h=110):
    m=Chem.MolFromSmiles(smiles)
    if m is None: return '<div class="bad">unparseable</div>'
    d=rdMolDraw2D.MolDraw2DSVG(w,h); o=d.drawOptions()
    o.clearBackground=False; o.bondLineWidth=1.4
    rdMolDraw2D.PrepareAndDrawMolecule(d,m); d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>","")

def card(smiles, match, rank=None, n_mets=1):
    """match is the metabolite number this prediction equals, or None."""
    cls="card hit" if match else "card"
    tag=f'<span class="rank">{rank}</span>' if rank else ""
    badge=f'<span class="badge">metabolite {match}</span>' if match and n_mets>1 else ""
    return (f'<div class="{cls}">{tag}{badge}{svg(smiles)}'
            f'<code>{html.escape(smiles[:34])}{"…" if len(smiles)>34 else ""}</code></div>')

sections=[]
for p,d in EX.items():
    name,rxn=NAMES.get(p,(p[:28],""))
    keys={ev.match_key(m,"standardized"): i for i,m in enumerate(truth[p],1)}
    nm=len(truth[p])
    rows=[]
    for key,number,label in CONFIGURATIONS:
        preds=d[key]
        found=sorted({keys[ev.match_key(c,"standardized")] for c in preds
                      if ev.match_key(c,"standardized") in keys})
        cards="".join(card(c, keys.get(ev.match_key(c,"standardized")), i+1, nm)
                      for i,c in enumerate(preds)) or '<div class="bad">no usable prediction</div>'
        tally=(f'<span class="tally">{len(found)} of {nm}</span>' if nm>1 else "")
        rows.append(f'''<tr class="{'ok' if found else 'no'}">
          <th scope="row"><div class="lbl">configuration {number}</div><span class="verdict">{'✓' if found else '✗'}</span>{html.escape(label)}{tally}</th>
          <td><div class="strip">{cards}</div>{prompt_panel(key,p,d)}</td></tr>''')
    mets="".join(
        f'<div><div class="lbl">metabolite{" "+str(i) if len(truth[p])>1 else ""}</div>'
        f'{svg(m,180,135)}<code>{html.escape(m)}</code></div>'
        for i,m in enumerate(truth[p],1))
    sections.append(f'''<section>
      <h2>{html.escape(name)} <span class="rxn">{html.escape(rxn)}</span></h2>
      <div class="truthrow">
        <div><div class="lbl">parent</div>{svg(p,190,140)}<code>{html.escape(p)}</code></div>
        <div class="arrow">&rarr;</div>
        {mets}
      </div>
      <table>{"".join(rows)}</table></section>''')

page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Worked examples</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--hit:#2a78d6;--miss:#eb6834;--card:#fff}}
:root:not([data-theme=light]) @media (prefers-color-scheme:dark){{}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#17171a;--ink:#f2f2ef;--muted:#a3a29e;--line:#2e2e33;--card:#1f1f23}}}}
:root[data-theme="dark"]{{--bg:#17171a;--ink:#f2f2ef;--muted:#a3a29e;--line:#2e2e33;--card:#1f1f23}}
*{{box-sizing:border-box}}
body{{margin:0;padding:28px 16px 60px;background:var(--bg);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 6px}} .sub{{color:var(--muted);margin:0 0 30px}}
section{{border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:28px;background:var(--card)}}
h2{{font-size:18px;margin:0 0 14px}} .rxn{{color:var(--muted);font-weight:400;font-size:14px}}
.truthrow{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding-bottom:14px;border-bottom:1px solid var(--line)}}
.truthrow>div{{text-align:center}} .arrow{{font-size:22px;color:var(--muted)}}
.lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:4px}}
code{{display:block;font-size:10.5px;color:var(--muted);word-break:break-all;max-width:220px;margin:4px auto 0}}
table{{width:100%;table-layout:fixed;border-collapse:collapse;margin-top:14px}}
th{{text-align:left;vertical-align:top;width:210px;padding:12px 12px 12px 0;font-size:13.5px;font-weight:600;border-top:1px solid var(--line)}}
th em{{display:block;font-weight:400;font-style:normal;color:var(--muted);font-size:12px;margin-top:2px}}
td{{padding:12px 0;border-top:1px solid var(--line)}}
.verdict{{display:inline-block;width:18px;font-weight:700}}
tr.ok .verdict{{color:var(--hit)}} tr.no .verdict{{color:var(--miss)}}
.strip{{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}}
.card{{position:relative;border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--bg);flex:0 0 auto;text-align:center}}
.card.hit{{border:2px solid var(--hit);background:color-mix(in srgb,var(--hit) 8%,var(--bg))}}
.card code{{max-width:150px}}
.rank{{position:absolute;top:4px;left:6px;font-size:10px;color:var(--muted)}}
.badge{{position:absolute;top:4px;right:6px;font-size:9.5px;color:var(--hit);font-weight:600}}
.tally{{margin-left:6px;font-weight:400;color:var(--muted);font-size:12px}}
.bad{{color:var(--miss);font-size:12.5px;padding:18px 10px}}
details{{margin-top:10px}} summary{{cursor:pointer;color:var(--muted);font-size:12.5px}}
.promptblock{{margin-top:16px;padding:14px;border:1px solid var(--line);border-radius:10px;background:var(--bg)}}
.promptblock summary{{color:var(--ink);font-weight:600}}
.promptcontent{{margin-top:12px}}
.scoremath{{margin-top:16px;padding-top:14px;border-top:1px solid var(--line);font-size:12.5px}}
.scoremath h3{{font-size:13px;margin:0 0 8px}}
.scoremath p{{margin:8px 0}}
.scoremath .equation{{font-size:12.5px;line-height:1.7}}
.note{{color:var(--muted);font-size:12.5px;margin:0}}
pre{{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;
 font-size:11.5px;white-space:pre-wrap;word-break:break-word;margin:4px 0 12px}}
svg{{max-width:100%;height:auto}}
.foot{{color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px}}
@media (max-width:700px){{th{{width:auto;display:block;border-top:none;padding-bottom:4px}}td{{display:block;padding-top:0}}}}
</style></head><body><div class="wrap">
<h1>Worked examples</h1>
{"".join(sections)}
</div></body></html>'''
out=ROOT/'docs'/'examples.html'
out.write_text(page)
print(f"wrote {out} ({len(page)/1024:.0f} KB)")
