"""Build docs/misses_by_chemistry.png: the 309 holdout pairs grouped by the rule that makes the
recorded metabolite, scored against the best ranker's top five.

    python3 docs/build_misses_figure.py
"""
import collections, json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import evaluation as ev
from src import decoding as dec
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

TOP_N = 10
BLUE, PALE, GREY = "#1f6fd0", "#aecbf0", "#d6d4cf"
INK, MUTED, LINE, SURFACE = "#0b0b0b", "#5c5b57", "#e2e1dd", "#fcfcfb"

rules = pd.read_csv(ROOT / "rules/metabolism_rules.csv")
COMPILED = []
for _, r in rules.iterrows():
    rx = AllChem.ReactionFromSmarts(r["smirks"]); rx.Initialize()
    COMPILED.append((str(r["name"]), rx, bool(r["keep_largest_fragment"])))

def largest(m):
    f = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    return max(f, key=lambda x: x.GetNumHeavyAtoms()) if len(f) > 1 else m

def products_by_rule(parent, cap=200):
    """rule name -> the set of molecules that rule makes from this parent."""
    mol = Chem.MolFromSmiles(parent)
    out = collections.defaultdict(set)
    if mol is None: return out
    for name, rx, keep in COMPILED:
        try: prods = rx.RunReactants((mol,))
        except Exception: continue
        n = 0
        for tup in prods:
            for p in tup:
                try:
                    q = largest(p) if keep else p
                    Chem.SanitizeMol(q); out[name].add(Chem.MolToSmiles(q, canonical=True))
                except Exception: pass
                n += 1
                if n > cap: break
            if n > cap: break
    return out

test = pd.read_csv(ROOT / "data/test_set.csv", dtype=str, keep_default_na=False, na_values=[])
test["parent"] = [ev.canonical(s) for s in test["parent_smiles"]]
test["metabolite"] = [ev.canonical(s) for s in test["metabolite_smiles"]]
scores = json.loads((ROOT / "SOM/som_candidate_scores.json").read_text())["test"]
top5 = {p: dec.rank_candidates(list(s), [s[c] for c in s], p, k=5).predictions
        for p, s in scores.items()}
BY_RULE = {p: products_by_rule(p) for p in sorted(set(test["parent"]))}

# found / on the list but outside the top five / no rule proposes it
tally = collections.defaultdict(lambda: [0, 0]); never = 0
alip_miss = alip_same_rule = 0
for r in test.itertuples():
    key = ev.match_key(r.metabolite, "standardized")
    made = [n for n, prods in BY_RULE[r.parent].items()
            if any(ev.match_key(c, "standardized") == key for c in prods)]
    if not made:
        never += 1; continue
    rule = made[0]
    hit = key in {ev.match_key(c, "standardized") for c in top5.get(r.parent, [])}
    tally[rule][0] += hit; tally[rule][1] += 1
    if "aliphatic" in rule.lower() and "hydroxyl" in rule.lower() and not hit:
        alip_miss += 1
        same = {ev.match_key(x, "standardized") for x in BY_RULE[r.parent][rule]}
        if any(ev.match_key(c, "standardized") in same for c in top5.get(r.parent, [])):
            alip_same_rule += 1

ordered = sorted(tally.items(), key=lambda kv: -kv[1][1])
rows = [(name, f, n) for name, (f, n) in ordered[:TOP_N]]
rest = ordered[TOP_N:]
if rest:
    rows.append((f"{len(rest)} other rules", sum(f for _, (f, _) in rest),
                 sum(n for _, (_, n) in rest)))
rows.append(("no rule proposes it", 0, never))
assert sum(n for _, _, n in rows) == len(test), "groups must cover every pair"

PRETTY = {"CH2 -> C=O": "CH\u2082 \u2192 C=O"}
labels = [PRETTY.get(r[0], r[0]) for r in rows][::-1]
found  = [r[1] for r in rows][::-1]
total  = [r[2] for r in rows][::-1]
missed = [t - f for f, t in zip(found, total)]
unreachable = [t if lab == "no rule proposes it" else 0 for lab, t in zip(labels, total)]
missed = [0 if lab == "no rule proposes it" else m for lab, m in zip(labels, missed)]

fig, ax = plt.subplots(figsize=(11, 6.0))
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.barh(labels, found, color=BLUE, height=0.62, label="found in the top five")
ax.barh(labels, missed, left=found, color=PALE, height=0.62, label="not in the top five")
ax.barh(labels, unreachable, left=[f + m for f, m in zip(found, missed)],
        color=GREY, height=0.62, label="never proposed")

for i, (lab, f, t) in enumerate(zip(labels, found, total)):
    x = t + 4
    if lab == "no rule proposes it":
        ax.annotate(f"{t}", xy=(x, i), va="center", color=MUTED, fontsize=11)
    else:
        ax.annotate(f"{f} of {t} found", xy=(x, i), va="center", color=INK, fontsize=11)

ax.set_xlim(0, max(total) * 2.05)
ax.set_xticks([])
ax.tick_params(axis="y", length=0, colors=INK, labelsize=11.5)
for s in ("top", "right", "left", "bottom"): ax.spines[s].set_visible(False)
ax.set_title(f"Outcome of the {sum(total)} holdout pairs, grouped by reaction type",
             color=INK, fontsize=13, fontweight="bold", loc="left", pad=34)
ax.legend(frameon=False, fontsize=10.5, labelcolor=MUTED, ncol=3,
          loc="lower left", bbox_to_anchor=(0, 1.0))
plt.savefig(ROOT / "docs/misses_by_chemistry.png", dpi=160, bbox_inches="tight",
            facecolor=SURFACE)
print(f"wrote docs/misses_by_chemistry.png")
print(f"  {len(rows)} groups covering {sum(total)} pairs, {sum(found)} found")
print(f"  aliphatic hydroxylation: {alip_same_rule} of {alip_miss} misses lost to a different carbon")
