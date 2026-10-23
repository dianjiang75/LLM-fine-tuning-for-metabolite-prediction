"""Interface to the metabolite predictor.

Run from the project root:

    streamlit run app.py

One parent molecule goes in as SMILES. The rule set enumerates every metabolite it can produce,
the fine-tuned ranker scores each one, and the top k come back with the transformation that
produced them. ``src/inference.py`` holds the pipeline; this file is presentation only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # so the app runs from any working directory
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from src.evaluation import match_key
from src.inference import PROMPT, InvalidInput, Pipeline, RankerUnavailable

RDLogger.DisableLog("rdApp.*")

# Named examples, each checked against its molecular formula. The first four carry saved scores,
# so they work with no API key; the last two have to be scored live.
EXAMPLES = {
    "Phenobarbital": "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Imipramine": "CN(C)CCCN1c2ccccc2CCc2ccccc21",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
}

MAX_K = 10

# Holdout pair recall, from docs/arm_results.csv, arm 6 with the expanded training mix.
HOLDOUT = {"recall@1": "18.1%", "recall@5": "46.6%", "any hit in 5": "63.2%"}


@st.cache_resource(show_spinner=False)
def pipeline() -> Pipeline:
    return Pipeline()


# Membership refers to the active ranker's examples. Reference structures have separate sources.
SPLIT_NOTE = {
    "holdout": "From the supplied benchmark. Its parents were excluded from ranker fitting and "
               "checkpoint selection. Rule development inspected its transformation misses, "
               "so performance on this set is exploratory.",
    "validation": "From the held-out validation pool. A subset of this pool selected the "
                  "ranker checkpoint; its parents were excluded from fitting the weights.",
    "training": "This parent appears in the shipped ranker training examples. A hit here may "
                "reflect that training exposure.",
}


@st.cache_data(show_spinner=False)
def recorded_metabolites() -> dict[tuple, list[dict]]:
    """Reference structures and ranker exposure, indexed by parent heavy-atom composition.

    The generator's source references remain useful for display. Actual ranker exposure comes
    from rerank_train.jsonl, including its GOSTAR additions. A source reference does not by itself
    establish that the ranker trained on that parent or pair. A composition index cheaply narrows
    the lookup; the selected parent's matches are confirmed with the scorer's standardisation.
    Full tautomer normalisation of thousands of parents would delay every app startup.
    """
    found: dict[str, dict] = {}

    def record(parent: str) -> dict:
        return found.setdefault(parent, {
            "memberships": set(), "references": {}, "trained_positive_smiles": set(),
        })

    def add_reference(parent: str, metabolite: str, source: str) -> None:
        item = record(parent)
        reference = item["references"].setdefault(metabolite, {"smiles": metabolite, "sources": set()})
        reference["sources"].add(source)

    test_path = ROOT / "data" / "test_set.csv"
    if test_path.exists():
        with test_path.open() as handle:
            for row in csv.DictReader(handle):
                item = record(row["parent_smiles"])
                item["memberships"].add("holdout")
                add_reference(row["parent_smiles"], row["metabolite_smiles"], "supplied benchmark")

    for filename, source in (("dev.jsonl", "validation references"),
                             ("train.jsonl", "base training references")):
        path = ROOT / "data" / "processed" / filename
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                item = record(row["parent"])
                if filename == "dev.jsonl":
                    item["memberships"].add("validation")
                add_reference(row["parent"], row["target"], source)

    ranker_path = ROOT / "data" / "processed" / "rerank_train.jsonl"
    if ranker_path.exists():
        with ranker_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                item = record(row["parent"])
                item["memberships"].add("training")
                if row["label"] == "yes":
                    item["trained_positive_smiles"].add(row["candidate"])
                    add_reference(row["parent"], row["candidate"], "ranker positive examples")

    index: dict[tuple, list[dict]] = {}
    for parent, item in found.items():
        bucket = parent_bucket(parent)
        if bucket is None:
            continue
        index.setdefault(bucket, []).append({
            "parent": parent,
            "memberships": sorted(item["memberships"]),
            "trained_positive_smiles": sorted(item["trained_positive_smiles"]),
            "references": [
                {"smiles": ref["smiles"], "sources": sorted(ref["sources"])}
                for _, ref in sorted(item["references"].items())
            ],
        })
    return index


@lru_cache(maxsize=20_000)
def parent_bucket(smiles: str) -> tuple | None:
    """Heavy-atom counts are unchanged by the scorer's normalization and tautomer policy."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = Counter(atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    return tuple(sorted(counts.items()))


def reference_record_for_parent(index: dict, parent: str) -> dict:
    """Confirm identity for the small composition bucket, then combine its source records."""
    parent_key = match_key(parent, "standardized")
    result = {"memberships": set(), "references": [], "trained_positive_smiles": set()}
    if parent_key is None:
        return result
    for item in index.get(parent_bucket(parent), []):
        if match_key(item["parent"], "standardized") != parent_key:
            continue
        result["memberships"].update(item["memberships"])
        result["references"].extend(item["references"])
        result["trained_positive_smiles"].update(item["trained_positive_smiles"])
    result["memberships"] = sorted(result["memberships"])
    return result


def reference_view(record: dict) -> tuple[list[dict], set[str]]:
    """Merge equivalent product references for the parent currently being displayed."""
    merged: dict[str, dict] = {}
    for reference in record.get("references", []):
        key = match_key(reference["smiles"], "standardized")
        if key is None:
            continue
        item = merged.setdefault(key, {"smiles": reference["smiles"], "sources": set()})
        item["sources"].update(reference["sources"])
    positives = {match_key(smiles, "standardized")
                 for smiles in record.get("trained_positive_smiles", [])}
    positives.discard(None)
    rows = [{"smiles": item["smiles"], "sources": sorted(item["sources"])}
            for _, item in sorted(merged.items())]
    return rows, positives


def membership_note(record: dict) -> str:
    """Explain exposure without inferring it from the presence of a reference answer."""
    if not (ROOT / "data" / "processed" / "rerank_train.jsonl").exists():
        return "The ranker training records are unavailable, so training membership is unverified."
    memberships = record.get("memberships", [])
    if len(memberships) > 1:
        return "The shipped membership records overlap across splits and need review."
    if memberships:
        return SPLIT_NOTE[memberships[0]]
    return "No membership was found in the shipped ranker training, validation or benchmark records."


@st.cache_data(show_spinner=False, ttl=3600)
def expiry_note() -> tuple[str, bool] | None:
    """When live scoring stops working, as a line to show and whether it is close.

    Checked against the service each hour rather than read from the run file, so a lifetime
    changed after the run was recorded is reflected here.
    """
    try:
        expires = pipeline().live.expires_at()
    except RankerUnavailable:
        return None
    if expires is None:
        return "The checkpoint has no expiry set.", False

    remaining = (expires - datetime.now(timezone.utc)).total_seconds()
    # Spelled out rather than %-d, which is not portable off Unix.
    date = f"{expires.day} {expires:%B %Y}"
    if remaining <= 0:
        return f"The checkpoint expired on {date}. Saved scores still work.", True

    # Rounded, so the first day of a 30-day lifetime reads as 30 rather than 29.
    days = round(remaining / 86400)
    close = remaining <= 7 * 86400
    if days == 0:
        return f"Live scoring stops today, {date}.", True
    return (
        f"Live scoring works until {date}, {days} day{'' if days == 1 else 's'} from now.",
        close,
    )


@st.cache_data(show_spinner=False, max_entries=64)
def scored(smiles: str, source: str) -> dict:
    """Every candidate's score for one molecule, cached on the molecule and the source.

    Scored to the largest k the interface offers and truncated for display, so moving the
    slider, or asking for the same molecule again, costs no further calls to the model.

    Returned as plain types rather than the pipeline's own dataclass. The cache stores what it
    is given by pickling it, and a class redefined while the app is running no longer matches
    what was stored, which fails the whole request.
    """
    result = pipeline().predict(smiles, source=source, k=MAX_K)
    return {
        "parent": result.parent,
        "n_candidates": result.n_candidates,
        "source": result.source,
        "checkpoint": result.checkpoint,
        "elapsed_seconds": result.elapsed_seconds,
        "predictions": [
            {"rank": p.rank, "smiles": p.smiles, "score": p.score, "rule": p.rule}
            for p in result.predictions
        ],
    }


@st.cache_data(show_spinner=False)
def svg(smiles: str, width: int = 420, height: int = 240) -> str:
    """The molecule as an inline SVG, on an opaque background so it reads under either theme."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.08
    options.bondLineWidth = 2
    options.setBackgroundColour((1, 1, 1, 1))
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    drawing = drawer.GetDrawingText()
    return drawing[drawing.index("<svg"):]


def card(body: str, accent: bool = False) -> None:
    """A molecule on a light panel, outlined when it matches a recorded metabolite."""
    border = "2px solid #2e7d32" if accent else "1px solid #e3e3e3"
    st.markdown(
        f'<div style="background:#fff;border:{border};border-radius:8px;'
        f'padding:4px;display:flex;justify-content:center">{body}</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Metabolite prediction", page_icon="⚗️", layout="wide")

pipe = pipeline()
truth = recorded_metabolites()

with st.sidebar:
    st.subheader("Model")
    st.caption(
        "Qwen3-8B with a rank-32 LoRA, fine-tuned to answer yes or no to a parent and candidate "
        "pair. Checkpoint 1890 of 5040, selected on the validation split."
    )
    for label, value in HOLDOUT.items():
        st.metric(label, value)
    st.caption(
        "Pair recall on the 309-pair exploratory benchmark, scored on standardised SMILES equality. "
        "The 42 SMIRKS rules reach 72.2% of recorded answers; five prediction slots limit their "
        "best possible recall to 71.5%."
    )

    st.divider()
    st.subheader("Scoring")
    source = st.radio(
        "Where scores come from",
        options=["live", "saved"],
        index=0 if pipe.live.configured() else 1,
        format_func=lambda value: {"live": "Call the model", "saved": "Replay saved scores"}[value],
        help="Replayed scores are the same checkpoint's output, recorded for 215 molecules.",
    )
    if source == "live" and not pipe.live.configured():
        st.warning("No TINKER_API_KEY found, so live calls will fail.")
    if source == "saved":
        st.caption(f"{len(pipe.saved.parents())} molecules have saved scores.")

    note = expiry_note() if source == "live" else None
    if note:
        message, close = note
        (st.warning if close else st.caption)(message)

    k = st.slider("Predictions to return", min_value=1, max_value=MAX_K, value=5)

st.title("Metabolite prediction")
st.write(
    "Enter a parent drug as SMILES. Every metabolite the rule set can make from it is scored by "
    "the fine-tuned ranker, and the highest scoring come back first."
)

if "smiles" not in st.session_state:
    st.session_state.smiles = EXAMPLES["Imipramine"]

st.caption("Examples")
columns = st.columns(len(EXAMPLES))
for column, (name, example) in zip(columns, EXAMPLES.items()):
    if column.button(name, use_container_width=True):
        st.session_state.smiles = example

with st.form("predict"):
    text = st.text_input("Parent SMILES", key="smiles")
    submitted = st.form_submit_button("Predict metabolites", type="primary")

if submitted or st.session_state.get("ran"):
    st.session_state.ran = True
    try:
        with st.spinner("Enumerating rule candidates and scoring them"):
            result = scored(text, source)
    except (InvalidInput, RankerUnavailable) as error:
        st.error(str(error))
    except Exception as error:  # nothing reaches the page as a traceback
        st.error(f"The prediction did not complete ({type(error).__name__}: {error}).")
    else:
        parent = result["parent"]
        predictions = result["predictions"][:k]
        predicted_keys = {match_key(p["smiles"], "standardized") for p in predictions}
        reference_record = reference_record_for_parent(truth, parent)
        reference_rows, trained_positive_keys = reference_view(reference_record)
        known = [row["smiles"] for row in reference_rows]
        known_keys = {match_key(m, "standardized") for m in known}

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Parent")
            card(svg(parent))
            st.code(parent, language=None)
            st.caption(
                f"{result['n_candidates']} rule candidates scored in "
                f"{result['elapsed_seconds']:.1f}s, "
                + ("by calling the model" if result["source"] == "live" else "from saved scores")
            )
            st.caption(membership_note(reference_record))

        with right:
            st.subheader(f"Top {len(predictions)}")
            if not predictions:
                st.info("The rule set produced nothing for this molecule.")
            for prediction in predictions:
                hit = match_key(prediction["smiles"], "standardized") in known_keys
                structure, detail = st.columns([1, 1.4])
                with structure:
                    card(svg(prediction["smiles"], width=300, height=170), accent=hit)
                with detail:
                    heading = f"**{prediction['rank']}.** {prediction['rule']}"
                    if hit:
                        heading += " · recorded metabolite"
                    st.markdown(heading)
                    st.code(prediction["smiles"], language=None)
                    st.caption(f"score {prediction['score']:+.2f} log-odds")
                st.divider()

        if known:
            found = sum(1 for m in known if match_key(m, "standardized") in predicted_keys)
            summary = (
                f"{found} of this molecule's {len(known)} recorded metabolites "
                f"{'is' if found == 1 else 'are'} in the top {len(predictions)}."
                if found else
                f"None of this molecule's {len(known)} recorded metabolites are in the "
                f"top {len(predictions)}."
            )
            (st.success if found else st.info)(summary)

            st.subheader(f"Recorded metabolites ({len(known)})")
            st.caption("Reference structures are grouped by standardised identity. Each source is shown below.")
            for reference in reference_rows:
                metabolite = reference["smiles"]
                key = match_key(metabolite, "standardized")
                rank = next((p["rank"] for p in predictions
                             if match_key(p["smiles"], "standardized") == key), None)
                structure, detail = st.columns([1, 1.4])
                with structure:
                    card(svg(metabolite, width=300, height=170), accent=rank is not None)
                with detail:
                    st.markdown(
                        f"**Predicted at rank {rank}**" if rank else "**Not predicted**"
                    )
                    st.code(metabolite, language=None)
                    st.caption("Reference source: " + ", ".join(reference["sources"]) + ".")
                    if key in trained_positive_keys:
                        st.caption("This parent/product identity is positive in the ranker training records.")
                st.divider()
        elif predictions:
            st.info(
                "No reference metabolite was found for this molecule in the shipped reference records."
            )

        with st.expander("What the model was asked"):
            st.caption(
                "One call per candidate. The score is the log probability the model assigns to "
                "yes minus the log probability it assigns to no, both conditioned on this prompt."
            )
            if predictions:
                st.code(
                    PROMPT.format(parent=parent, candidate=predictions[0]["smiles"]) + "yes | no",
                    language=None,
                )
            st.caption(f"Checkpoint: {result['checkpoint']}")

        if predictions:
            rows = "rank,smiles,score,rule\n" + "\n".join(
                f'{p["rank"]},"{p["smiles"]}",{p["score"]:.4f},"{p["rule"]}"' for p in predictions
            )
            st.download_button(
                "Download predictions as CSV",
                data=rows,
                file_name="metabolite_predictions.csv",
                mime="text/csv",
            )
