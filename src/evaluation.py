"""Evaluation for metabolite prediction.

The provided harness computes top-k recall by exact canonical-SMILES equality, per test row.
That number is reported here, but on its own it is misleading in three ways that this module
corrects for.

1. **It has a ceiling below 1.0.** A parent receives one list of k predictions, scored separately
   against each of its rows, so a parent's rows can only be satisfied k answers at a time. On the
   provided test set top-1 recall cannot exceed 0.502 under exact matching. Every recall here is
   therefore also reported as a fraction of what is achievable.

2. **Exact string equality counts correct answers as wrong.** The source data is inconsistent
   about tautomers and records stereochemistry that cannot be predicted from the parent alone.
   Three match criteria are reported side by side; the gaps between them are the error analysis.

3. **Per-row scoring is dominated by well-studied parents.** A parent with eight known metabolites
   contributes eight rows. Per-parent views are reported alongside.

Predictions that do not parse are counted and reported separately.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

CRITERIA = ("exact", "nostereo", "standardized")

_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_PARAMS = rdMolStandardize.CleanupParameters()
_TAUTOMER_PARAMS.maxTautomers = 200
_TAUTOMER = rdMolStandardize.TautomerEnumerator(_TAUTOMER_PARAMS)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# RDKit's default parser reads "SMILES<whitespace>NAME", so a refusal such as "I cannot determine
# this." parses as iodine and scores as a prediction while invalid_rate reports nothing wrong.
# src/inference.py guards app input the same way, so both paths agree on what a molecule is.
_PARSER = Chem.SmilesParserParams()
_PARSER.parseName = False


def _parse(smiles: str):
    return Chem.MolFromSmiles(smiles, _PARSER) if smiles else None


# --------------------------------------------------------------------------------------
# Molecule keys
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=200_000)
def canonical(smiles: str, keep_stereo: bool = True) -> str | None:
    """Canonical SMILES, optionally with stereochemistry removed."""
    mol = _parse(smiles)
    if mol is None:
        return None
    if not keep_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=200_000)
def standardize(smiles: str) -> str | None:
    """Normalise functional groups, neutralise charges, pick a canonical tautomer, drop stereo.

    Applied identically to training targets, predictions and references. This key merges
    stereoisomers and some charge and tautomer forms, so matches measure normalized connectivity.
    """
    mol = _parse(smiles)
    if mol is None:
        return None
    try:
        # Canonicalise atom order before tautomer selection to make the key stable across SMILES
        # spellings. Otherwise tied tautomers can be selected differently when atom order changes.
        mol = _parse(Chem.MolToSmiles(mol))
        mol = rdMolStandardize.Normalize(mol)
        mol = _UNCHARGER.uncharge(mol)
        mol = _TAUTOMER.Canonicalize(mol)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def match_key(smiles: str, criterion: str = "standardized") -> str | None:
    """Key under which two molecules are considered the same answer."""
    if criterion == "exact":
        return canonical(smiles, keep_stereo=True)
    if criterion == "nostereo":
        return canonical(smiles, keep_stereo=False)
    if criterion == "standardized":
        return standardize(smiles)
    raise ValueError(f"unknown criterion {criterion!r}, expected one of {CRITERIA}")


@lru_cache(maxsize=100_000)
def _fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return _MORGAN.GetFingerprint(mol) if mol is not None else None


def tanimoto(a: str, b: str) -> float:
    """Morgan-fingerprint similarity, standardising both molecules first.

    Standardising first is required. Fingerprinting raw strings scores paracetamol written as its
    imidic-acid tautomer at 0.345 against itself, below the 0.571 of a genuinely wrong isomer.
    """
    sa, sb = standardize(a), standardize(b)
    fa, fb = _fingerprint(sa), _fingerprint(sb)
    if fa is None or fb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fa, fb)


# --------------------------------------------------------------------------------------
# Ceiling
# --------------------------------------------------------------------------------------

def recall_ceiling(parents: list[str], references: list[str], k: int,
                   criterion: str = "standardized") -> float:
    """Highest per-row recall@k any model could achieve, given one prediction list per parent.

    Recall counts **rows satisfied**, so the ceiling counts rows too. One prediction satisfies
    several rows when their references collapse to the same key, as happens for four parents in
    the provided test set whose rows differ only in stereochemistry.

    Group each parent's rows by answer key, take the ``k`` largest groups, and sum the rows they
    cover. Counting distinct keys instead understates the ceiling.
    """
    groups: dict[str, Counter] = defaultdict(Counter)
    for parent, ref in zip(parents, references):
        key = match_key(ref, criterion)
        if key is not None:
            groups[parent][key] += 1
    best = sum(sum(sorted(c.values(), reverse=True)[:k]) for c in groups.values())
    return best / len(references) if references else 0.0


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

def evaluate(
    parents: list[str],
    references: list[str],
    predictions: list[list[str]],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    criterion: str = "standardized",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Score one prediction list per test row.

    Args:
        parents: parent SMILES, one per row. Rows sharing a parent share a prediction list.
        references: the true metabolite for that row.
        predictions: ranked candidate SMILES for that row's parent, best first.
        k_values: report budgets.
        criterion: one of ``exact``, ``nostereo``, ``standardized``.
        deduplicate: collapse repeated molecules before taking the top k. Sampling a peaked model
            k times usually returns the same string several times, so leaving this off understates
            a generating model badly. The provided harness does not deduplicate.

    Returns:
        A DataFrame indexed by k with columns:

        ``recall_pair``
            fraction of rows whose reference is in the top k. The same form as MetaTrans's
            recall; the match rule differs.
        ``ceiling`` and ``pct_of_ceiling``
            the best achievable at this k, and the fraction of it reached.
        ``recall_parent``
            mean over parents of the fraction of that parent's metabolites found.
        ``parent_hit_rate``
            fraction of parents with at least one metabolite found.
        ``precision_known``
            *distinct* known metabolites appearing in the first k slots, divided by k. A **lower
            bound** on precision, since a prediction absent from the literature is unlabelled
            and may still be right.

            The denominator is k. An arm whose samples are mostly unreadable returns short lists,
            and dividing by list length would score one surviving correct answer as 1.000, the
            same as five out of five. An unusable prediction is a wasted slot.

            The numerator counts distinct molecules, so with ``deduplicate=False`` the same
            correct molecule filling two slots scores 1/k, where counting slots would give 2/k.
            Every reported number uses the default
            ``deduplicate=True``, where no two slots hold the same molecule and the distinction
            does not arise.
        ``tanimoto_wrong``
            mean similarity of the best prediction **for rows that were scored wrong**. The
            conventional figure that mixes in the correct answers carries no information beyond
            recall, because after standardisation a correct prediction scores exactly 1.000, so
            it is just ``recall + (1 - recall) * tanimoto_wrong``.

            Only the two tails of this number mean anything. Above roughly 0.85 only a wrong-site
            isomer reaches, and below 0.40 the prediction is unrelated to the parent. In between
            it cannot separate a wrong-site candidate from the unchanged parent (measured in
            ``docs/FINDINGS.md`` section 5), so use ``parent_echo_rate``.
        ``parent_echo_rate``
            fraction of parents whose top k contains the unchanged parent. Similarity cannot
            reveal this, for the reason above.

            **This is 0 for every arm in this project**, because ``src/decoding.py`` drops a
            prediction equal to the parent before it is ever scored. The metric only becomes
            non-zero for predictions assembled some other way. For the rate before filtering,
            which is the one that says whether a model is declining to predict a transformation,
            read ``Decoded.n_parent_echo / Decoded.n_raw``.

        Scalars are attached to ``df.attrs``: ``criterion``, ``n_rows``, ``n_parents`` and
        ``invalid_rate``.
    """
    if not (len(parents) == len(references) == len(predictions)):
        raise ValueError("parents, references and predictions must be the same length")

    known: dict[str, set] = defaultdict(set)
    for parent, ref in zip(parents, references):
        key = match_key(ref, criterion)
        if key is not None:
            known[parent].add(key)

    ranked: dict[str, list] = {}
    invalid = total = 0
    for parent, preds in zip(parents, predictions):
        if parent in ranked:
            continue
        keys, seen = [], set()
        for p in preds:
            total += 1
            key = match_key(p, criterion)
            if key is None:
                invalid += 1
                continue
            if deduplicate and key in seen:
                continue
            seen.add(key)
            keys.append((key, p))
        ranked[parent] = keys

    rows = []
    for k in k_values:
        top_keys = {parent: {key for key, _ in preds[:k]} for parent, preds in ranked.items()}

        hits, wrong_sims = 0, []
        for parent, ref in zip(parents, references):
            ref_key = match_key(ref, criterion)
            if ref_key is None:
                continue
            if ref_key in top_keys[parent]:
                hits += 1
            else:
                best = max((tanimoto(smiles, ref) for _, smiles in ranked[parent][:k]), default=0.0)
                wrong_sims.append(best)

        recalls, any_hit, precisions = [], [], []
        for parent, truths in known.items():
            found = len(top_keys[parent] & truths)
            recalls.append(found / len(truths))
            any_hit.append(found > 0)
            precisions.append(found / k)

        echoes = [match_key(parent, criterion) in top_keys[parent] for parent in top_keys]

        ceiling = recall_ceiling(parents, references, k, criterion)
        recall_pair = hits / len(references)
        rows.append({
            "k": k,
            "recall_pair": recall_pair,
            "ceiling": ceiling,
            "pct_of_ceiling": recall_pair / ceiling if ceiling else np.nan,
            "recall_parent": float(np.mean(recalls)),
            "parent_hit_rate": float(np.mean(any_hit)),
            "precision_known": float(np.mean(precisions)),
            "tanimoto_wrong": float(np.mean(wrong_sims)) if wrong_sims else np.nan,
            "parent_echo_rate": float(np.mean(echoes)),
        })

    df = pd.DataFrame(rows).set_index("k")
    df.attrs = {
        "criterion": criterion,
        "n_rows": len(references),
        "n_parents": len(known),
        "invalid_rate": invalid / total if total else 0.0,
    }
    return df


def evaluate_all_criteria(parents, references, predictions, **kwargs) -> pd.DataFrame:
    """Score under all three match criteria. The gaps between them are the error analysis."""
    frames = []
    for criterion in CRITERIA:
        df = evaluate(parents, references, predictions, criterion=criterion, **kwargs)
        frames.append(df.assign(criterion=criterion).set_index("criterion", append=True))
    return pd.concat(frames).reorder_levels(["criterion", "k"]).sort_index()


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

def perfect_oracle(parents: list[str], references: list[str]) -> list[list[str]]:
    """Predictions of a model that already knows every answer, for use as a self-test."""
    truth: dict[str, list[str]] = defaultdict(list)
    for parent, ref in zip(parents, references):
        if ref not in truth[parent]:
            truth[parent].append(ref)
    return [truth[p] for p in parents]


def check_ceiling_is_an_upper_bound(
    parents: list[str],
    references: list[str],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> list[str]:
    """Assert no model can beat :func:`recall_ceiling`, by scoring one that knows every answer.

    Asserting an invariant is stronger than asserting a remembered value: a ceiling that counts
    distinct answers is right at k=1 and wrong at k=3 and k=5 under
    any criterion that merges rows. Only a perfect model exposes it.

    Returns:
        A list of human-readable violations. Empty means the invariant holds.
    """
    oracle = perfect_oracle(parents, references)
    violations = []
    for criterion in CRITERIA:
        df = evaluate(parents, references, oracle, k_values=k_values, criterion=criterion)
        for k, row in df.iterrows():
            if row["recall_pair"] > row["ceiling"] + 1e-9:
                violations.append(
                    f"{criterion} k={k}: a perfect model scores {row['recall_pair']:.4f} "
                    f"against a stated ceiling of {row['ceiling']:.4f}"
                )
    return violations


# --------------------------------------------------------------------------------------
# Baselines that need no model
# --------------------------------------------------------------------------------------

def copy_parent(parents: list[str]) -> list[list[str]]:
    """Predict the parent unchanged. Scores 0% exact but 0.634 Tanimoto, which is why the
    Tanimoto column is meaningless unless printed against this floor."""
    return [[p] for p in parents]


def nearest_neighbour(
    parents: list[str],
    train_parents: list[str],
    train_metabolites: list[str],
    k: int = 10,
) -> list[list[str]]:
    """For each test parent, return the metabolites of the most similar training parents.

    This is the memorisation-by-analogy control. It scores 4.2% at k=5 on this test set under
    exact matching (5.5% standardised), so a model well above that is doing more than lookup.
    """
    by_parent: dict[str, list[str]] = defaultdict(list)
    for parent, metabolite in zip(train_parents, train_metabolites):
        if metabolite not in by_parent[parent]:
            by_parent[parent].append(metabolite)

    unique = [p for p in by_parent if _fingerprint(p) is not None]
    train_fps = [_fingerprint(p) for p in unique]

    out = []
    for parent in parents:
        fp = _fingerprint(parent)
        if fp is None:
            out.append([])
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        preds: list[str] = []
        for idx in np.argsort(sims)[::-1]:
            for metabolite in by_parent[unique[idx]]:
                if metabolite not in preds:
                    preds.append(metabolite)
            if len(preds) >= k:
                break
        out.append(preds[:k])
    return out
