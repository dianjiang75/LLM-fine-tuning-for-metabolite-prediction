"""How raw model samples become the k predictions that are scored.

Every arm that generates uses this, so differences between arms measure the model. The policy
is fixed and untuned.

Four steps:

**Drop what cannot be parsed.** The provided harness silently discards unparseable predictions
before taking the top k, so an invalid answer costs a model nothing there. We count them
and report the rate.

**Drop predictions identical to the parent.** A model that returns its input has predicted no
metabolism. Measured here, roughly one reply in ten from an untuned model is the unchanged parent.
Similarity scores cannot detect this: a wrong-site isomer and the unchanged parent sit about 0.02
apart, so it is filtered by identity.

**Collapse duplicates.** Sampling a peaked model k times usually returns the same string several
times. Beams of 5, 10 and 44 were measured to contain only 42%, 28% and 13% unique molecules, so
leaving duplicates in means an effective k far below the nominal one.

**Rank by how often the model produces it.** The same molecule written different ways is the
same question asked different ways, so a candidate several spellings agree on is more likely right
than one that appears under a single spelling. Each occurrence contributes 1, and the counts add
across spellings and temperatures.

Occurrences are counted without position weights, because the API returns independent
draws, so position within a batch is an artefact of sampling. Counting is the
maximum-likelihood estimate of which molecule the model holds most probable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .evaluation import canonical, match_key

ANSWER = re.compile(r"<metabolite>\s*(.*?)\s*</metabolite>", re.S)


def extract(text: str) -> str | None:
    """Pull the SMILES out of a model's reply.

    Three shapes are handled. A prompt that ends with ``<metabolite>`` leaves the model to supply
    only the closing tag, so a pattern requiring both tags would fail on every reply.

    - fully tagged, ``<metabolite>CCO</metabolite>``
    - closing tag only, ``CCO</metabolite>``, when the prompt opened it
    - untagged, ``CCO``, which untuned models often produce

    Returns None when nothing usable is there, which the caller counts as unparseable.
    """
    if not text:
        return None
    match = ANSWER.search(text)
    if match:
        candidate = match.group(1)
    else:
        candidate = text.split("</metabolite>")[0].strip().split("\n")[0]
    candidate = candidate.strip().strip("`").strip()
    return candidate or None


@dataclass
class Decoded:
    """The k predictions for one parent, plus what was thrown away getting there."""

    predictions: list[str]
    n_raw: int = 0
    n_unparseable: int = 0
    n_parent_echo: int = 0
    n_distinct: int = 0
    votes: dict[str, float] = field(default_factory=dict)

    @property
    def invalid_rate(self) -> float:
        return self.n_unparseable / self.n_raw if self.n_raw else 0.0


def collapse(
    sample_lists: list[list[str]],
    parent: str,
    k: int = 5,
    criterion: str = "standardized",
) -> Decoded:
    """Turn raw samples into at most k distinct molecules, best first.

    Args:
        sample_lists: one list per prompt spelling, each holding that spelling's raw model
            replies in the order they were returned. Passing a single list is fine; test-time
            augmentation passes several.
        parent: the input molecule, so predictions identical to it can be dropped.
        k: how many predictions to report.
        criterion: which notion of "the same molecule" to collapse on. Defaults to the
            standardised form, matching how the harness decides whether an answer is correct.

    Returns:
        A :class:`Decoded` carrying the predictions and the counts behind them.
    """
    parent_key = match_key(parent, criterion)

    votes: dict[str, float] = defaultdict(float)
    best_text: dict[str, str] = {}
    n_raw = n_unparseable = n_parent_echo = 0

    for samples in sample_lists:
        for raw in samples:
            n_raw += 1
            text = extract(raw)
            if text is None:
                n_unparseable += 1
                continue
            key = match_key(text, criterion)
            if key is None:
                n_unparseable += 1
                continue
            if key == parent_key:
                n_parent_echo += 1
                continue
            votes[key] += 1.0
            # keep the earliest spelling seen for this molecule, written in canonical form
            best_text.setdefault(key, canonical(text) or text)

    # Break equal vote counts on normalized identity to keep ranking stable across input orders.
    ordered = sorted(votes, key=lambda key: (-votes[key], key))
    return Decoded(
        predictions=[best_text[key] for key in ordered[:k]],
        n_raw=n_raw,
        n_unparseable=n_unparseable,
        n_parent_echo=n_parent_echo,
        n_distinct=len(ordered),
        votes=dict(votes),
    )


def rank_candidates(
    candidates: list[str],
    scores: list[float],
    parent: str,
    k: int = 5,
    criterion: str = "standardized",
) -> Decoded:
    """Order a fixed candidate list by score, applying the same filters as :func:`collapse`.

    Used by the ranking arms, where the candidates come from rule enumeration.
    Keeping the filtering identical is what makes a ranking arm comparable to a generating one.

    Args:
        candidates: the molecules to rank.
        scores: one score per candidate, higher is better. Length-normalised log probability,
            for the arms that rank by model likelihood.
    """
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must be the same length")

    parent_key = match_key(parent, criterion)

    best: dict[str, tuple[float, str]] = {}
    n_unparseable = n_parent_echo = 0
    for smiles, score in zip(candidates, scores):
        key = match_key(smiles, criterion)
        if key is None:
            n_unparseable += 1
            continue
        if key == parent_key:
            n_parent_echo += 1
            continue
        if key not in best or score > best[key][0]:
            best[key] = (score, canonical(smiles) or smiles)

    # Cached scores land on 1/8 steps. Break ties on normalized identity for stable ranking.
    ordered = sorted(best, key=lambda key: (-best[key][0], key))
    return Decoded(
        predictions=[best[key][1] for key in ordered[:k]],
        n_raw=len(candidates),
        n_unparseable=n_unparseable,
        n_parent_echo=n_parent_echo,
        n_distinct=len(ordered),
        votes={key: best[key][0] for key in ordered},
    )
