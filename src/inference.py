"""Parent SMILES in, ranked metabolite predictions out.

Wraps the configuration that scored highest on the holdout set. Two stages: enumerate every
molecule the SMIRKS rule set can produce from the parent, then score each candidate with the
fine-tuned ranker and order by its preference.

The rules reach 72.2% of the recorded answers. Five prediction slots limit the best possible
recall on those candidate lists to 71.5%; the ranker orders that finite set of candidates.
Pair recall on the holdout set is 18.1% at k=1 and 46.6% at k=5, and at least one correct
metabolite appears in the top 5 for 63.2% of parents.

A candidate's score is logP("yes") - logP("no") under the prompt the ranker was trained on. Both
verdicts are conditioned on the same prompt, so the difference cancels whatever that prompt costs
in absolute probability and leaves only the model's preference between the two answers. It is a
log-odds and is not calibrated, so scores order candidates and nothing more.

Scores come from one of two sources, with the same checkpoint behind both:

- :class:`LiveRanker` calls the checkpoint recorded in ``data/processed/reranker_run.json``. It
  needs ``TINKER_API_KEY`` and a checkpoint that has not expired. The notebooks save checkpoints
  with a seven-day lifetime; this one's was extended to 30 days. :meth:`LiveRanker.expires_at`
  reads the current date from the service, which the interface shows.
- :class:`SavedRanker` replays ``SOM/som_candidate_scores.json``, which that same checkpoint
  produced for the 155 holdout and 60 dev parents. It makes no network call and covers only
  those molecules.

Both reach the k predictions through :func:`src.decoding.rank_candidates`, the same filtering and
collapsing that produced every scored number in the project.
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .decoding import rank_candidates
from .evaluation import match_key

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
RULES_CSV = ROOT / "rules" / "metabolism_rules.csv"
RANKER_RUN = ROOT / "data" / "processed" / "reranker_run.json"
SAVED_SCORES = ROOT / "SOM" / "som_candidate_scores.json"

PROMPT = "<parent>{parent}</parent>\n<candidate>{candidate}</candidate>\n"

# Limits on what one request may ask for. Enumeration is quadratic in matching sites for a few
# rules, and every candidate costs two scoring calls, so both are bounded.
MAX_SMILES_CHARS = 1000
MAX_HEAVY_ATOMS = 100
MAX_CANDIDATES = 500
PER_RULE_CAP = 200


class InvalidInput(ValueError):
    """An input the pipeline will not run on, carrying the reason for display."""


class RankerUnavailable(RuntimeError):
    """Scores could not be obtained, carrying the reason for display."""


def clean_parent(text: str) -> str:
    """Validate one parent SMILES and return it canonicalised.

    Canonicalising first means the same molecule written two ways produces one cache key, one
    enumeration and one set of scores. The rejections below are inputs the rest of the pipeline
    has no defined behaviour for.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidInput("Enter a parent molecule as a SMILES string.")
    text = text.strip()
    if len(text) > MAX_SMILES_CHARS:
        raise InvalidInput(f"SMILES strings are accepted up to {MAX_SMILES_CHARS} characters.")
    if any(character.isspace() for character in text) or "<" in text or ">" in text:
        raise InvalidInput("Enter the SMILES string alone, without prompt tags or a name.")

    parser = Chem.SmilesParserParams()
    parser.parseName = False
    mol = Chem.MolFromSmiles(text, parser)
    if mol is None:
        raise InvalidInput("This string could not be parsed as a molecule.")
    if mol.GetNumHeavyAtoms() == 0:
        raise InvalidInput("Enter a molecule with at least one heavy atom.")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise InvalidInput("Enter a single connected molecule, without counterions or fragments.")
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise InvalidInput(f"Molecules are accepted up to {MAX_HEAVY_ATOMS} heavy atoms.")
    if any(atom.GetAtomicNum() == 0 or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum()
           for atom in mol.GetAtoms()):
        raise InvalidInput("Enter an ordinary SMILES, without wildcard atoms, radicals or atom maps.")
    return Chem.MolToSmiles(mol, canonical=True)


def compile_rules(path: Path = RULES_CSV) -> list[tuple[str, object, bool]]:
    """Compile the rule set to ``(name, reaction, keep_largest_fragment)`` triples."""
    compiled = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            reaction = AllChem.ReactionFromSmarts(row["smirks"])
            reaction.Initialize()
            compiled.append((row["name"], reaction, bool(int(row["keep_largest_fragment"]))))
    return compiled


def _largest_fragment(mol):
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(fragments, key=lambda f: f.GetNumHeavyAtoms()) if len(fragments) > 1 else mol


def enumerate_candidates(parent: str, rules, per_rule_cap: int = PER_RULE_CAP) -> dict[str, str]:
    """Apply every rule at every matching site.

    Returns ``{candidate_smiles: name of the rule that produced it}``, first rule to reach a
    molecule keeping the attribution. Identical to the enumeration behind every recorded
    coverage and recall figure, so a candidate list here matches the one those numbers came from.
    """
    mol = Chem.MolFromSmiles(parent)
    if mol is None:
        return {}
    out: dict[str, str] = {}
    for name, reaction, keep_largest in rules:
        try:
            products = reaction.RunReactants((mol,))
        except Exception:
            continue
        made = 0
        for product_tuple in products:
            for product in product_tuple:
                try:
                    candidate = _largest_fragment(product) if keep_largest else product
                    Chem.SanitizeMol(candidate)
                    smiles = Chem.MolToSmiles(candidate, canonical=True)
                except Exception:
                    smiles = None
                if smiles:
                    out.setdefault(smiles, name)
                made += 1
                if made > per_rule_cap:
                    break
            if made > per_rule_cap:
                break
    return out


@dataclass
class Prediction:
    """One ranked metabolite."""

    rank: int
    smiles: str
    score: float
    rule: str


@dataclass
class Result:
    """What one parent produced."""

    parent: str
    predictions: list[Prediction]
    n_candidates: int
    source: str
    checkpoint: str
    elapsed_seconds: float
    scores: dict[str, float] = field(default_factory=dict)


class LiveRanker:
    """Scores candidates by calling the fine-tuned checkpoint.

    The client is created on first use rather than at construction, so an app can start, show
    the saved results and report a credential or checkpoint problem only when a live score is
    actually asked for.
    """

    source = "live"

    def __init__(self, run_path: Path = RANKER_RUN, api_key: str | None = None):
        self.run = json.loads(Path(run_path).read_text())
        self.checkpoint = self.run["best_checkpoint"]
        self._api_key = api_key
        self._client = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def api_key(self) -> str | None:
        """The key from the environment, falling back to the project's .env file."""
        if self._api_key:
            return self._api_key
        if os.environ.get("TINKER_API_KEY"):
            return os.environ["TINKER_API_KEY"]
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("TINKER_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return None

    def configured(self) -> bool:
        return bool(self.api_key())

    def expires_at(self) -> datetime | None:
        """When the checkpoint stops answering, or None when it has no expiry set.

        Read from the service rather than from the run file, because a checkpoint's lifetime can
        be changed after the run that created it was recorded, which leaves the file's
        ``checkpoint_ttl_seconds`` describing the run rather than the checkpoint's current state.
        """
        key = self.api_key()
        if not key:
            raise RankerUnavailable("Reading the expiry needs TINKER_API_KEY.")
        try:
            import tinker

            rest = tinker.ServiceClient(api_key=key).create_rest_client()
            run_id = self.checkpoint.split("tinker://", 1)[1].split("/", 1)[0]
            listed = rest.list_checkpoints(run_id).result().checkpoints
        except Exception as exc:
            raise RankerUnavailable(
                f"The expiry could not be read ({type(exc).__name__})."
            ) from exc
        for checkpoint in listed:
            if checkpoint.tinker_path == self.checkpoint:
                return checkpoint.expires_at
        raise RankerUnavailable("The checkpoint is no longer listed, so it has expired.")

    def _connect(self):
        """Open the sampling client and check the two conditions scoring depends on.

        The verdict must be a single token, or the score would mix in a length effect, and the
        prompt's last token must not merge with it, or the scored position would not be the
        verdict. Both are asserted at training time and re-checked here because a tokenizer
        change would otherwise shift the score silently.
        """
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            key = self.api_key()
            if not key:
                raise RankerUnavailable(
                    "Live scoring needs TINKER_API_KEY in the environment or in .env."
                )
            try:
                import tinker

                service = tinker.ServiceClient(api_key=key)
                client = service.create_sampling_client(model_path=self.checkpoint)
                tokenizer = client.get_tokenizer()
                yes, no = tokenizer.encode("yes"), tokenizer.encode("no")
                if len(yes) != 1 or len(no) != 1:
                    raise RuntimeError("the verdict is not a single token")
                probe = PROMPT.format(parent="CCO", candidate="CCOC")
                if tokenizer.encode(probe + "yes")[: len(tokenizer.encode(probe))] != tokenizer.encode(probe):
                    raise RuntimeError("the prompt's last token merges with the verdict")
            except RankerUnavailable:
                raise
            except Exception as exc:
                raise RankerUnavailable(
                    "The checkpoint could not be loaded. It may have expired, or the service "
                    f"is unreachable ({type(exc).__name__})."
                ) from exc
            self._client, self._tokenizer = client, tokenizer
            self._yes, self._no = yes, no

    def score(self, parent: str, candidates: list[str], chunk: int = 64,
              timeout: float = 240.0) -> list[float]:
        """One log-odds per candidate, in the order given.

        Requests inside a chunk are submitted before any result is collected, so the chunk's
        candidates are scored concurrently, and the chunk size bounds how many are in flight.
        A 35-candidate molecule takes 20.5s at a chunk of 12 and 7.0s at 40 or above, the
        service being the limit rather than the client, so the default covers most molecules
        in one round.
        """
        self._connect()
        import tinker.types as tt

        # The client takes no timeout of its own, so the bound is applied when each result is
        # collected, which also covers the whole request rather than one call within it.
        deadline = time.monotonic() + timeout
        scores: list[float] = []
        for start in range(0, len(candidates), chunk):
            pairs = []
            for candidate in candidates[start:start + chunk]:
                prefix = self._tokenizer.encode(PROMPT.format(parent=parent, candidate=candidate))
                pairs.append(tuple(
                    self._client.compute_logprobs(prompt=tt.ModelInput.from_ints(prefix + verdict))
                    for verdict in (self._yes, self._no)))
            for yes_future, no_future in pairs:
                values = []
                for future in (yes_future, no_future):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RankerUnavailable("Live scoring did not finish within its time limit.")
                    try:
                        values.append(float(future.result(timeout=remaining)[-1]))
                    except Exception as exc:
                        raise RankerUnavailable(
                            f"Live scoring failed ({type(exc).__name__})."
                        ) from exc
                score = values[0] - values[1]
                if not math.isfinite(score):
                    raise RankerUnavailable("Live scoring returned a non-finite score.")
                scores.append(score)
        return scores


class SavedRanker:
    """Replays the scores the same checkpoint recorded for the holdout and dev parents.

    Covers 215 molecules and nothing else, which is what makes it useful without a key: the
    numbers it returns are the model's own.
    """

    source = "saved"

    def __init__(self, path: Path = SAVED_SCORES, run_path: Path = RANKER_RUN):
        data = json.loads(Path(path).read_text())
        self.scores: dict[str, dict[str, float]] = {}
        for split in ("test", "dev"):
            self.scores.update(data.get(split, {}))
        self.checkpoint = json.loads(Path(run_path).read_text())["best_checkpoint"]

    def parents(self) -> list[str]:
        return sorted(self.scores)

    def covers(self, parent: str) -> bool:
        return parent in self.scores

    def candidates(self, parent: str) -> list[str]:
        if parent not in self.scores:
            raise RankerUnavailable(
                "No saved scores exist for this molecule. Choose a saved molecule, or score it live."
            )
        return list(self.scores[parent])

    def score(self, parent: str, candidates: list[str], **_) -> list[float]:
        saved = self.scores.get(parent)
        if saved is None:
            raise RankerUnavailable(
                "No saved scores exist for this molecule. Choose a saved molecule, or score it live."
            )
        missing = [c for c in candidates if c not in saved]
        if missing:
            raise RankerUnavailable(
                f"{len(missing)} of this molecule's candidates were not in the saved scores."
            )
        return [saved[c] for c in candidates]


class Pipeline:
    """Rule enumeration plus a ranker, holding the compiled rules so they are read once."""

    def __init__(self, rules_csv: Path = RULES_CSV):
        self.rules = compile_rules(rules_csv)
        self.live = LiveRanker()
        self.saved = SavedRanker()

    def ranker(self, source: str):
        if source == "live":
            return self.live
        if source == "saved":
            return self.saved
        raise InvalidInput("Choose live or saved scoring.")

    def predict(self, text: str, source: str = "live", k: int = 5) -> Result:
        """Run one parent end to end.

        In saved mode the candidate list comes from the saved file rather than from enumeration,
        so the result is exactly the one recorded for that molecule. Enumeration is deterministic
        and the rule set has not changed, so the two lists agree, but reading the file cannot
        drift from it.
        """
        started = time.monotonic()
        parent = clean_parent(text)
        ranker = self.ranker(source)

        if source == "saved":
            candidates = ranker.candidates(parent)
            rules = enumerate_candidates(parent, self.rules)
        else:
            rules = enumerate_candidates(parent, self.rules)
            candidates = list(rules)
            if len(candidates) > MAX_CANDIDATES:
                raise InvalidInput(
                    f"This molecule produces {len(candidates)} rule candidates, above the "
                    f"{MAX_CANDIDATES} this interface scores in one request."
                )

        scores = ranker.score(parent, candidates) if candidates else []
        decoded = rank_candidates(candidates, scores, parent, k=k)

        # rank_candidates collapses spellings of one molecule and keeps the best-scoring one, so
        # the score and rule are looked up through the same key it collapsed on.
        by_key: dict[str, tuple[float, str]] = {}
        for candidate, score in zip(candidates, scores):
            key = match_key(candidate, "standardized")
            if key is None:
                continue
            if key not in by_key or score > by_key[key][0]:
                by_key[key] = (score, rules.get(candidate, "unattributed"))

        predictions = []
        for rank, smiles in enumerate(decoded.predictions, 1):
            score, rule = by_key.get(match_key(smiles, "standardized"), (float("nan"), "unattributed"))
            predictions.append(Prediction(rank=rank, smiles=smiles, score=score, rule=rule))

        return Result(
            parent=parent,
            predictions=predictions,
            n_candidates=len(candidates),
            source=source,
            checkpoint=ranker.checkpoint,
            elapsed_seconds=time.monotonic() - started,
            scores=dict(zip(candidates, scores)),
        )
