"""SMILES spellings for input augmentation in notebooks 02, 03 and 04.

The canonical spelling comes first. Other spellings use Python's ``random``
module, so ``random.seed(...)`` controls their order and content.
"""

from __future__ import annotations

import random

from rdkit import Chem


def _molecule(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    parser = Chem.SmilesParserParams()
    parser.parseName = False
    return Chem.MolFromSmiles(smiles, parser)


def canonicalize(smiles: str) -> str | None:
    """Return canonical SMILES while retaining specified stereochemistry."""
    mol = _molecule(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def randomize_smiles(smiles: str) -> str | None:
    """Write the molecule with shuffled atom indices, retaining stereochemistry."""
    mol = _molecule(smiles)
    if mol is None:
        return None
    order = list(range(mol.GetNumAtoms()))
    random.shuffle(order)
    return Chem.MolToSmiles(
        Chem.RenumberAtoms(mol, order), canonical=False, isomericSmiles=True
    )


def get_smiles_variants(smiles: str, n: int = 4) -> list[str]:
    """Return up to ``n`` distinct spellings, with the canonical spelling first.

    Symmetric or very small molecules may have fewer distinct spellings. Invalid
    input returns an empty list. The attempt limit bounds work for those cases.
    """
    if n < 1:
        return []
    canonical = canonicalize(smiles)
    if canonical is None:
        return []
    variants = [canonical]
    seen = {canonical}
    for _ in range(max(20, n * 20)):
        if len(variants) >= n:
            break
        variant = randomize_smiles(canonical)
        if variant is not None and variant not in seen:
            variants.append(variant)
            seen.add(variant)
    return variants
