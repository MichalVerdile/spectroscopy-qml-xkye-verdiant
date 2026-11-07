"""Versioned 37-label Jung SMARTS vocabulary, in historical repository order.

Kept independent of model loaders and all signal preprocessing.
"""

from functools import lru_cache

SMARTS = {
    "Acid anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "Acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Alcohol": "[#6][OX2H]",
    "Aldehyde": "[CX3H1](=O)[#6,H]",
    "Alkane": "[CX4;H3,H2]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Arene": "[cX3]1[cX3][cX3][cX3][cX3][cX3]1",
    "Azo compound": "[#6][NX2]=[NX2][#6]",
    "Carbamate": "[NX3][CX3](=[OX1])[OX2H0]",
    "Carboxylic acid": "[CX3](=O)[OX2H]",
    "Enamine": "[NX3][CX3]=[CX3]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Haloalkane": "[#6][F,Cl,Br,I]",
    "Hydrazine": "[NX3][NX3]",
    "Hydrazone": "[NX3][NX2]=[#6]",
    "Imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Isocyanate": "[NX2]=[C]=[O]",
    "Isothiocyanate": "[NX2]=[C]=[S]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Phenol": "[OX2H][cX3]:[c]",
    "Phosphine": "[PX3]",
    "Sulfide": "[#16X2H0]",
    "Sulfonamide": "[#16X4]([NX3])(=[OX1])(=[OX1])[#6]",
    "Sulfonate": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]",
    "Sulfone": "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "Sulfonic acid": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H]",
    "Sulfoxide": "[#16X3]=[OX1]",
    "Thial": "[CX3H1](=S)[#6,H]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    "Thiol": "[#16X2H]",
}
LABEL_NAMES = tuple(SMARTS)


@lru_cache(maxsize=1)
def patterns():
    from rdkit import Chem

    return tuple(Chem.MolFromSmarts(s) for s in SMARTS.values())


def molecule_metadata(smiles):
    """Canonical identity, conservative stereo-collapsed group, scaffold, labels."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("Missing SMILES")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError("Invalid SMILES")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    group = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    # Empty Murcko scaffolds would put all acyclic molecules in one giant group.
    # Generic acyclic connectivity groups retain topology, remove atom identities.
    if not scaffold:
        generic = MurckoScaffold.MakeScaffoldGeneric(mol)
        scaffold = "acyclic:" + Chem.MolToSmiles(generic, isomericSmiles=False)
    labels = np.asarray([mol.HasSubstructMatch(p) for p in patterns()], dtype=np.uint8)
    return canonical, group, scaffold, labels
