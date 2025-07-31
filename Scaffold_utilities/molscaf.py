# -*- coding: utf-8 -*-
"""
Scaffold utilities for cheminformatics pipelines (RDKit + pandas).

- get_mol_from_smiles: robust SMILES → Mol
- get_scaffold_smiles: Murcko scaffold (SMILES)
- get_scaffold_mol: Murcko scaffold (Mol)
- annotate_scaffolds: agrega columnas mol / scaffold / is_acyclic
- scaffold_frequency: tabla de frecuencia y % por scaffold

Requisitos:
    - RDKit
    - pandas
    - tqdm (opcional para progress_apply)

Autor: Tu nombre
Licencia: MIT
"""

from __future__ import annotations
from typing import Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm

# Habilita barras de progreso en pandas.apply
tqdm.pandas(desc="Processing")


# ---------------------------------------------------------------------
# Utilidades atómicas
# ---------------------------------------------------------------------

def get_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Convierte un SMILES a RDKit Mol de forma robusta.

    Args:
        smiles: cadena SMILES (puede venir vacía o NaN).

    Returns:
        Mol válido o None si falla.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        return mol if mol is not None else None
    except Exception:
        return None


def get_scaffold_smiles(smiles: str, include_chirality: bool = True) -> Optional[str]:
    """
    Calcula el Murcko scaffold en formato SMILES a partir de un SMILES.

    Args:
        smiles: cadena SMILES de entrada.
        include_chirality: si se deben incluir centros quirales en el scaffold.

    Returns:
        SMILES del scaffold o None si falla o si no hay anillo (acyclic).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
        # RDKit devuelve "" si es acíclica; estandarizamos a None
        return scaf if scaf else None
    except Exception:
        return None



def get_scaffold_mol(smiles: str, include_chirality: bool = True) -> Optional[Chem.Mol]:
    """
    Calcula el Murcko scaffold y lo devuelve como Mol.

    Args:
        smiles: cadena SMILES de entrada.
        include_chirality: si se deben incluir centros quirales en el scaffold.

    Returns:
        Mol del scaffold o None si falla o si es acíclica.
    """
    scaf_smiles = get_scaffold_smiles(smiles, include_chirality=include_chirality)
    if not scaf_smiles:
        return None
    try:
        return Chem.MolFromSmiles(scaf_smiles)
    except Exception:
        return None

def is_acyclic_mol(mol: Optional[Chem.Mol]) -> bool:
    """
    Determina si una molécula es acíclica en base al número de anillos.

    Args:
        mol: RDKit Mol o None.

    Returns:
        True si no tiene anillos o si es None (tratamos None como acíclica).
    """
    if mol is None:
        return True
    try:
        return rdMolDescriptors.CalcNumRings(mol) == 0
    except Exception:
        return True

# ---------------------------------------------------------------------
# Pipeline sobre DataFrames
# ---------------------------------------------------------------------
def annotate_scaffolds(
    df: pd.DataFrame,
    smiles_col: str = "Canonical_SMILES",
    include_chirality: bool = True,
    name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Anota un DataFrame con columnas:
        - mol: Mol RDKit de la molécula
        - scaffold_smiles: SMILES del Murcko scaffold (None si acíclica)
        - scaffold_mol: Mol del scaffold (None si acíclica)
        - is_acyclic: bool indicando si la molécula de entrada es acíclica

    Args:
        df: DataFrame de entrada con una columna de SMILES.
        smiles_col: nombre de la columna con SMILES.
        include_chirality: si se incluye quiralidad en el scaffold.
        name: nombre opcional para logging.

    Returns:
        Copia del DataFrame con columnas nuevas.
    """
    df = df.copy()
    df_name = name or getattr(df, "name", "DataFrame")

    if smiles_col not in df.columns:
        raise ValueError(f"La columna '{smiles_col}' no está en el DataFrame.")

    print(f"▶ Anotando scaffolds: {df_name}")

    # Mol original
    df["mol"] = df[smiles_col].progress_apply(get_mol_from_smiles)

    # Scaffold (SMILES y Mol) — si no hay anillo, devolvemos None
    df["scaffold_smiles"] = df[smiles_col].progress_apply(
        lambda s: get_scaffold_smiles(s, include_chirality=include_chirality)
    )
    df["scaffold_mol"] = df["scaffold_smiles"].progress_apply(
        lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    )

    # Acíclica si scaffold es None o si mol no tiene anillos
    df["is_acyclic"] = df["mol"].progress_apply(is_acyclic_mol)

    n_total = len(df)
    n_acyclic = int(df["is_acyclic"].sum())
    print(f"  - Moléculas totales: {n_total}")
    print(f"  - Moléculas acíclicas: {n_acyclic} ({n_acyclic / max(n_total,1):.1%})\n")

    return df


def scaffold_frequency(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold_smiles",
    dropna: bool = True,
    top: Optional[int] = None,
) -> pd.DataFrame:
    """
    Agrupa por scaffold y devuelve una tabla de frecuencia y porcentaje.

    Args:
        df: DataFrame anotado (con 'scaffold_smiles').
        scaffold_col: columna con SMILES del scaffold.
        dropna: si True, excluye entradas sin scaffold (acíclicas) del ranking.
        top: si no None, devuelve solo los 'top' scaffolds más frecuentes.

    Returns:
        DataFrame con columnas: ['scaffold_smiles', 'amount', 'frequency']
    """
    if scaffold_col not in df.columns:
        raise ValueError(f"La columna '{scaffold_col}' no está en el DataFrame.")

    work = df.copy()
    if dropna:
        work = work[work[scaffold_col].notna()]

    counts = work.groupby(scaffold_col).size().sort_values(ascending=False)
    total = int(counts.sum())

    out = (
        counts.rename("amount")
        .reset_index()
        .assign(frequency=lambda x: (x["amount"] / max(total, 1) * 100).round(2))
    )

    if top is not None and top > 0:
        out = out.head(top)

    # Reporte breve
    print("▶ Resumen scaffolds")
    print(f"  - Unique scaffolds: {len(counts)}")
    print(f"  - Total instancias (con scaffold): {total}")
    print(f"  - Suma % frecuencia: {out['frequency'].sum():.2f}%")
    print("-" * 50)

    return out
