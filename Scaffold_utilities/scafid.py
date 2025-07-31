# -*- coding: utf-8 -*-
"""
ID utilities for molecular scaffolds.

- id_generator: IDs aleatorios (A-Z, 0-9)
- make_scaffold_id: IDs 'stable' (hash) o 'random'
- assign_ids: asigna IDs a un DataFrame con scaffolds
- extrapolate_ids: conserva IDs existentes y genera nuevos para scaffolds nuevos
- generate_composite_label: etiqueta compuesta usando amount / frequency (+ aleatorio opcional)

Requisitos:
    - pandas
    - rdkit (opcional si integras con annotate_scaffolds del módulo anterior)
    - tqdm
"""

from __future__ import annotations
from typing import Optional, Dict, Set

import base64
import hashlib
import random
import string

import pandas as pd
from tqdm import tqdm

# Habilita progreso para pandas
tqdm.pandas(desc="Assigning IDs")


# ---------------------------------------------------------------------
# Generadores de ID
# ---------------------------------------------------------------------

def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    """
    Genera un ID aleatorio (no determinista).

    Args:
        size: longitud del ID.
        chars: alfabeto permitido.

    Returns:
        Cadena aleatoria de longitud 'size'.
    """
    return ''.join(random.choice(chars) for _ in range(size))


def _hash_b32(text: str, length: int = 8) -> str:
    """
    Hash estable (SHA1 → Base32) truncado a 'length'.

    Args:
        text: texto a hashear (e.g., SMILES del scaffold).
        length: longitud del ID final.

    Returns:
        ID alfanumérico en mayúsculas, reproducible.
    """
    h = hashlib.sha1(text.encode('utf-8')).digest()
    b32 = base64.b32encode(h).decode('ascii').rstrip('=')
    return b32[:length]


def make_scaffold_id(scaffold_smiles: Optional[str],
                     strategy: str = "stable",
                     length: int = 8,
                     rnd_chars: str = string.ascii_uppercase + string.digits) -> str:
    """
    Crea un ID para un scaffold.

    Estrategias:
      - 'stable': hash del SMILES → reproducible
      - 'random': aleatorio (A-Z0-9)

    Args:
        scaffold_smiles: SMILES del scaffold (o None).
        strategy: 'stable' | 'random'.
        length: longitud del ID a generar.
        rnd_chars: alfabeto para IDs aleatorios.

    Returns:
        ID de scaffold (string).
    """
    if strategy not in {"stable", "random"}:
        raise ValueError("strategy debe ser 'stable' o 'random'.")

    # Para scaffolds vacíos/None, generamos un marcador reproducible o aleatorio
    key = scaffold_smiles.strip() if isinstance(scaffold_smiles, str) else "NA_SCAFFOLD"

    if strategy == "stable":
        return _hash_b32(key, length=length)
    else:
        return id_generator(size=length, chars=rnd_chars)


# ---------------------------------------------------------------------
# Asignación y extrapolación de IDs
# ---------------------------------------------------------------------

def assign_ids(df: pd.DataFrame,
               scaffold_col: str = "scaffold_smiles",
               id_col: str = "scaffold_id",
               strategy: str = "stable",
               length: int = 8) -> pd.DataFrame:
    """
    Asigna IDs a cada fila, basados en el valor de 'scaffold_col'.

    Si strategy='stable', el mismo scaffold SMILES producirá el mismo ID
    en cualquier DataFrame (recomendado para trazabilidad).

    Args:
        df: DataFrame con columna de scaffold.
        scaffold_col: nombre de la columna con SMILES del scaffold (puede ser None/NaN).
        id_col: nombre de la columna de salida con IDs.
        strategy: 'stable' | 'random'.
        length: longitud del ID.

    Returns:
        Copia de df con columna 'id_col'.
    """
    if scaffold_col not in df.columns:
        raise ValueError(f"La columna '{scaffold_col}' no está en el DataFrame.")

    out = df.copy()
    # Generación inicial
    out[id_col] = out[scaffold_col].progress_apply(
        lambda s: make_scaffold_id(s, strategy=strategy, length=length)
    )

    if strategy == "random":
        # Aseguramos unicidad (evitar raras colisiones aleatorias)
        seen: Set[str] = set()
        def ensure_unique(x: str) -> str:
            nonlocal seen
            if x not in seen:
                seen.add(x)
                return x

            # Colisión: regenerar hasta ser único
            new_id = x
            while new_id in seen:
                new_id = id_generator(size=length)
            seen.add(new_id)
            return new_id

        out[id_col] = out[id_col].progress_apply(ensure_unique)

    return out


def extrapolate_ids(existing_df: pd.DataFrame,
                    new_df: pd.DataFrame,
                    scaffold_col: str = "scaffold_smiles",
                    id_col: str = "scaffold_id",
                    strategy_new: str = "stable",
                    length: int = 8) -> pd.DataFrame:
    """
    Asigna IDs a un nuevo DataFrame preservando los IDs existentes.

    - Si el scaffold ya está en 'existing_df', reusa el ID.
    - Si es nuevo, lo genera con 'strategy_new'.

    Args:
        existing_df: DataFrame ya anotado con [scaffold_col, id_col].
        new_df: DataFrame nuevo que requiere IDs.
        scaffold_col: nombre de columna de scaffold.
        id_col: nombre de columna de ID.
        strategy_new: 'stable' | 'random' para nuevos scaffolds.
        length: longitud del ID nuevo.

    Returns:
        Copia de 'new_df' con columna 'id_col'.
    """
    for c in (scaffold_col, id_col):
        if c not in existing_df.columns:
            raise ValueError(f"'{c}' no está en existing_df.")

    if scaffold_col not in new_df.columns:
        raise ValueError(f"'{scaffold_col}' no está en new_df.")

    out = new_df.copy()

    # Mapeo scaffold -> ID ya existente
    mapping: Dict[Optional[str], str] = dict(zip(existing_df[scaffold_col],
                                                 existing_df[id_col]))
    existing_ids: Set[str] = set(existing_df[id_col].astype(str).tolist())

    def get_or_make_id(scaffold: Optional[str]) -> str:
        # Preserva ID si existe
        if scaffold in mapping:
            return mapping[scaffold]
        # Si no existe, crea uno nuevo (y evita colisiones)
        new_id = make_scaffold_id(scaffold, strategy=strategy_new, length=length)
        if strategy_new == "random":
            while new_id in existing_ids:
                new_id = id_generator(size=length)
        existing_ids.add(new_id)
        return new_id

    out[id_col] = out[scaffold_col].progress_apply(get_or_make_id)
    return out


# ---------------------------------------------------------------------
# Etiquetas compuestas para tablas de frecuencia
# ---------------------------------------------------------------------

def generate_composite_label(row: pd.Series,
                             size: int = 3,
                             chars: str = string.ascii_uppercase) -> str:
    """
    Genera una etiqueta legible para filas con columnas ['amount', 'frequency'].
    Formato: '{amount}.{int(frequency)}.{RND}'  (e.g., '12.7.ABC')

    Args:
        row: fila con 'amount' (int) y 'frequency' (float).
        size: longitud de sufijo aleatorio.
        chars: alfabeto para sufijo aleatorio.

    Returns:
        etiqueta compuesta.
    """
    amount = int(row.get('amount', 0))
    freq_int = int(round(float(row.get('frequency', 0.0))))
    rnd = ''.join(random.choice(chars) for _ in range(size))
    return f"{amount}.{freq_int}.{rnd}"
