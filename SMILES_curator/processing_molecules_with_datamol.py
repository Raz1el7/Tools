from __future__ import annotations
import logging
from typing import Optional, Iterable, Tuple
import pandas as pd
import datamol as dm

logger = logging.getLogger(__name__)

def _preprocess_smiles(smiles: str) -> Optional[str]:
    """
    Toma un SMILES, intenta estandarizarlo y devuelve un Canonical SMILES
    (incluyendo estereoquímica). Devuelve None si no se puede curar.
    """
    if smiles is None or pd.isna(smiles) or str(smiles).strip() == "":
        return None

    try:
        mol = dm.to_mol(str(smiles), ordered=True)
        if mol is None:
            return None

        # Correcciones y sanitización
        mol = dm.fix_mol(mol, largest_only=True) #Ajusta la estructura de la molécula para corregir errores comunes
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False) #Sanitiza la molécula asegurándose de que sea químicamente válida

        # Estandarización
        mol = dm.standardize_mol(
            mol,
            disconnect_metals=False,
            normalize=True,
            reionize=True,
            uncharge=False,
            stereo=True,  # re-asigna estereo si falta
        )

        cano = dm.standardize_smiles(dm.to_smiles(mol, isomeric=True))
        return cano
    except Exception as e:
        logger.debug("Error curando SMILES %s: %s", smiles, e)
        return None

def curate_dataframe(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    n_jobs: Optional[int] = None,
    show_progress: bool = True,
    drop_errors: bool = True,
) -> pd.DataFrame:
    """
    Curación paralelizada de una tabla con una columna de SMILES.
    Agrega 'Canonical_SMILES' y 'curation_error' (bool).

    - n_jobs: número de procesos/hilos para dm.parallelized (None = auto).
    - drop_errors: si True, filtra filas con error.

    Returns
    -------
    pd.DataFrame
    """
    if smiles_column not in df.columns:
        raise ValueError(
            f"La columna '{smiles_column}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Preparamos el iterable para parallelized: (index, row)
    iterable: Iterable[Tuple[int, pd.Series]] = df.iterrows()

    def _wrapper(i: int, row: pd.Series) -> pd.Series:
        cano = _preprocess_smiles(row[smiles_column])
        row = row.copy()
        row["Canonical_SMILES"] = cano if cano is not None else "error"
        row["curation_error"] = cano is None
        return row

    logger.info("Curando %d compuestos…", len(df))
    curated_rows = dm.parallelized(
        _wrapper,
        iterable,
        arg_type="args",
        progress=show_progress,
        total=len(df),
        n_jobs=n_jobs,
        # backend="loky",  # puedes habilitarlo si lo prefieres
    )

    df_curated = pd.DataFrame(curated_rows)

    # Métricas rápidas
    n_total = len(df_curated)
    n_errors = int(df_curated["curation_error"].sum())
    n_unique = df_curated.loc[~df_curated["curation_error"], "Canonical_SMILES"].nunique()

    logger.info("Total original: %d", n_total)
    logger.info("Curados únicos (sin error): %d", n_unique)
    logger.info("Errores de curación: %d", n_errors)

    if drop_errors:
        df_curated = df_curated.loc[~df_curated["curation_error"]].reset_index(drop=True)

    return df_curated
