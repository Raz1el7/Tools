import matplotlib.pyplot as plt
import numpy as np
from math import pi
from rdkit import Chem
from rdkit.Chem import Descriptors Draw
from rdkit.Chem import Crippen, rdMolDescriptors, Lipinski


def normalize_df_to_radar_plot(df: pd.DataFrame, smiles_col='SMILES') -> pd.DataFrame:
    """
    Normaliza propiedades para radar plot basadas en la regla de los 5 (Ro5).
    Calcula propiedades faltantes usando SMILES si es necesario.

    Parámetros:
        df : pd.DataFrame - DataFrame que contiene propiedades o SMILES.
        smiles_col : str - Nombre de la columna con los SMILES.

    Retorna:
        pd.DataFrame - DataFrame normalizado con columnas: MW, logP, HBA, HBD, RotB, TPSA
    """

    # Define nombres posibles, valores de normalización y función de rdkit para su cálculo en caso de ser
    # necesario
    columns_info = {
        'MW':     (['MW', 'molecular_weight'],            500,  Descriptors.MolWt),
        'logP':   (['logP', 'Consensus Log P'],            5,   Crippen.MolLogP),
        'HBA':    (['HBA', '#H-bond acceptors', 'hydrogen_bond_acceptors'], 10, Lipinski.NumHAcceptors),
        'HBD':    (['HBD', '#H-bond donors', 'hydrogen_bond_donors'],       5,  Lipinski.NumHDonors),
        'RotB':   (['RB', '#Rotatable bonds'],            10,   Lipinski.NumRotatableBonds),
        'TPSA':   (['TPSA', 'tpsa'],                     140,   rdMolDescriptors.CalcTPSA),
    }
 
    norm_data = {key: [] for key in columns_info}
    failed_idx = []

    for idx, row in df.iterrows():
        mol = None
        if smiles_col in df.columns:
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None

        for key, (aliases, divisor, descriptor_func) in columns_info.items():
            value = None
            for alt in aliases:
                if alt in df.columns:
                    value = row[alt]
                    break
            if value is None and mol is not None:
                try:
                    value = descriptor_func(mol)
                except Exception:
                    value = np.nan
            norm_data[key].append(value / divisor if value is not None else np.nan)

        if mol is None:
            failed_idx.append(idx)

    if failed_idx:
        print(f"⚠️ Molecules at indices {failed_idx} could not be parsed from SMILES.")

    return pd.DataFrame(norm_data)

def make_radar_plot(data, title="Radar Plot", filename="radar_plot.jpg", max_profiles=250):
    categories = list(data.columns)
    N = len(categories)

    # Ángulos para el radar
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Límites superior e inferior (ajustados al número de categorías)
    Ro5_up = [1] * N + [1]
    #Ro5_low = [0.5, 0.1, 0, 0.25,  0.5] + [0.5] 
    Ro5_low = [0.5, 0.1, 0, 0.25, 0.1, 0.5] + [0.5]  # Debe coincidir con N+1


    # Validación de dimensiones
    if len(Ro5_low) != N + 1:
        raise ValueError(f"Ro5_low must have {N+1} elements. Found {len(Ro5_low)}.")

    #Figura
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    ax.set(alpha=0.3)


    # Estética
    plt.xticks(angles[:-1], categories, color='k', size=14, ha='center', fontweight='medium')
    plt.tick_params(axis='y', width=2, labelsize=12, grid_alpha=0.2, tickdir='out')
    ax.set_rlabel_position(0)
    ax.tick_params(axis='x', pad=10)

    # Plot de límites
    ax.plot(angles, Ro5_up, linewidth=3, linestyle='-', color='indianred', label='Ro5 upper')
    ax.plot(angles, Ro5_low, linewidth=3, linestyle='--', color='indianred', label='Ro5 lower')

    # Perfil
    for i in data.index[:max_profiles]:
        values = data.loc[i].values.tolist()
        values += values[:1]  # cerrar figura
        ax.plot(angles, values, linewidth=2, color='steelblue', alpha=0.25)

    # Título y leyenda
    plt.title(title, fontsize=20, weight='bold', y=1.1)
    ax.grid(axis='y', linewidth=1.2, linestyle='dotted', alpha=0.3, color='gray')
    ax.grid(axis='x', linewidth=1.5, linestyle='-', alpha=0.3, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()
