import re
import pandas as pd

def _norm_name(name: str) -> str:
    # normaliza nomes removendo diferenças irrelevantes
    # 1) tira espaços nas pontas
    # 2) baixa tudo para lower
    # 3) converte qualquer coisa não alfanumérica em ponto
    s = name.strip().lower()
    s = re.sub(r'[^0-9a-z]+', '.', s)
    s = re.sub(r'\.+', '.', s).strip('.')
    return s

def align_columns_to_reference(df_target: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Tenta renomear colunas de df_target para os nomes de df_ref
    com base em uma normalização tolerante.
    Apenas renomeia correspondências 1-para-1 não ambíguas.
    """
    norm_ref = {}
    for c in df_ref.columns:
        norm_ref[_norm_name(c)] = c

    mapping = {}
    for c in df_target.columns:
        nc = _norm_name(c)
        if nc in norm_ref:
            mapping[c] = norm_ref[nc]

    # aplica as renomeações encontradas
    df_renamed = df_target.rename(columns=mapping)

    return df_renamed

def common_columns(df_a: pd.DataFrame, df_b: pd.DataFrame):
    return [c for c in df_a.columns if c in df_b.columns]