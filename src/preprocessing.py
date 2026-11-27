"""Regras de qualidade e pré-processamento sem ajuste no conjunto de teste."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import adicionar_comprometimento_renda


COLUNAS_NUMERICAS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "cb_person_cred_hist_length",
]
COLUNAS_CATEGORICAS = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]
COLUNAS_ENTRADA = COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS
COLUNAS_MEDIANA = [c for c in COLUNAS_NUMERICAS if c != "loan_int_rate"]


def tratar_inconsistencias(dados: pd.DataFrame) -> pd.DataFrame:
    """Aplique regras fixas sem aprender estatísticas nem remover registros.

    Os limites de idade (18 a 100 anos) são uma convenção educacional do projeto.
    A coluna redundante loan_percent_income é desconsiderada. loan_status não
    pode estar na entrada. Campos desconhecidos também não entram no modelo.
    """
    if not isinstance(dados, pd.DataFrame):
        raise TypeError("A entrada deve ser um DataFrame com colunas nomeadas.")
    if "loan_status" in dados.columns:
        raise ValueError("Separe loan_status dos atributos antes do pipeline.")
    ausentes = sorted(set(COLUNAS_ENTRADA) - set(dados.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {ausentes}")

    resultado = dados.loc[:, COLUNAS_ENTRADA].copy()
    for coluna in COLUNAS_NUMERICAS:
        resultado[coluna] = pd.to_numeric(resultado[coluna], errors="coerce")
        resultado[coluna] = resultado[coluna].replace([np.inf, -np.inf], np.nan)

    idade_original = resultado["person_age"].copy()
    idade_invalida = ~resultado["person_age"].between(18, 100)
    resultado.loc[idade_invalida, "person_age"] = np.nan

    for coluna in ["person_income", "loan_amnt", "loan_int_rate"]:
        resultado.loc[resultado[coluna] <= 0, coluna] = np.nan

    for coluna in ["person_emp_length", "cb_person_cred_hist_length"]:
        invalido = (resultado[coluna] < 0) | (resultado[coluna] > idade_original)
        resultado.loc[invalido, coluna] = np.nan

    for coluna in COLUNAS_CATEGORICAS:
        valores = resultado[coluna].astype("string").str.strip().replace("", pd.NA)
        # SimpleImputer recebe np.nan também em colunas de texto do pandas 3.
        resultado[coluna] = valores.astype(object).where(valores.notna(), np.nan)
    return resultado


def criar_preprocessador(escalonar: bool = True) -> Pipeline:
    """Crie um pipeline ainda não ajustado, pronto para receber um estimador.

    fit deve receber apenas a partição de treino. Durante validação cruzada,
    inclua este pipeline no mesmo Pipeline que o classificador para que cada
    fold aprenda seus próprios parâmetros de imputação, encoding e escala.
    """
    imputacao = ColumnTransformer(
        transformers=[
            ("medianas", SimpleImputer(strategy="median", keep_empty_features=True), COLUNAS_MEDIANA),
            ("taxa_media", SimpleImputer(strategy="mean", keep_empty_features=True), ["loan_int_rate"]),
            ("categorias", SimpleImputer(strategy="most_frequent", keep_empty_features=True), COLUNAS_CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    codificacao = ColumnTransformer(
        transformers=[
            ("numericas", StandardScaler() if escalonar else "passthrough",
             COLUNAS_NUMERICAS + ["comprometimento_renda"]),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUNAS_CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("regras", FunctionTransformer(tratar_inconsistencias, validate=False)),
        ("imputacao", imputacao),
        ("atributos", FunctionTransformer(adicionar_comprometimento_renda, validate=False)),
        ("codificacao", codificacao),
    ])
