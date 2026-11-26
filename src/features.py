"""Engenharia de atributos compartilhada entre treinamento e API."""

import numpy as np
import pandas as pd


def adicionar_comprometimento_renda(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcule a razão percentual entre empréstimo e renda anual.

    No pipeline, esta função recebe os componentes já imputados. Em uso isolado,
    renda não positiva, componentes ausentes e resultados não finitos produzem
    NaN, nunca zero artificial ou infinito. O DataFrame de entrada é preservado.
    """
    resultado = dados.copy()
    renda = pd.to_numeric(resultado["person_income"], errors="coerce")
    emprestimo = pd.to_numeric(resultado["loan_amnt"], errors="coerce")
    validos = (
        np.isfinite(renda) & (renda > 0)
        & np.isfinite(emprestimo) & (emprestimo > 0)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        razao = emprestimo.where(validos).div(renda.where(validos)) * 100
    resultado["comprometimento_renda"] = razao.where(np.isfinite(razao), np.nan)
    return resultado
