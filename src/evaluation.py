"""Custos simulados e seleção de limiar com previsões de validação no treino."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.modeling import RANDOM_STATE, criar_modelo


def gerar_probabilidades_oof(configuracao_id, X, y, n_splits=3) -> np.ndarray:
    """Preveja cada linha com um pipeline ajustado sem aquela linha.

    Receba apenas o treino. O alvo deve usar as classes 0 e 1, nessa ordem nas
    colunas de predict_proba. Reutilize os hiperparâmetros congelados na fase 5.
    """
    if set(np.asarray(y)) != {0, 1}:
        raise ValueError("O alvo deve conter as classes 0 e 1.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probabilidades = cross_val_predict(
        criar_modelo(configuracao_id), X, y, cv=cv,
        method="predict_proba", n_jobs=1,
    )
    return probabilidades[:, 1]


def tabela_custos(y, probabilidades, custo_fp=1.0, custo_fn=5.0, thresholds=None) -> pd.DataFrame:
    """Avalie uma grade fixa: custo = FP*custo_fp + FN*custo_fn.

    A classe 1 é atribuída quando probabilidade >= threshold. Custos de acertos
    são zero por hipótese. Os custos são constantes por erro, não por empréstimo.
    """
    alvo = np.asarray(y)
    probabilidades = np.asarray(probabilidades, dtype=float)
    thresholds = np.linspace(0, 1, 201) if thresholds is None else np.asarray(thresholds, dtype=float)
    if alvo.ndim != 1 or probabilidades.ndim != 1 or len(alvo) == 0 or len(alvo) != len(probabilidades):
        raise ValueError("Alvo e probabilidades devem ser vetores não vazios do mesmo tamanho.")
    if not np.isin(alvo, [0, 1]).all():
        raise ValueError("O alvo deve conter apenas 0 e 1.")
    if not np.isfinite(probabilidades).all() or not ((probabilidades >= 0) & (probabilidades <= 1)).all():
        raise ValueError("Probabilidades devem ser finitas e estar entre 0 e 1.")
    if thresholds.ndim != 1 or thresholds.size == 0 or not np.isfinite(thresholds).all() or not ((thresholds >= 0) & (thresholds <= 1)).all():
        raise ValueError("Thresholds devem ser um vetor não vazio com valores entre 0 e 1.")
    if not np.isfinite([custo_fp, custo_fn]).all() or custo_fp <= 0 or custo_fn <= 0:
        raise ValueError("Os custos de FP e FN devem ser positivos e finitos.")

    predicoes = probabilidades[:, None] >= thresholds[None, :]
    positivos = (alvo == 1)[:, None]
    negativos = ~positivos
    tp = (predicoes & positivos).sum(axis=0)
    fp = (predicoes & negativos).sum(axis=0)
    fn = (~predicoes & positivos).sum(axis=0)
    tn = (~predicoes & negativos).sum(axis=0)
    custo = fp * custo_fp + fn * custo_fn
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    precisao = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    return pd.DataFrame({
        "threshold": thresholds, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "custo_total": custo, "custo_medio": custo / len(alvo),
        "recall": recall, "precision": precisao, "f1": f1,
    })


def selecionar_threshold(tabela: pd.DataFrame) -> dict:
    """Minimize custo, depois FN e, em novo empate, o menor ponto de corte."""
    melhor = tabela.sort_values(["custo_total", "fn", "threshold"], kind="stable").iloc[0]
    resultado = melhor.to_dict()
    for coluna in ["tn", "fp", "fn", "tp"]:
        resultado[coluna] = int(resultado[coluna])
    return resultado
