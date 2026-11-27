"""Comparação de famílias com pré-processamento ajustado dentro de cada fold."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import criar_preprocessador


RANDOM_STATE = 42


def _pontuar_pipeline(pipeline, X, y) -> dict:
    """Use a mesma regra inclusiva de corte na CV e na avaliação final."""
    indice = list(pipeline.classes_).index(1)
    probabilidades = pipeline.predict_proba(X)[:, indice]
    return metricas_binarias(y, probabilidades, threshold=0.5)


def configuracoes_modelos() -> list[dict]:
    """Retorne quatro valores de complexidade por família, do mais simples ao maior."""
    configuracoes = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        configuracoes.append({
            "id": f"rl_C_{c:g}", "familia": "regressao_logistica",
            "parametro": "C", "valor": c, "rotulo": f"C={c:g}",
        })
    for profundidade in [3, 6, 10, None]:
        rotulo = "sem limite" if profundidade is None else str(profundidade)
        configuracoes.append({
            "id": f"rf_depth_{profundidade}", "familia": "random_forest",
            "parametro": "max_depth", "valor": profundidade,
            "rotulo": f"profundidade={rotulo}",
        })
    return configuracoes


def criar_modelo(configuracao_id: str) -> Pipeline:
    """Monte um pipeline novo, com balanceamento no estimador e sem ajuste prévio."""
    configuracoes = {c["id"]: c for c in configuracoes_modelos()}
    if configuracao_id not in configuracoes:
        raise ValueError(f"Configuração desconhecida: {configuracao_id}")
    configuracao = configuracoes[configuracao_id]
    logistica = configuracao["familia"] == "regressao_logistica"
    if logistica:
        modelo = LogisticRegression(
            C=configuracao["valor"], class_weight="balanced",
            max_iter=2000, random_state=RANDOM_STATE,
        )
    else:
        modelo = RandomForestClassifier(
            n_estimators=100, max_depth=configuracao["valor"],
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        )
    return Pipeline([
        ("preprocessamento", criar_preprocessador(escalonar=logistica)),
        ("modelo", modelo),
    ])


def comparar_validacao(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> pd.DataFrame:
    """Compare oito configurações apenas com o treino fornecido.

    A mesma divisão estratificada é usada nas oito configurações. Cada execução
    clona e reajusta o pipeline completo, incluindo as estatísticas e o encoding.
    O desvio reportado é o desvio padrão entre folds, não intervalo de confiança.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metricas = ["roc_auc", "f1", "recall"]
    linhas = []
    for configuracao in configuracoes_modelos():
        print(f"Validação: {configuracao['id']}", flush=True)
        scores = cross_validate(
            criar_modelo(configuracao["id"]), X, y, cv=cv,
            scoring=_pontuar_pipeline, return_train_score=True, error_score="raise", n_jobs=1,
        )
        linha = {"configuracao": configuracao["id"], "familia": configuracao["familia"],
                 "complexidade": configuracao["rotulo"]}
        for metrica in metricas:
            linha[f"{metrica}_treino_cv"] = float(np.mean(scores[f"train_{metrica}"]))
            linha[f"{metrica}_validacao_cv"] = float(np.mean(scores[f"test_{metrica}"]))
            linha[f"{metrica}_desvio_cv"] = float(np.std(scores[f"test_{metrica}"], ddof=1))
        linha["gap_roc_auc_cv"] = linha["roc_auc_treino_cv"] - linha["roc_auc_validacao_cv"]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def selecionar_finalistas(resultados_cv: pd.DataFrame) -> pd.DataFrame:
    """Escolha um modelo por família por ROC-AUC CV, com F1 CV como desempate.

    Em empate nas duas métricas, use a ordem fixa das configurações (mais simples
    primeiro). Nenhuma coluna de métricas de teste participa da escolha.
    """
    ordem = {c["id"]: i for i, c in enumerate(configuracoes_modelos())}
    tabela = resultados_cv.copy()
    tabela["_ordem"] = tabela["configuracao"].map(ordem)
    tabela = tabela.sort_values(
        ["roc_auc_validacao_cv", "f1_validacao_cv", "_ordem"],
        ascending=[False, False, True],
    )
    return tabela.groupby("familia", sort=False).head(1).drop(columns="_ordem").reset_index(drop=True)


def metricas_binarias(y, probabilidades, threshold: float = 0.5) -> dict:
    """Classe positiva 1 quando probabilidade >= threshold (inicialmente 0,5)."""
    classes = (np.asarray(probabilidades) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probabilidades)),
        "f1": float(f1_score(y, classes, zero_division=0)),
        "recall": float(recall_score(y, classes, zero_division=0)),
    }


def avaliar_treino_teste(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Produza a comparação exigida no roteiro após congelar a seleção por CV.

    Ajuste cada pipeline no treino completo e apenas transforme o teste. Não
    utilize os resultados retornados para refazer a busca de hiperparâmetros.
    """
    linhas = []
    for configuracao in configuracoes_modelos():
        print(f"Avaliação congelada: {configuracao['id']}", flush=True)
        pipeline = criar_modelo(configuracao["id"])
        pipeline.fit(X_train, y_train)
        classe_positiva = list(pipeline.classes_).index(1)
        probabilidades_treino = pipeline.predict_proba(X_train)[:, classe_positiva]
        probabilidades_teste = pipeline.predict_proba(X_test)[:, classe_positiva]
        linha = {"configuracao": configuracao["id"], "familia": configuracao["familia"],
                 "complexidade": configuracao["rotulo"], "threshold": 0.5}
        for conjunto, alvo, probabilidades in [
            ("treino", y_train, probabilidades_treino),
            ("teste", y_test, probabilidades_teste),
        ]:
            for metrica, valor in metricas_binarias(alvo, probabilidades).items():
                linha[f"{metrica}_{conjunto}"] = valor
        linha["gap_roc_auc"] = linha["roc_auc_treino"] - linha["roc_auc_teste"]
        linhas.append(linha)
    return pd.DataFrame(linhas)
