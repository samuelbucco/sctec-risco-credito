"""Carregamento do artefato e inferência sem ajuste do pipeline."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import PrevisaoCredito, SolicitacaoCredito
from src.preprocessing import COLUNAS_ENTRADA


CAMINHO_MODELO = Path(__file__).resolve().parents[1] / "models/modelo_credito.joblib"


@dataclass(frozen=True)
class PreditorCredito:
    pipeline: Any
    threshold: float
    configuracao: str
    indice_positivo: int

    def prever(self, solicitacao: SolicitacaoCredito) -> PrevisaoCredito:
        dados = pd.DataFrame([solicitacao.model_dump()]).loc[:, COLUNAS_ENTRADA]
        probabilidade = float(self.pipeline.predict_proba(dados)[0, self.indice_positivo])
        if not np.isfinite(probabilidade) or not 0 <= probabilidade <= 1:
            raise RuntimeError("O modelo retornou um score inválido.")
        classe = int(probabilidade >= self.threshold)
        return PrevisaoCredito(
            probabilidade_inadimplencia=probabilidade,
            classe_prevista=classe,
            classificacao_risco="elevado" if classe else "baixo",
            acao_recomendada=("Encaminhar para revisão de crédito." if classe
                              else "Prosseguir com a análise de crédito."),
            threshold=self.threshold,
            modelo=self.configuracao,
        )


def carregar_preditor(caminho_modelo: Path = CAMINHO_MODELO) -> PreditorCredito:
    caminho_modelo = Path(caminho_modelo)
    caminho_metadata = caminho_modelo.with_name("metadata_credito.json")
    if not caminho_modelo.is_file() or not caminho_metadata.is_file():
        raise RuntimeError("Modelo ou metadados ausentes. Execute o notebook 06 antes de iniciar a API.")
    metadata = json.loads(caminho_metadata.read_text(encoding="utf-8"))
    sha256 = hashlib.sha256(caminho_modelo.read_bytes()).hexdigest()
    if sha256 != metadata["sha256_artefato"]:
        raise RuntimeError("O SHA-256 do modelo difere dos metadados. Gere novamente o artefato pelo notebook 06.")
    artefato = joblib.load(caminho_modelo)
    threshold = float(artefato["threshold"])
    classes = list(artefato["pipeline"].classes_)
    if (artefato.get("schema_version") != 1 or classes != [0, 1]
            or not np.isfinite(threshold) or not 0 <= threshold <= 1
            or threshold != metadata["threshold"]
            or artefato["configuracao"] != metadata["configuracao"]
            or artefato["campos_entrada"] != COLUNAS_ENTRADA
            or metadata["campos_entrada"] != COLUNAS_ENTRADA):
        raise RuntimeError("Modelo e metadados incompatíveis com o contrato da API.")
    return PreditorCredito(artefato["pipeline"], threshold, artefato["configuracao"], classes.index(1))
