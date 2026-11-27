"""Contrato HTTP, carregamento do artefato e regra de inferência."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import criar_app
from app.prediction import PreditorCredito, carregar_preditor
from app.schemas import SolicitacaoCredito


RAIZ = Path(__file__).resolve().parents[1]


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(criar_app())
        cls.client.__enter__()
        cls.baixo = json.loads((RAIZ / "docs/exemplos/credito_baixo_risco.json").read_text())
        cls.alto = json.loads((RAIZ / "docs/exemplos/credito_alto_risco.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_e_requisicoes_reutilizam_modelo_sem_fit(self):
        preditor = self.client.app.state.preditor
        with patch("app.main.carregar_preditor") as carregar, patch.object(preditor.pipeline, "fit") as fit:
            resposta = self.client.get("/health")
            self.assertEqual(resposta.status_code, 200)
            self.assertEqual(resposta.json(), {"status": "ok", "modelo": "rf_depth_None", "threshold": .295})
            self.assertEqual(self.client.post("/predict", json=self.baixo).status_code, 200)
            self.assertEqual(self.client.post("/predict", json=self.alto).status_code, 200)
            carregar.assert_not_called()
            fit.assert_not_called()
        self.assertIs(self.client.app.state.preditor, preditor)

    def test_exemplos_cobrem_baixo_e_alto_risco(self):
        for dados, classe in [(self.baixo, 0), (self.alto, 1)]:
            with self.subTest(classe=classe):
                resposta = self.client.post("/predict", json=dados)
                self.assertEqual(resposta.status_code, 200)
                corpo = resposta.json()
                self.assertEqual(corpo["classe_prevista"], classe)
                self.assertEqual(corpo["classe_prevista"], int(corpo["probabilidade_inadimplencia"] >= corpo["threshold"]))
                self.assertEqual(corpo["classificacao_risco"], "elevado" if classe else "baixo")
                self.assertTrue(corpo["acao_recomendada"])

    def test_preditor_aplica_threshold_salvo_inclusive_na_fronteira(self):
        pipeline = Mock()
        preditor = PreditorCredito(pipeline, .295, "exemplo", 1)
        entrada = SolicitacaoCredito.model_validate(self.baixo)
        for probabilidade, classe in [(.294, 0), (.295, 1), (.3, 1)]:
            pipeline.predict_proba.return_value = np.array([[1 - probabilidade, probabilidade]])
            self.assertEqual(preditor.prever(entrada).classe_prevista, classe)
        pipeline.predict.assert_not_called()
        pipeline.fit.assert_not_called()

    def test_contrato_rejeita_valores_incoerentes_e_campos_extras(self):
        casos = [
            {"person_age": 17}, {"person_age": 101}, {"person_age": True}, {"person_age": "35"},
            {"person_income": 0}, {"person_income": -1}, {"person_income": None},
            {"loan_amnt": 0}, {"person_emp_length": 36}, {"person_emp_length": -1},
            {"cb_person_cred_hist_length": 36}, {"loan_int_rate": 0},
            {"loan_grade": "  "}, {"cb_person_default_on_file": "INVALIDO"},
            {"loan_status": 1}, {"loan_percent_income": .1},
        ]
        for alteracao in casos:
            with self.subTest(alteracao=alteracao):
                resposta = self.client.post("/predict", json={**self.baixo, **alteracao})
                self.assertEqual(resposta.status_code, 422, resposta.text)
        faltando = self.baixo.copy()
        faltando.pop("loan_amnt")
        self.assertEqual(self.client.post("/predict", json=faltando).status_code, 422)

    def test_nao_finitos_retornam_422_com_json_valido(self):
        for valor in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(valor=valor):
                resposta = self.client.post(
                    "/predict", content=json.dumps({**self.baixo, "person_income": valor}),
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(resposta.status_code, 422)
                self.assertTrue(resposta.json()["detail"])

    def test_opcionais_ausentes_usam_imputacao_salva(self):
        dados = {"person_age": 35, "person_income": 80000, "loan_amnt": 6000}
        resposta = self.client.post("/predict", json=dados)
        self.assertEqual(resposta.status_code, 200, resposta.text)
        self.assertTrue(0 <= resposta.json()["probabilidade_inadimplencia"] <= 1)

    def test_categorias_novas_e_normalizacao(self):
        dados = {**self.baixo, "person_home_ownership": " NOVA_CATEGORIA ",
                 "loan_grade": "z", "cb_person_default_on_file": " n "}
        self.assertEqual(self.client.post("/predict", json=dados).status_code, 200)
        validado = SolicitacaoCredito.model_validate(dados)
        self.assertEqual(validado.cb_person_default_on_file, "N")
        self.assertEqual(validado.loan_grade, "Z")

    def test_modelo_indisponivel_retorna_503(self):
        preditor = self.client.app.state.preditor
        try:
            self.client.app.state.preditor = None
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.assertEqual(self.client.post("/predict", json=self.baixo).status_code, 503)
        finally:
            self.client.app.state.preditor = preditor

    def test_swagger_e_openapi_expoem_contrato(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        resposta = self.client.get("/openapi.json")
        self.assertEqual(resposta.status_code, 200)
        documento = resposta.json()
        self.assertIn("get", documento["paths"]["/health"])
        self.assertIn("post", documento["paths"]["/predict"])
        contrato = documento["components"]["schemas"]["SolicitacaoCredito"]
        self.assertFalse(contrato["additionalProperties"])
        self.assertEqual(set(contrato["required"]), {"person_age", "person_income", "loan_amnt"})

    def test_startup_falha_com_modelo_ausente(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(RuntimeError, "Execute o notebook 06"):
                with TestClient(criar_app(Path(pasta) / "modelo_credito.joblib")):
                    pass

    def test_hash_incorreto_impede_carregamento(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "modelo_credito.joblib"
            arquivo.write_bytes(b"conteudo incorreto")
            arquivo.with_name("metadata_credito.json").write_text(json.dumps({"sha256_artefato": "0" * 64}))
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                carregar_preditor(arquivo)


if __name__ == "__main__":
    unittest.main()
