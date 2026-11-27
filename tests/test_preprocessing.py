"""Verificações de regras de negócio, isolamento do ajuste e serialização."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import adicionar_comprometimento_renda
from src.preprocessing import COLUNAS_MEDIANA, criar_preprocessador, tratar_inconsistencias


def amostra():
    return pd.DataFrame({
        "person_age": [25, 30, 35, 40],
        "person_income": [10000, 20000, 30000, 40000],
        "person_emp_length": [2, np.nan, 10, 8],
        "loan_amnt": [1000, 2000, 3000, 4000],
        "loan_int_rate": [10, 12, np.nan, 14],
        "cb_person_cred_hist_length": [2, 3, 5, 8],
        "person_home_ownership": ["RENT", "OWN", "RENT", "OWN"],
        "loan_intent": ["PERSONAL"] * 4,
        "loan_grade": ["A", "B", "A", "B"],
        "cb_person_default_on_file": ["N", "Y", "N", "N"],
        "loan_percent_income": [0.1] * 4,
    })


class TestPreprocessing(unittest.TestCase):
    def test_razao_percentual_e_divisores_invalidos(self):
        dados = pd.DataFrame({
            "person_income": [10000, 0, np.nan, np.inf, -10000],
            "loan_amnt": [2000, 2000, 2000, 2000, 2000],
        })
        original = dados.copy(deep=True)
        resultado = adicionar_comprometimento_renda(dados)
        self.assertEqual(resultado.loc[0, "comprometimento_renda"], 20)
        self.assertTrue(resultado.loc[1:, "comprometimento_renda"].isna().all())
        self.assertFalse(np.isinf(resultado["comprometimento_renda"]).any())
        pd.testing.assert_frame_equal(dados, original)

    def test_inconsistencias_sem_remover_linhas_ou_extremos_validos(self):
        dados = amostra()
        dados.loc[0, "person_age"] = 144
        dados.loc[1, "person_emp_length"] = 123
        dados.loc[2, "person_income"] = 6000000
        dados.loc[2, "loan_int_rate"] = np.inf
        dados.loc[3, "person_emp_length"] = 0
        original = dados.copy(deep=True)
        resultado = tratar_inconsistencias(dados)
        self.assertTrue(pd.isna(resultado.loc[0, "person_age"]))
        self.assertTrue(pd.isna(resultado.loc[1, "person_emp_length"]))
        self.assertTrue(pd.isna(resultado.loc[2, "loan_int_rate"]))
        self.assertEqual(resultado.loc[2, "person_income"], 6000000)
        self.assertEqual(resultado.loc[3, "person_emp_length"], 0)
        self.assertTrue(resultado.index.equals(dados.index))
        self.assertNotIn("loan_percent_income", resultado.columns)
        pd.testing.assert_frame_equal(dados, original)

    def test_alvo_e_colunas_ausentes_sao_rejeitados(self):
        with self.assertRaisesRegex(ValueError, "Separe loan_status"):
            tratar_inconsistencias(amostra().assign(loan_status=0))
        with self.assertRaisesRegex(ValueError, "person_income"):
            tratar_inconsistencias(amostra().drop(columns="person_income"))

    def test_imputacao_antes_da_feature(self):
        dados = amostra()
        dados.loc[0, "person_income"] = np.nan
        dados.loc[0, "loan_amnt"] = 2000
        dados.loc[0, "person_home_ownership"] = "  "
        pipeline = criar_preprocessador()
        transformado = pipeline.fit_transform(dados)
        antes_encoding = pipeline[:3].transform(dados)
        self.assertEqual(antes_encoding.loc[0, "person_income"], 30000)
        self.assertAlmostEqual(antes_encoding.loc[0, "comprometimento_renda"], 2000 / 30000 * 100)
        self.assertEqual(antes_encoding.loc[2, "loan_int_rate"], 12)
        self.assertTrue(np.isfinite(transformado).all())

    def test_validacao_nao_altera_estatisticas_e_aceita_categoria_nova(self):
        dados = amostra()
        ajuste = dados.iloc[:3].copy()
        validacao = dados.iloc[[3]].copy()
        validacao.loc[:, "person_income"] = 6000000
        validacao.loc[:, "person_home_ownership"] = "CATEGORIA_NOVA"
        pipeline = criar_preprocessador().fit(ajuste)
        imputador = pipeline.named_steps["imputacao"].named_transformers_["medianas"]
        antes = imputador.statistics_.copy()
        self.assertEqual(antes[COLUNAS_MEDIANA.index("person_income")], 20000)
        matriz = pipeline.transform(validacao)
        np.testing.assert_array_equal(antes, imputador.statistics_)
        self.assertEqual(matriz.shape[0], 1)
        self.assertTrue(np.isfinite(matriz).all())
        codificador = pipeline.named_steps["codificacao"].named_transformers_["categoricas"]
        self.assertNotIn("CATEGORIA_NOVA", codificador.categories_[0])

    def test_sem_escalonamento_preserva_valores_numericos(self):
        dados = amostra()
        pipeline = criar_preprocessador(escalonar=False)
        resultado = pipeline.fit_transform(dados)
        nomes = pipeline.named_steps["codificacao"].get_feature_names_out().tolist()
        np.testing.assert_allclose(resultado[:, nomes.index("comprometimento_renda")], 10)
        np.testing.assert_allclose(resultado[:, nomes.index("person_income")], dados.person_income)

    def test_serializacao_preserva_resultados(self):
        dados = amostra()
        pipeline = criar_preprocessador().fit(dados)
        esperado = pipeline.transform(dados)
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "preprocessador.joblib"
            joblib.dump(pipeline, arquivo)
            restaurado = joblib.load(arquivo)
            np.testing.assert_allclose(restaurado.transform(dados), esperado)


if __name__ == "__main__":
    unittest.main()
