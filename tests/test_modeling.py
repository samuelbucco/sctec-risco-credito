"""Verifique integração dos estimadores e escolha baseada apenas em validação."""

import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from src.modeling import (
    comparar_validacao, configuracoes_modelos, criar_modelo,
    metricas_binarias, selecionar_finalistas,
)


class TestModeling(unittest.TestCase):
    def test_oito_configuracoes_com_balanceamento_e_escala_adequada(self):
        configuracoes = configuracoes_modelos()
        self.assertEqual(len({c["id"] for c in configuracoes}), 8)
        for familia in ["regressao_logistica", "random_forest"]:
            self.assertEqual(sum(c["familia"] == familia for c in configuracoes), 4)
        for c in configuracoes:
            modelo = criar_modelo(c["id"])
            self.assertEqual(modelo.named_steps["modelo"].class_weight, "balanced")
            numericas = modelo.named_steps["preprocessamento"].named_steps["codificacao"].transformers[0][1]
            if c["familia"] == "random_forest":
                self.assertEqual(numericas, "passthrough")
            else:
                self.assertEqual(type(numericas).__name__, "StandardScaler")

    def test_escolha_ignora_teste_e_desempata_por_complexidade(self):
        resultados = pd.DataFrame([
            {"configuracao": "rl_C_0.1", "familia": "regressao_logistica", "roc_auc_validacao_cv": .8, "f1_validacao_cv": .6, "roc_auc_teste": .99},
            {"configuracao": "rl_C_0.01", "familia": "regressao_logistica", "roc_auc_validacao_cv": .8, "f1_validacao_cv": .6, "roc_auc_teste": .7},
            {"configuracao": "rf_depth_3", "familia": "random_forest", "roc_auc_validacao_cv": .85, "f1_validacao_cv": .7, "roc_auc_teste": .6},
            {"configuracao": "rf_depth_6", "familia": "random_forest", "roc_auc_validacao_cv": .8, "f1_validacao_cv": .75, "roc_auc_teste": .99},
        ])
        escolhidos = selecionar_finalistas(resultados)
        self.assertEqual(set(escolhidos.configuracao), {"rl_C_0.01", "rf_depth_3"})

    def test_metricas_usam_classe_positiva_e_limiar_explicito(self):
        y = [0, 0, 1, 1]
        p = [.1, .6, .4, .9]
        padrao = metricas_binarias(y, p)
        ajustado = metricas_binarias(y, p, threshold=.3)
        self.assertEqual(padrao["f1"], .5)
        self.assertEqual(padrao["recall"], .5)
        self.assertEqual(ajustado["recall"], 1)
        self.assertEqual(padrao["roc_auc"], ajustado["roc_auc"])
        self.assertEqual(metricas_binarias([0, 1], [.5, .5])["recall"], 1)

    def test_validacao_das_duas_familias_com_entradas_brutas_e_nulos(self):
        n = 60
        i = np.arange(n)
        X = pd.DataFrame({
            "person_age": 25 + i % 20,
            "person_income": 30000 + i * 1000,
            "person_emp_length": np.where(i % 7 == 0, np.nan, i % 10),
            "loan_amnt": 3000 + (i % 3) * 4000,
            "loan_int_rate": np.where(i % 8 == 0, np.nan, 8 + i % 12),
            "cb_person_cred_hist_length": 3 + i % 5,
            "person_home_ownership": np.where(i % 2, "RENT", "OWN"),
            "loan_intent": ["PERSONAL"] * n,
            "loan_grade": np.where(i % 3, "B", "A"),
            "cb_person_default_on_file": ["N"] * n,
        })
        y = pd.Series((i % 3 == 2).astype(int))
        with contextlib.redirect_stdout(io.StringIO()):
            resultado = comparar_validacao(X, y, n_splits=2)
        self.assertEqual(len(resultado), 8)
        for coluna in ["roc_auc_validacao_cv", "f1_validacao_cv", "recall_validacao_cv"]:
            self.assertTrue(resultado[coluna].between(0, 1).all())
        self.assertEqual(len(selecionar_finalistas(resultado)), 2)


if __name__ == "__main__":
    unittest.main()
