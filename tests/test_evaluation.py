"""Verificações dos custos e da regra de decisão financeira simulada."""

import unittest

import numpy as np

from src.evaluation import selecionar_threshold, tabela_custos


class TestEvaluation(unittest.TestCase):
    def test_contagens_custo_e_recall(self):
        tabela = tabela_custos([0, 0, 1, 1], [.1, .6, .4, .9], thresholds=[.3, .5])
        padrao = tabela.loc[tabela.threshold == .5].iloc[0]
        self.assertEqual((padrao.tn, padrao.fp, padrao.fn, padrao.tp), (1, 1, 1, 1))
        self.assertEqual(padrao.custo_total, 6)
        self.assertEqual(padrao.custo_medio, 1.5)
        escolhido = selecionar_threshold(tabela)
        self.assertEqual(escolhido["threshold"], .3)
        self.assertEqual(escolhido["custo_total"], 1)
        self.assertEqual(escolhido["recall"], 1)

    def test_empate_favorece_menos_falsos_negativos_e_menor_corte(self):
        tabela = tabela_custos([0, 1], [.5, .5], custo_fp=1, custo_fn=1, thresholds=[1, .5, 0])
        escolhido = selecionar_threshold(tabela)
        self.assertEqual(escolhido["threshold"], 0)
        self.assertEqual(escolhido["fn"], 0)
        self.assertEqual(tabela.loc[tabela.threshold == .5, "tp"].iloc[0], 1)

    def test_escala_dos_custos_preserva_a_decisao(self):
        alvo, probabilidades = [0, 0, 1, 1], [.1, .6, .4, .9]
        a = selecionar_threshold(tabela_custos(alvo, probabilidades, 1, 5))
        b = selecionar_threshold(tabela_custos(alvo, probabilidades, 100, 500))
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertEqual(b["custo_total"], a["custo_total"] * 100)

    def test_entradas_invalidas_sao_rejeitadas(self):
        for alvo, probabilidades, opcoes in [
            ([0, 1], [.1, np.nan], {}),
            ([0, 2], [.1, .9], {}),
            ([0, 1], [.1], {}),
            ([0, 1], [.1, 1.1], {}),
            ([0, 1], [.1, .9], {"custo_fn": -1}),
            ([0, 1], [.1, .9], {"thresholds": [1.1]}),
        ]:
            with self.subTest(alvo=alvo, opcoes=opcoes):
                with self.assertRaises(ValueError):
                    tabela_custos(alvo, probabilidades, **opcoes)


if __name__ == "__main__":
    unittest.main()
