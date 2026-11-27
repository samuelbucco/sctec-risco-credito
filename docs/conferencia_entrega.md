# Conferência da entrega do projeto

Conferência dos critérios do [roteiro do professor](SCTEC%20Aperfei%C3%A7oar.pdf). A implementação e a documentação técnica estão prontas. O vídeo e a submissão no AVA continuam pendentes.

## Critérios de avaliação

| Critério | Evidência no projeto | Situação |
| --- | --- | --- |
| 1 Apresentação e didática | [Roteiro de 6 minutos e 35 segundos](roteiro_video.md), com os cinco tópicos e demonstração ao vivo | Gravação e publicação pendentes |
| 2 Versionamento no GitHub | Commits incrementais e branches `feat/eda-limpeza`, `feat/pipeline-modelagem` e `feat/api-fastapi` | Histórico registrado |
| 3 Documentação | [README](../README.md) com contexto, dicionário das colunas, EDA, resultados, veredito e instalação | Atendido nos arquivos |
| 4 EDA e tratamento | [Notebook 03](../notebooks/03_analise_exploratoria.ipynb) com `.describe()`, quatro figuras e diagnóstico; [notebooks 02](../notebooks/02_separacao_treino_teste.ipynb) e [04](../notebooks/04_tratamento_features.ipynb) com repetições, nulos e inconsistências | Atendido nos arquivos |
| 5 Engenharia de atributos e prevenção de vazamento | [Função reutilizável](../src/features.py), imputação antes da razão, proteção contra divisor inválido e divisão estratificada antes das estatísticas | Atendido nos arquivos |
| 6 Pipeline e balanceamento | [Pré-processador](../src/preprocessing.py) com imputação, codificação e escala; pesos de classe ajustados apenas no treino | Atendido nos arquivos |
| 7 Hiperparâmetros | [Notebook 05](../notebooks/05_modelagem_validacao.ipynb): duas famílias, quatro valores por família e validação estratificada | Atendido nos arquivos |
| 8 Diagnóstico de overfitting | Tabelas e gráficos de ajuste, validação e teste, com interpretação e seleção por validação | Atendido nos arquivos |
| 9 Matriz de confusão e veredito | [Notebook 06](../notebooks/06_avaliacao_financeira.ipynb): relatórios completos dos dois candidatos, quatro matrizes, sensibilidade de custos e corte definido por OOF | Atendido nos arquivos |
| 10 FastAPI | [Contratos Pydantic](../app/schemas.py), [servidor](../app/main.py), carga do [artefato completo](../models/modelo_credito.joblib), `/health` e `/predict` | Atendido e verificado |

## Verificações da revisão técnica

- Os 26 testes passaram também em uma cópia limpa dos arquivos versionados, sem os CSVs locais, usando o ambiente Python 3.13 preparado para o projeto.
- A API utiliza o artefato incluído no repositório e pode ser executada sem repetir o treinamento.
- A [verificação HTTP](resultados/verificacao_api.json) registra exemplos de risco baixo e elevado, documentação e rejeição de renda zero.
- O [artefato e seus metadados](../models/metadata_credito.json) preservam o corte 0,295 e a identificação da versão usada na decisão.
- Os CSVs originais e derivados permanecem locais conforme o `.gitignore`; o README explica o download e os notebooks reproduzem as etapas.
- Os custos são fictícios: FP custa 1 UM e FN custa 5 UM no cenário principal. O ganho no teste foi de 33 UM, equivalente a 1,76%, sem demonstração de lucro real ou significância estatística.
- A floresta mantém sinal de overfitting e os scores não são probabilidades calibradas; essas limitações estão documentadas.

## Pendências para enviar ao AVA

- [ ] Ensaiar o [roteiro do vídeo](roteiro_video.md) com cronômetro.
- [ ] Gravar a tela e a voz, incluindo Uvicorn e uma requisição real no Swagger.
- [ ] Conferir duração de até sete minutos, áudio e legibilidade.
- [ ] Publicar o vídeo como não listado no YouTube ou com acesso de leitura no Google Drive.
- [ ] Testar o link do vídeo em uma janela sem login.
- [ ] Adicionar o link do vídeo ao README e registrá-lo em um novo commit.
- [ ] Submeter os dois links no AVA e conferir o registro da entrega.

**Repositório:** [samuelbucco/sctec-risco-credito](https://github.com/samuelbucco/sctec-risco-credito).

**Vídeo:** aguardando gravação e publicação.

**Prazo indicado no roteiro:** 30/11/2026, na atividade Projeto Avaliativo Final do AVA. O roteiro informa a data, sem horário limite.
