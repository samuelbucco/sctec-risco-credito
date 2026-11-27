# Previsão de risco de inadimplência em operações de crédito

Projeto avaliativo do curso **Integração de IA com Dados**, da etapa **Aperfeiçoar do programa SCTec**.

**Status:** implementação, modelo salvo, API e revisão técnica concluídos. O roteiro de apresentação e a conferência da entrega estão em `docs/`. Pendem a gravação do vídeo, a conferência de acesso e a submissão dos links no AVA.

## Início rápido da API

Com Python 3.13 instalado, execute os comandos na raiz deste repositório (macOS ou Linux):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Abra o [Swagger UI](http://127.0.0.1:8000/docs) para testar `GET /health` e `POST /predict`. O pipeline já está incluído em `models/`; executar a API não exige baixar os CSVs ou refazer o treinamento. Para reproduzir o desenvolvimento, siga os notebooks em ordem.

## Contexto e objetivo

Uma instituição financeira precisa avaliar o risco de inadimplência antes de conceder um empréstimo. Aprovar crédito para quem não pagará pode gerar perdas financeiras; recusar um bom pagador pode resultar em perda de receita e de relacionamento com o cliente.

O objetivo deste projeto é desenvolver e avaliar modelos de aprendizado de máquina para estimar a probabilidade de inadimplência e disponibilizar o modelo escolhido por meio de uma API em FastAPI.

A solução tem finalidade educacional. Os custos financeiros adotados na comparação serão identificados como hipóteses, caso não sejam fornecidos na base ou no roteiro.

## Dados e variável alvo

- **Base:** [CSV de risco de crédito indicado pelo professor](https://drive.google.com/file/d/1R4WCMc_56lv3fMaDalUBAz5jSxeZOcwI/view), referente à Opção A do roteiro.
- **Variável alvo:** `loan_status`.
- **Classe 0:** cliente que honra o empréstimo em dia.
- **Classe 1:** cliente inadimplente.
- **Roteiro:** [PDF do projeto](docs/SCTEC%20Aperfei%C3%A7oar.pdf).

### Dicionário de dados

Este dicionário de trabalho descreve as 12 colunas originais e a variável criada. As definições são baseadas no roteiro, nos nomes e nos valores observados. A moeda dos valores, o período da taxa e os critérios das notas não foram informados; a disponibilidade temporal dos atributos precisa de confirmação antes de uso fora do projeto educacional.

| Coluna | Significado | Tipo lógico | Uso |
| --- | --- | --- | --- |
| `person_age` | Idade em anos | Numérico inteiro | Entrada |
| `person_income` | Renda anual, moeda não informada | Numérico | Entrada e componente da nova variável |
| `person_home_ownership` | Situação de moradia: aluguel, imóvel próprio, financiamento ou outra | Categórico | Entrada |
| `person_emp_length` | Tempo de emprego, interpretado em anos | Numérico | Entrada; aceita ausência na API |
| `loan_intent` | Finalidade do empréstimo | Categórico | Entrada |
| `loan_grade` | Nota atribuída ao empréstimo | Categórico | Entrada, sem impor escala ordinal |
| `loan_amnt` | Valor total do empréstimo, na mesma moeda da renda | Numérico | Entrada e componente da nova variável |
| `loan_int_rate` | Taxa de juros registrada, com período e unidade a confirmar | Numérico | Entrada; aceita ausência na API |
| `loan_status` | `0` para adimplente e `1` para inadimplente | Categoria binária | Alvo; excluído das entradas |
| `loan_percent_income` | Indicador original da relação entre empréstimo e renda | Numérico | Desconsiderado no pipeline, conforme diagnóstico de divergência |
| `cb_person_default_on_file` | Histórico de inadimplência: `Y` ou `N` | Categórico | Entrada; diferente do alvo atual |
| `cb_person_cred_hist_length` | Tempo de histórico de crédito, interpretado em anos | Numérico | Entrada; aceita ausência na API |
| `comprometimento_renda` | `(loan_amnt / person_income) × 100`, em porcentagem | Numérico contínuo | Criado no pipeline após imputar os componentes |

### Obtenção e conferência do arquivo original

1. Acesse o link da base acima e faça o download de `credit_risk_dataset.csv`.
2. Salve o arquivo em `data/raw/credit_risk_dataset.csv`, sem editar ou salvar novamente o conteúdo em uma planilha.
3. Mantenha o original nessa pasta; arquivos derivados serão gravados separadamente em `data/processed/`.

Conferência de aquisição realizada em **20/11/2026**:

- Tamanho: **1.804.682 bytes**, igual ao informado pelo Google Drive.
- Dimensões: **32.581 registros e 12 colunas**.
- A coluna alvo `loan_status` está presente e todas as linhas possuem a quantidade esperada de campos.
- O arquivo local preserva os bytes do download, sem limpeza, conversão ou outras transformações.
- SHA-256: `ce3c6d2167717bf1627d1c0c81cbccd28323cd4aa7b96d542599366d5ff6aac8`.

O [primeiro notebook](notebooks/01_conhecimento_dados.ipynb) apresenta os tipos, um dicionário inicial das 12 colunas e as verificações de reconhecimento. As descrições inferidas dos nomes e valores estão identificadas como provisórias; moeda, critérios das notas e algumas unidades precisam de confirmação na origem.

A inspeção inicial identificou 895 valores ausentes em `person_emp_length`, 3.116 em `loan_int_rate` e 165 linhas repetidas além da primeira ocorrência. Não foi aplicado tratamento ao CSV. A decisão sobre imputação e valores discrepantes será fundamentada na análise do conjunto de treino.

### Repetições e separação de treino e teste

O [segundo notebook](notebooks/02_separacao_treino_teste.ipynb) remove as 165 repetições exatas na cópia em memória, mantendo a primeira ocorrência, e divide os 32.416 registros restantes com `test_size=0.20`, `stratify=y` e `random_state=42`.

Como não há identificador de contrato, tratar linhas idênticas em todas as colunas como repetições é uma hipótese de trabalho. A regra atende ao roteiro e evita cópias em treino e teste, mas pode excluir empréstimos distintos com os mesmos valores nas colunas disponíveis. Não são removidos registros por coincidências parciais, como ter a mesma idade ou renda.

| Conjunto | Registros | Arquivo local |
| --- | ---: | --- |
| Treino | 25.932 | `data/processed/treino.csv` |
| Teste | 6.484 | `data/processed/teste.csv` |

Cada CSV contém os 11 atributos originais e o alvo `loan_status`, sem coluna adicional de índice. Ao preparar os modelos, o alvo deve ser separado dos atributos. `data/processed/separacao.json` registra a semente, a regra de repetição, as versões e os hashes para reprodução.

Foram conferidos o alinhamento dos atributos com o alvo, a cobertura dos registros, a ausência de entradas idênticas entre os conjuntos, a leitura dos arquivos exportados e a preservação do original. Nulos e valores discrepantes permanecem nos conjuntos. Nenhum imputador, escalonador, codificador ou técnica de balanceamento foi ajustado nesta etapa.

A análise exploratória utiliza apenas `treino.csv`. O teste fica reservado para a avaliação final; a escolha dos hiperparâmetros e do ponto de corte utilizará validação dentro do treino.

### Principais descobertas da análise exploratória

O [terceiro notebook](notebooks/03_analise_exploratoria.ipynb) apresenta dimensões, tipos, estatísticas descritivas, quatro figuras e um diagnóstico que orienta o tratamento. Todos os resultados abaixo se referem exclusivamente aos **25.932 registros de treino**.

| Achado | Evidência e implicação |
| --- | --- |
| Classes desbalanceadas | 20.261 adimplentes (78,13%) e 5.671 inadimplentes (21,87%). Acurácia isolada é insuficiente; avaliar recall, F1, ROC-AUC e custos dos erros. |
| Tempo de emprego ausente | 727 nulos (2,80%), média 4,77, mediana 4 e assimetria 2,94. A mediana é candidata à imputação após o tratamento de inconsistências. |
| Taxa de juros ausente | 2.488 nulos (9,59%), média 11,02, mediana 10,99 e assimetria 0,20. A média é candidata à imputação pela distribuição mais simétrica. |
| Renda com cauda longa | Mediana 55.000, média 65.828,51 e máximo 6.000.000, em moeda não informada. Extremos financeiros não serão excluídos apenas por serem raros. |
| Plausibilidade a investigar | Cinco idades acima de 100 anos e dois registros com tempo de emprego maior que a idade. As regras de tratamento precisam ser explícitas. |
| Associações com inadimplência | Correlação de Pearson de 0,380 com `loan_percent_income` e 0,334 com `loan_int_rate`; associação não demonstra causalidade. |
| Nova variável não é mera renomeação | A razão empréstimo/renda calculada difere da coluna existente em mais de 0,5 ponto percentual em 339 registros, com diferença máxima de 9,34 pontos. |

A regra IQR identifica valores incomuns sem provar erro: por exemplo, seu limite superior para idade é 40,5 anos, o que não justifica remover todos os clientes acima dessa idade. A EDA não modifica os dados. A imputação e qualquer transformação estatística serão ajustadas dentro dos pipelines e das partições de treino da validação.

Figuras exportadas pelo notebook:

- [Distribuição do alvo](docs/figuras/01_distribuicao_alvo.png).
- [Distribuições numéricas](docs/figuras/02_distribuicoes_numericas.png).
- [Boxplots com valores extremos](docs/figuras/03_boxplots.png).
- [Matriz de correlação](docs/figuras/04_correlacao.png).

### Atributo obrigatório

Criar uma variável numérica que compare o valor do empréstimo à renda anual:

```text
comprometimento_renda = (loan_amnt / person_income) × 100
```

Esse indicador representa a razão percentual entre o valor total do empréstimo e a renda anual, e não o comprometimento mensal com parcelas.

O cálculo está implementado em `adicionar_comprometimento_renda()`, no arquivo [src/features.py](src/features.py). A função preserva a entrada e produz `NaN` para componentes ausentes, não finitos ou não positivos, sem divisão por zero. No pipeline, os componentes são imputados antes do cálculo. A API utilizará o pipeline completo salvo, com a mesma função.

| Coluna derivada | Tipo | Significado e unidade |
| --- | --- | --- |
| `comprometimento_renda` | Numérico contínuo | Valor total do empréstimo dividido pela renda anual, multiplicado por 100; resultado em porcentagem |

### Tratamento e pré-processamento reutilizáveis

O [quarto notebook](notebooks/04_tratamento_features.ipynb) demonstra as funções de [src/preprocessing.py](src/preprocessing.py) e a nova variável. As regras não removem registros nem alteram o CSV original:

- Idades fora de 18 a 100 anos viram ausentes. Esse intervalo é uma convenção educacional para o projeto, não um limite universal.
- Renda, valor emprestado e taxa de juros não positivos viram ausentes, conforme a hipótese de produto adotada. Campos numéricos não interpretáveis ou infinitos também viram ausentes.
- Tempo de emprego e histórico de crédito negativos ou superiores à idade original viram ausentes; zero continua permitido.
- Espaços nas extremidades das categorias são removidos; textos vazios viram ausentes.
- Extremos financeiros positivos são mantidos. O IQR serve como evidência de raridade, sem justificar corte automático.
- `loan_percent_income` é desconsiderada em favor da nova razão calculada, evitando duas versões quase redundantes e divergentes do indicador.

No treino, as regras adicionam cinco ausências em idade e duas em tempo de emprego. Os valores imputados são aproximações estatísticas, não correções comprovadas dos valores originais.

`criar_preprocessador(escalonar=True)` constrói um pipeline ainda não ajustado, com esta ordem:

1. Regras fixas de qualidade.
2. Imputação pela mediana dos campos numéricos, exceto taxa de juros, que usa média; categorias usam moda quando necessário.
3. Cálculo de `comprometimento_renda` a partir dos componentes já imputados.
4. `OneHotEncoder(handle_unknown="ignore")` nas categorias e `StandardScaler` nos numéricos.

A mediana é motivada pela assimetria do tempo de emprego e pela robustez aos extremos; a média da taxa é motivada pela distribuição mais simétrica observada na EDA. A opção `escalonar=False` preserva os valores numéricos para modelos baseados em árvores.

Na demonstração, o ajuste usa 20.745 registros e a validação interna, 5.187, ambos derivados do treino. As duas matrizes têm 26 colunas, sem valores ausentes ou infinitos. O teste final não é utilizado. Durante a comparação dos modelos, o pré-processador deverá ser incluído junto do classificador em um único pipeline para que cada partição de validação aprenda suas próprias estatísticas. Não se deve fornecer à validação cruzada uma tabela imputada globalmente.

### Comparação de modelos e overfitting

O [quinto notebook](notebooks/05_modelagem_validacao.ipynb) executa as funções de [src/modeling.py](src/modeling.py), comparando duas famílias com quatro configurações cada:

- Regressão Logística: `C` em 0,01; 0,1; 1; 10, com padronização numérica.
- Random Forest: `max_depth` em 3; 6; 10; sem limite, com 100 árvores e sem padronização.

Ambas utilizam `class_weight="balanced"` apenas no ajuste. A validação estratificada tem três folds, semente 42 e as mesmas partições para todas as configurações. O pipeline completo, incluindo imputação e encoding, é ajustado dentro de cada fold. O critério predefinido é maior ROC-AUC média de validação por família, com F1 como desempate e, depois, menor complexidade.

Os finalistas são congelados antes de carregar o teste. A tabela de treino e teste para as oito configurações atende ao roteiro e não reabre a busca. Custos e ponto de corte são escolhidos por validação no treino, como descrito na seção financeira, sem adaptar o protocolo aos resultados do teste.

| Finalista por família | ROC-AUC CV (média ± desvio) | ROC-AUC teste | F1 teste | Recall teste |
| --- | ---: | ---: | ---: | ---: |
| Regressão Logística, `C=10` | 0,8729 ± 0,0053 | 0,8720 | 0,6480 | 0,7743 |
| Random Forest, `max_depth=None` | 0,9284 ± 0,0008 | 0,9299 | 0,8198 | 0,7525 |

F1 e recall usam a classe positiva inadimplente e ponto de corte 0,5. O desvio é calculado entre folds e não representa um intervalo de confiança. A vantagem de `C=10` sobre `C=1` é de aproximadamente 0,00003 na ROC-AUC média, sem demonstrar superioridade estatística.

**Diagnóstico:** a Regressão Logística apresenta diferenças pequenas entre ajuste e validação (até aproximadamente 0,0013). Na Random Forest, o gap de ROC-AUC CV passa de 0,0012 na profundidade 3 para 0,0295 na 10 e 0,0716 sem limite. A floresta sem limite atinge ROC-AUC 1,0000 no ajuste, evidenciando memorização; sua média de validação, porém, é a maior entre as configurações comparadas. A profundidade 10 reduz o sobreajuste com ROC-AUC CV 0,9269. O protocolo mantém a configuração sem limite como candidata pelo critério fixado, com essa limitação documentada.

O gap de treino e teste dos finalistas é 0,0020 na logística e 0,0701 na floresta. A floresta apresenta melhor ranking e F1, mas menor recall no corte inicial. A avaliação financeira a seguir fundamenta a escolha do modelo e do ponto de corte.

Resultados reproduzíveis:

- [Comparação de ajuste e validação cruzada](docs/resultados/comparacao_validacao.csv).
- [Comparação de treino e teste](docs/resultados/comparacao_treino_teste.csv).
- [Finalistas e protocolo registrados](docs/resultados/selecao_modelos.json).
- [Gráfico de validação](docs/figuras/05_validacao_modelos.png).
- [Gráfico de treino e teste](docs/figuras/06_overfitting_modelos.png).

### Avaliação financeira, veredito e artefato

O [sexto notebook](notebooks/06_avaliacao_financeira.ipynb) usa as funções de [src/evaluation.py](src/evaluation.py) para gerar previsões fora da partição de ajuste (OOF), comparar 201 cortes entre 0 e 1 e escolher a política antes de avaliar o teste.

**Hipótese didática:** falso positivo (sinalizar como inadimplente quem pagaria) custa **1 UM**; falso negativo (sinalizar como seguro quem ficaria inadimplente) custa **5 UM**. UM é unidade monetária fictícia, sem equivalência atribuída à moeda dos empréstimos. O custo é `FP × custo_fp + FN × custo_fn`, com custo zero nos acertos. Não representa lucro completo, recuperação de crédito ou custo individual dos contratos.

O corte minimiza o custo OOF, com desempate por menos FN e menor corte. O candidato minimiza o custo OOF, depois FN e, em empate completo, prefere a logística por simplicidade. Os custos de desenvolvimento são usados na escolha e não constituem uma estimativa imparcial de desempenho após essa seleção.

| Candidato | Corte escolhido por OOF | Custo OOF (UM fictícias) | Custo teste no corte escolhido (UM fictícias) |
| --- | ---: | ---: | ---: |
| Random Forest sem limite de profundidade | **0,295** | **7.253** | **1.840** |
| Regressão Logística `C=10` | 0,490 | 9.859 | 2.480 |

**Veredito:** a Random Forest com corte **0,295** é o modelo escolhido para a API educacional, pelo menor custo OOF no cenário definido. A decisão foi registrada antes da leitura do teste nesta etapa, preservando os hiperparâmetros da fase anterior.

No teste, alterar o corte da floresta de 0,5 para 0,295 produz esta troca:

| Indicador | Corte 0,5 | Corte 0,295 |
| --- | ---: | ---: |
| Falsos positivos | 118 | 590 |
| Falsos negativos | 351 | 250 |
| Recall dos inadimplentes | 75,25% | 82,37% |
| Precisão dos alertas | 90,04% | 66,44% |
| F1 | 0,8198 | 0,7355 |
| Custo simulado (UM fictícias) | 1.873 | 1.840 |

O custo cai **33 UM (1,76%)** nessa divisão: 101 FN evitados correspondem a 505 UM, contra 472 UM por FP adicionais. É um ganho pequeno, sem demonstração de significância estatística ou lucro real. Na logística, o custo sobe de 2.473 para 2.480 UM no teste apesar da ligeira melhora OOF; mantemos o corte fixado no desenvolvimento.

A análise de sensibilidade usa relações FN:FP de 1:1, 2:1, 5:1, 10:1 e 20:1. A floresta tem menor custo OOF nos cinco cenários, mas seu corte vai de 0,675 a 0,105. O corte 0,295 depende das hipóteses escolhidas. Os scores não foram calibrados probabilisticamente; esse corte não comprova risco real de 29,5%. O sinal de overfitting da floresta permanece como limitação.

Arquivos gerados:

- [Curvas de custo OOF](docs/resultados/custos_threshold_oof.csv) e [sensibilidade às hipóteses](docs/resultados/sensibilidade_custos.csv).
- [Decisão congelada antes do teste](docs/resultados/decisao_financeira.json).
- [Resultados financeiros no teste](docs/resultados/avaliacao_financeira_teste.csv) e [relatórios completos de classificação](docs/resultados/relatorios_classificacao.json).
- [Gráfico de custo por corte](docs/figuras/07_custos_threshold.png) e [quatro matrizes de confusão](docs/figuras/08_matrizes_confusao.png).
- [Pipeline campeão e threshold](models/modelo_credito.joblib), acompanhado de [metadados, versões e SHA-256](models/metadata_credito.json).

O `.joblib` contém o pipeline completo ajustado no treino, `threshold`, configuração, campos de entrada e cenário de custos. A recarga preservou exatamente as previsões. A API deve aplicar o corte salvo aos scores da classe 1; usar somente `pipeline.predict()` retornaria à decisão padrão do estimador.

### API REST com FastAPI

O [servidor](app/main.py) carrega o artefato uma vez na inicialização, com [validação do SHA-256 e inferência](app/prediction.py). Os [contratos Pydantic](app/schemas.py) definem as entradas e saídas. Cada requisição utiliza `predict_proba` e aplica o corte salvo, sem chamar `fit`.

| Endpoint | Função | Resposta |
| --- | --- | --- |
| `GET /health` | Conferir que o modelo está carregado | Status, configuração e corte |
| `POST /predict` | Prever o risco de um registro | Score estimado, classe, risco, ação recomendada, corte e configuração |
| `GET /docs` | Abrir o Swagger UI | Documentação interativa |
| `GET /openapi.json` | Consultar o contrato da API | Especificação OpenAPI |

Regras do contrato:

- `person_age`, `person_income` e `loan_amnt` são obrigatórios. Idade deve ser inteira entre 18 e 100; renda e valor emprestado devem ser numéricos, finitos e positivos.
- Os outros sete campos podem ser omitidos ou enviados como `null`; a imputação usa os parâmetros já aprendidos no treino. Taxa informada deve ser positiva; tempos informados não podem ser negativos ou maiores que a idade.
- Categorias são normalizadas para maiúsculas e têm espaços externos removidos. Categorias novas de moradia, finalidade e nota são aceitas pelo encoder. O histórico de inadimplência aceita `Y`, `N` ou ausência.
- Campos extras são rejeitados, incluindo `loan_status`, `loan_percent_income` e a variável calculada. A API calcula a nova variável pelo pipeline.
- Entradas inválidas retornam HTTP 422. Modelo indisponível retorna HTTP 503; arquivo ausente ou SHA-256 incompatível impede a inicialização.

Os exemplos são sintéticos e representam os dois lados do corte no artefato atual:

| Exemplo | Score estimado | Classe | Risco |
| --- | ---: | ---: | --- |
| [Baixo risco](docs/exemplos/credito_baixo_risco.json) | 0,03 | 0 | baixo |
| [Alto risco](docs/exemplos/credito_alto_risco.json) | 0,97 | 1 | elevado |

As ações são `Prosseguir com a análise de crédito.` para classe 0 e `Encaminhar para revisão de crédito.` para classe 1. O campo `probabilidade_inadimplencia` contém o score estimado do modelo, sem calibração probabilística.

Uma [verificação com requisições HTTP reais](docs/resultados/verificacao_api.json) registra `/health`, Swagger, OpenAPI, os dois exemplos e a rejeição de renda zero. O serviço foi executado localmente com Uvicorn; o repositório não configura hospedagem externa.

## Escopo e requisitos

1. **Análise exploratória:** apresentar dimensões, tipos, estatísticas descritivas, pelo menos três gráficos explicativos e interpretação dos resultados.
2. **Limpeza:** identificar duplicatas, nulos e valores discrepantes e justificar as decisões de tratamento.
3. **Engenharia de atributos:** implementar a variável obrigatória em uma função reutilizável.
4. **Pré-processamento seguro:** reservar 20% dos dados para teste com separação estratificada; ajustar imputação, codificação e escalonamento apenas no treino, usando `ColumnTransformer` e `Pipeline`. Aplicar `OneHotEncoder(handle_unknown="ignore")` às variáveis categóricas e `StandardScaler` às numéricas quando exigido pelo algoritmo. Tratar o desbalanceamento exclusivamente no treino.
5. **Modelagem:** comparar pelo menos duas famílias de algoritmos e testar pelo menos quatro valores de um hiperparâmetro de complexidade por família. Comparar métricas de treino e teste e discutir overfitting.
6. **Avaliação e escolha:** apresentar relatórios de classificação e matrizes de confusão dos dois melhores modelos, analisar os custos dos erros e avaliar o ponto de corte de probabilidade. Salvar o pipeline escolhido e o ponto de corte com `joblib` em `models/`.
7. **API:** definir os contratos de entrada e saída com Pydantic em `app/schemas.py` e implementar `app/main.py`, carregando o artefato na inicialização. Disponibilizar `GET /health` e `POST /predict`, retornando probabilidade, classificação de risco e ação recomendada.

### Cuidados de avaliação

A base de teste será separada antes de ajustar qualquer transformação estatística. A análise que orientar as escolhas de tratamento se concentrará nos dados de treino. Duplicatas serão verificadas para evitar que cópias do mesmo registro apareçam nos dois conjuntos.

A seleção de hiperparâmetros e do ponto de corte será orientada por validação dentro do conjunto de treino. As comparações de treino e teste exigidas no roteiro serão apresentadas, preservando o teste para a avaliação final.

## Planejamento de execução

- [x] Criar a pasta do projeto e abri-la no VS Code.
- [x] Ler o roteiro e escolher o tema de risco de crédito.
- [x] Registrar o planejamento inicial neste README.
- [x] Criar a estrutura de pastas e o `.gitignore`.
- [x] Preparar o ambiente Python e registrar as dependências.
- [x] Iniciar o Git e adotar branches e commits por etapa.
- [x] Obter a base e preservar os dados originais.
- [x] Criar e executar o notebook de reconhecimento, com tipos, dicionário inicial, nulos e verificação de duplicatas.
- [x] Tratar as repetições exatas com justificativa e separar treino e teste.
- [x] Fazer a análise exploratória e documentar os achados.
- [x] Implementar o tratamento dos dados e a nova variável.
- [x] Construir os pipelines e comparar os modelos.
- [x] Avaliar overfitting, custos dos erros e pontos de corte.
- [x] Salvar o pipeline e o ponto de corte escolhidos.
- [x] Implementar e verificar a API.
- [x] Completar a documentação e conferir a execução do projeto.
- [x] Publicar o repositório público no GitHub.
- [ ] Gravar o vídeo de até sete minutos e conferir o acesso ao link.
- [ ] Entregar os links do repositório e do vídeo no AVA.

## Estrutura do projeto

As pastas de trabalho e o arquivo de dependências já estão criados.

```text
Projeto/
├── docs/              # Roteiro e materiais de apoio
├── data/
│   ├── raw/           # Dados originais, preservados sem alterações
│   └── processed/     # Dados derivados do processamento
├── notebooks/         # Análise exploratória e experimentos
├── src/               # Funções reutilizáveis
├── tests/             # Verificações de regras e pré-processamento
├── models/            # Pipeline e ponto de corte salvos
├── app/               # API FastAPI e contratos Pydantic
├── README.md          # Apresentação e guia do projeto
├── requirements.txt   # Dependências Python com versões registradas
├── requirements-dev.txt # Dependências adicionais dos testes da API
└── .gitignore         # Arquivos excluídos do versionamento
```

Os arquivos vazios `.gitkeep` permitem que o Git registre as pastas enquanto elas ainda não têm conteúdo. Eles podem ser removidos quando as pastas receberem os arquivos do projeto.

O `.gitignore` exclui do versionamento o ambiente virtual, caches do Python, checkpoints dos notebooks, logs, arquivos locais do macOS e arquivos `.env` com credenciais. As bases em `data/raw/` e `data/processed/` também ficam locais por padrão; a origem e o procedimento para obtê-las serão documentados para permitir a reprodução do projeto. Os artefatos finais em `models/` poderão ser versionados para acompanhar a entrega exigida no roteiro.

## Controle de versão

O projeto utiliza Git para registrar sua evolução. A branch `main` guarda a versão consolidada; `feat/eda-limpeza` reúne análise e preparação dos dados, `feat/pipeline-modelagem` contém modelagem e avaliação financeira, e `feat/api-fastapi` acrescenta o serviço de inferência.

Os commits devem descrever mudanças concretas, por exemplo:

```text
chore: estrutura projeto e configura ambiente Python
feat: adiciona analise exploratoria de credito
fix: protege calculo de comprometimento contra divisao por zero
```

Repositório público: [samuelbucco/sctec-risco-credito](https://github.com/samuelbucco/sctec-risco-credito).

A conexão remota `origin` aponta para `https://github.com/samuelbucco/sctec-risco-credito.git`. O envio inicial foi concluído e a branch local `main` acompanha `origin/main` no GitHub.

## Resultados e decisões documentados

Os notebooks, módulos e seções acima registram:

- Dicionário de dados, incluindo `comprometimento_renda`.
- Principais descobertas da análise exploratória.
- Justificativas para limpeza e pré-processamento.
- Algoritmos, hiperparâmetros e resultados comparativos.
- Diagnóstico de overfitting.
- Hipóteses de custos para falsos positivos e falsos negativos.
- Modelo e ponto de corte escolhidos, com justificativa.
- Limitações da solução.

## Como executar

### Preparação do ambiente

Ambiente preparado com **Python 3.13**. Execute os comandos abaixo no terminal, dentro da pasta do projeto.

Para recriar o ambiente no macOS ou Linux, com Python 3.13 instalado:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Após preparar o ambiente e instalar as bibliotecas, ative-o novamente ao abrir um novo terminal:

```bash
source .venv/bin/activate
```

O ambiente virtual mantém as bibliotecas deste projeto separadas das bibliotecas de outros projetos. Para conferir qual Python está em uso:

```bash
python -c "import sys; print(sys.executable)"
```

O caminho exibido deve terminar em `.venv/bin/python`. Para sair do ambiente, execute `deactivate`.

### Uso no VS Code

1. Com a extensão Python da Microsoft instalada, abra a paleta de comandos com `Cmd+Shift+P` no macOS.
2. Execute `Python: Select Interpreter` e selecione o Python da pasta `.venv`. Se ele não aparecer, informe o caminho `.venv/bin/python` pela opção de inserir o caminho do interpretador.
3. Para trabalhar com arquivos `.ipynb`, use a extensão Jupyter da Microsoft e selecione esse mesmo ambiente no seletor de kernel do notebook.

### Primeiro notebook

Abra `notebooks/01_conhecimento_dados.ipynb` no VS Code, selecione o kernel da `.venv` e execute as células em ordem, de cima para baixo. O notebook já contém os resultados da execução inicial e pode ser reexecutado com o CSV no caminho indicado acima.

Ele utiliza `pandas` para ler os dados e conferir a estrutura, apresenta um dicionário inicial e registra os pontos de qualidade a investigar. Ao final, verifica o SHA-256 para confirmar a preservação do original.

A análise detalhada, incluindo estatísticas descritivas e gráficos, será realizada após a separação de treino e teste. O primeiro notebook não conclui a fase de EDA exigida no roteiro.

### Segundo notebook

Execute `notebooks/02_separacao_treino_teste.ipynb` com o kernel da `.venv`, de cima para baixo. Ele lê o CSV original, trata repetições exatas, realiza a divisão estratificada e gera `treino.csv`, `teste.csv` e `separacao.json` em `data/processed/`.

O notebook é independente do primeiro e pode ser reexecutado para recriar os conjuntos. A estratificação mantém aproximadamente as proporções das classes; a semente fixa permite reproduzir a divisão com a mesma base, ordem e ambiente. Os arquivos de dados continuam locais e excluídos do Git.

### Terceiro notebook

Execute `notebooks/03_analise_exploratoria.ipynb` com o kernel da `.venv`. É necessário ter executado o notebook 02 para gerar `treino.csv` e `separacao.json`. O notebook confere o hash do treino, executa a EDA e exporta quatro gráficos para `docs/figuras/`; os resultados também ficam incorporados ao notebook.

O CSV de teste não é lido nessa análise. Os CSVs de entrada são preservados e nenhuma limpeza ou imputação é aplicada durante a EDA.

### Quarto notebook e testes

Execute `notebooks/04_tratamento_features.ipynb` com o kernel da `.venv`. Ele importa os módulos de `src/`, apresenta os efeitos das regras e demonstra imputação, engenharia de atributos, codificação e escala com validação interna. As transformações ficam em memória; os arquivos de entrada são preservados.

Para verificar as regras de negócio e o isolamento do ajuste, execute na raiz do projeto:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Os sete testes verificam razões e divisores inválidos, tratamento das inconsistências, preservação das entradas, bloqueio do alvo nas entradas, imputação antes da nova variável, categorias desconhecidas, isolamento das estatísticas, opção sem escala e serialização com `joblib`.

### Quinto notebook e testes de modelagem

Execute `notebooks/05_modelagem_validacao.ipynb` com o kernel da `.venv`, após gerar os conjuntos no notebook 02. A execução faz 24 ajustes na validação cruzada e oito ajustes no treino completo para produzir a comparação exigida no roteiro. Os resultados agregados ficam em `docs/resultados/` e dois gráficos são exportados para `docs/figuras/`.

O teste é carregado após a gravação dos finalistas. As regras de seleção estão no código e o hash do arquivo de seleção é conferido após a avaliação de teste, garantindo que essa seleção não foi alterada durante a execução.

A modelagem possui quatro testes que verificam as oito configurações e seus pesos de classe, a seleção baseada apenas em validação, as métricas com limiar explícito e a integração das duas famílias com entradas brutas e valores ausentes.

### Sexto notebook e testes de custos

Execute `notebooks/06_avaliacao_financeira.ipynb` com o kernel da `.venv`, após gerar os conjuntos no notebook 02 e os finalistas no notebook 05. Ele faz seis ajustes para as previsões OOF, define os cortes e o candidato, registra a decisão e ajusta dois pipelines no treino completo para a avaliação final. O campeão é salvo em `models/modelo_credito.joblib`.

Pré-processamento, modelagem e custos somam **15 testes**. Os quatro testes de custos verificam contagens de FP/FN, cálculo de custo e recall, desempates, invariância ao multiplicar ambos os custos pela mesma constante e rejeição de entradas inválidas. A serialização do campeão também é verificada pela execução do notebook.

### Executar e demonstrar a API

Com o ambiente ativado e as dependências instaladas, mantenha este comando rodando no terminal:

```bash
python -m uvicorn app.main:app --reload
```

No Swagger em `http://127.0.0.1:8000/docs`:

1. Abra `GET /health`, clique em **Try it out** e em **Execute**. A resposta deve indicar `status: ok`, modelo `rf_depth_None` e corte `0.295`.
2. Abra `POST /predict`, clique em **Try it out** e mantenha o exemplo de entrada ou cole o conteúdo de um dos arquivos em `docs/exemplos/`.
3. Clique em **Execute** e confira o score, a classificação e a ação recomendada.
4. Troque `person_income` para `0` e execute novamente para observar a validação HTTP 422.

Para testar por outro terminal, a partir da raiz do projeto:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @docs/exemplos/credito_baixo_risco.json
```

Resposta do exemplo de baixo risco no artefato atual:

```json
{
  "probabilidade_inadimplencia": 0.03,
  "classe_prevista": 0,
  "classificacao_risco": "baixo",
  "acao_recomendada": "Prosseguir com a análise de crédito.",
  "threshold": 0.295,
  "modelo": "rf_depth_None"
}
```

Para parar um servidor iniciado no terminal, pressione `Ctrl+C`. Se a porta 8000 estiver ocupada, use `--port 8001` e ajuste os endereços de teste.

A suíte completa contém **26 testes**. Os 11 testes da API verificam carregamento e reutilização do modelo, corte salvo e sua fronteira inclusiva, respostas dos exemplos, validação de entradas e valores não finitos, imputação dos opcionais, categorias novas, indisponibilidade, OpenAPI/Swagger e falhas de inicialização. Instale `requirements-dev.txt` e use o comando de testes mostrado acima.

### Bibliotecas do projeto

| Bibliotecas | Finalidade |
| --- | --- |
| NumPy e pandas | Cálculos e manipulação dos dados |
| Matplotlib e seaborn | Gráficos e análise exploratória |
| scikit-learn | Pré-processamento, modelos e métricas |
| joblib | Gravação e carregamento do pipeline |
| FastAPI, Pydantic e Uvicorn | API, validação e servidor |
| ipykernel | Execução dos notebooks no VS Code |

As versões das dependências diretas estão registradas em `requirements.txt`. A pasta `.venv` é local e está excluída do Git pelo `.gitignore`.

As instruções de treinamento e geração do artefato estão nos notebooks 05 e 06. A API utiliza o artefato pronto conforme os passos acima.

## Entrega

**Prazo indicado no roteiro: 30/11/2026.**

- Repositório público no GitHub, com histórico incremental, branches por etapa, código organizado, `requirements.txt` e README completo.
- Vídeo de até sete minutos, no YouTube como não listado ou no Google Drive com acesso de leitura.
- Submissão dos dois links na atividade Projeto Avaliativo Final do AVA.

O vídeo deve abordar o problema de negócio, a análise e limpeza dos dados, a modelagem e o overfitting, a matriz de confusão e a escolha final, além de demonstrar uma requisição real à API pelo Swagger UI (`/docs`) com o servidor Uvicorn em execução.

- [Roteiro de apresentação com duração planejada de 6 minutos e 35 segundos](docs/roteiro_video.md).
- [Conferência dos critérios e passos para a entrega](docs/conferencia_entrega.md).

**Repositório para entregar:** [samuelbucco/sctec-risco-credito](https://github.com/samuelbucco/sctec-risco-credito).

**Vídeo:** pendente de gravação e publicação. Após publicar como não listado no YouTube ou com acesso de leitura no Google Drive, confira o link em uma janela sem login, registre o endereço aqui e submeta os dois links no AVA.
