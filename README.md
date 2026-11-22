# Previsão de risco de inadimplência em operações de crédito

Projeto avaliativo do curso **Integração de IA com Dados**, da etapa **Aperfeiçoar do programa SCTec**.

**Status:** estrutura e ambiente preparados; base original obtida e conferida. A análise exploratória e o treinamento ainda não foram realizados.

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

Os tipos e o dicionário completo dos dados serão documentados na próxima etapa. Esta conferência verifica o arquivo recebido; a avaliação da qualidade dos dados será feita na análise exploratória.

### Atributo obrigatório

Criar uma variável numérica que compare o valor do empréstimo à renda anual:

```text
comprometimento_renda = (loan_amnt / person_income) × 100
```

Esse indicador representa a razão percentual entre o valor total do empréstimo e a renda anual, e não o comprometimento mensal com parcelas.

O cálculo será implementado em uma função reutilizável em `src/features.py`, com tratamento de valores ausentes e proteção contra divisão por zero. A mesma regra será aplicada no treinamento e na API.

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
- [ ] Inspecionar os dados, verificar duplicatas e separar treino e teste.
- [ ] Fazer a análise exploratória e documentar os achados.
- [ ] Implementar o tratamento dos dados e a nova variável.
- [ ] Construir os pipelines e comparar os modelos.
- [ ] Avaliar overfitting, custos dos erros e pontos de corte.
- [ ] Salvar o pipeline e o ponto de corte escolhidos.
- [ ] Implementar e verificar a API.
- [ ] Completar a documentação e conferir a execução do projeto.
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
├── models/            # Pipeline e ponto de corte salvos
├── app/               # API FastAPI e contratos Pydantic
├── README.md          # Apresentação e guia do projeto
├── requirements.txt   # Dependências Python com versões registradas
└── .gitignore         # Arquivos excluídos do versionamento
```

Os arquivos vazios `.gitkeep` permitem que o Git registre as pastas enquanto elas ainda não têm conteúdo. Eles podem ser removidos quando as pastas receberem os arquivos do projeto.

O `.gitignore` exclui do versionamento o ambiente virtual, caches do Python, checkpoints dos notebooks, logs, arquivos locais do macOS e arquivos `.env` com credenciais. As bases em `data/raw/` e `data/processed/` também ficam locais por padrão; a origem e o procedimento para obtê-las serão documentados para permitir a reprodução do projeto. Os artefatos finais em `models/` poderão ser versionados para acompanhar a entrega exigida no roteiro.

## Controle de versão

O projeto utiliza Git para registrar sua evolução. A branch `main` guarda a versão consolidada; cada etapa será desenvolvida em uma branch própria, começando por `feat/eda-limpeza`.

Os commits devem descrever mudanças concretas, por exemplo:

```text
chore: estrutura projeto e configura ambiente Python
feat: adiciona analise exploratoria de credito
fix: protege calculo de comprometimento contra divisao por zero
```

Repositório público: [samuelbucco/sctec-risco-credito](https://github.com/samuelbucco/sctec-risco-credito).

A conexão remota `origin` aponta para `https://github.com/samuelbucco/sctec-risco-credito.git`. O envio inicial foi concluído e a branch local `main` acompanha `origin/main` no GitHub.

## Resultados e decisões a documentar

Estas informações serão preenchidas com evidências durante o desenvolvimento:

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

Neste computador, o ambiente `.venv` e as bibliotecas já foram preparados. Ao abrir um novo terminal, basta ativá-lo:

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

As instruções de treinamento e inicialização da API serão adicionadas após a implementação e verificação desses passos.

## Entrega

**Prazo indicado no roteiro: 30/11/2026.**

- Repositório público no GitHub, com histórico incremental, branches por etapa, código organizado, `requirements.txt` e README completo.
- Vídeo de até sete minutos, no YouTube como não listado ou no Google Drive com acesso de leitura.
- Submissão dos dois links na atividade Projeto Avaliativo Final do AVA.

O vídeo deve abordar o problema de negócio, a análise e limpeza dos dados, a modelagem e o overfitting, a matriz de confusão e a escolha final, além de demonstrar uma requisição real à API pelo Swagger UI (`/docs`) com o servidor Uvicorn em execução.
