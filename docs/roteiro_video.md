# Roteiro de apresentação do projeto

Apresentação de Samuel Bucco para o projeto de risco de crédito do curso Integração de IA com Dados. A duração planejada é de **6 minutos e 35 segundos**, com 25 segundos de margem até o limite de sete minutos. Use as falas como base para explicar o que compreendeu e ensaie com cronômetro.

## Preparar a gravação

Ative a `.venv` na raiz do projeto e inicie o servidor:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --reload
```

Se já houver uma instância na porta 8000, use o servidor existente ou encerre aquela instância antes de iniciar outra. Deixe o terminal com a mensagem de inicialização visível e abra `http://127.0.0.1:8000/docs`.

Mantenha abertas estas telas, com fonte legível:

- README no contexto de negócio e no resumo dos resultados.
- Notebook 03 nos gráficos e no diagnóstico dos nulos.
- `src/features.py` e `src/preprocessing.py`.
- Notebook 05 no gráfico de overfitting e na tabela de comparação.
- Notebook 06 nas matrizes de confusão e nos custos.
- Swagger UI e os dois JSONs em `docs/exemplos/`.

Os notebooks já contêm os resultados executados. Mostre o código e os resultados salvos; a requisição à API será executada ao vivo. Faça um teste curto de gravação para conferir voz, legibilidade e captura da janela do navegador.

## Sequência de apresentação

| Intervalo | Assunto | Tela principal |
| --- | --- | --- |
| 0:00 a 0:35 | Problema de negócio | README |
| 0:35 a 1:35 | Dados, EDA e limpeza | Notebook 03 |
| 1:35 a 2:15 | Nova variável e prevenção de vazamento | Módulos de `src/` |
| 2:15 a 3:25 | Modelos e overfitting | Notebook 05 |
| 3:25 a 4:35 | Custos, matrizes e veredito | Notebook 06 |
| 4:35 a 6:20 | Demonstração da API | Terminal e Swagger |
| 6:20 a 6:35 | Encerramento | README ou GitHub |

### Problema de negócio

Mostre o objetivo e a definição de `loan_status`.

> Sou Samuel Bucco e este é meu projeto de integração de IA com dados, com o tema risco de crédito. O objetivo é estimar inadimplência e disponibilizar a previsão em uma API. A classe um representa inadimplentes e a classe zero, adimplentes. O problema de negócio envolve dois erros: sinalizar um bom pagador como risco e deixar passar alguém que ficará inadimplente. Esses erros têm custos diferentes.

### Dados e limpeza

Mostre a distribuição do alvo, os nulos e os boxplots.

> A base original tem 32.581 registros e 12 colunas. Tratei 165 repetições exatas, mantendo o original preservado. Separei 20% para teste com estratificação e semente fixa. A análise detalhada foi feita nos 25.932 registros de treino, dos quais 21,87% são inadimplentes.
>
> No treino encontrei 727 ausências no tempo de emprego e 2.488 na taxa de juros. A assimetria do tempo de emprego fundamentou o uso da mediana; a taxa é mais simétrica e usa média. Cinco idades e dois tempos de emprego foram marcados como inconsistentes. Mantive extremos financeiros positivos, porque um valor raro não comprova erro.

### Nova variável e prevenção de vazamento

Mostre a função de cálculo e a ordem das etapas do pré-processador.

> A nova variável é o valor do empréstimo dividido pela renda anual, multiplicado por cem. Ela é calculada depois da imputação dos componentes e tem proteção contra divisão por zero. Representa a relação entre empréstimo total e renda anual, não a parcela mensal.
>
> As regras estão em funções reutilizáveis. O pipeline faz tratamento, imputação, cálculo, codificação e padronização quando necessária. Durante a validação, cada partição de treino aprende suas próprias estatísticas. A API carrega esse mesmo pipeline salvo.

### Modelagem e overfitting

Mostre as oito configurações e o gráfico de treino e validação.

> Comparei Regressão Logística, com quatro valores de C, e Random Forest, com profundidades três, seis, dez e sem limite. Usei pesos de classe para o desbalanceamento e validação estratificada em três partições. Escolhi um finalista por família pela maior ROC-AUC média de validação, antes de consultar o teste.
>
> A logística ficou próxima de 0,873 na validação, com pequena diferença para o ajuste. A floresta sem limite atingiu um no ajuste e aproximadamente 0,928 na validação. Essa diferença evidencia overfitting, que continua sendo uma limitação. Mesmo assim, ela obteve a maior média de validação no conjunto de configurações testado. Os finalistas foram a logística com C igual a dez e a floresta sem limite.

### Matriz de confusão e decisão financeira

Mostre as matrizes da floresta e a comparação dos custos.

> Como a base não informa perdas reais, simulei custo de uma unidade para falso positivo e cinco para falso negativo. Comparei os dois finalistas e os cortes usando previsões fora da partição de ajuste, apenas no treino. A floresta venceu com corte de 0,295.
>
> No teste, os falsos negativos caíram de 351 para 250 e o recall subiu de 75,25% para 82,37%. Em troca, os falsos positivos subiram de 118 para 590. O custo simulado caiu de 1.873 para 1.840 unidades, uma redução pequena de 1,76%. A decisão vale para as hipóteses declaradas; não é lucro real garantido. Os scores também não passaram por calibração probabilística.

### Demonstração ao vivo

Mostre primeiro o terminal com Uvicorn em execução. Depois, execute as requisições no Swagger:

1. Abra `GET /health`, clique em **Try it out** e em **Execute**. Mostre `status: ok`, configuração `rf_depth_None` e corte `0.295`.
2. Abra `POST /predict` e clique em **Try it out**. Execute o exemplo padrão de baixo risco. Mostre score `0.03`, classe `0`, risco `baixo` e a ação recomendada.
3. Cole o conteúdo de `docs/exemplos/credito_alto_risco.json` e execute. Mostre score `0.97`, classe `1`, risco `elevado` e a recomendação de revisão.

> O servidor carregou o artefato na inicialização. Esta consulta confirma que o modelo está disponível. Agora envio um registro e recebo o score, a classificação e a ação recomendada. Neste outro exemplo, o score ultrapassa o corte salvo e a resposta indica risco elevado. A API aplica o pré-processamento do treinamento e não treina durante a requisição.

Se estiver dentro do tempo, altere a renda para zero e mostre o HTTP 422. Essa demonstração é opcional; priorize a previsão real e os cinco tópicos exigidos.

### Encerramento

> O repositório reúne os notebooks, as funções, o modelo salvo, os testes e as instruções de execução. A solução conclui o ciclo dos dados até uma API funcional, com as decisões e limitações documentadas.

## Conferir antes de publicar

- Vídeo com no máximo sete minutos, voz compreensível e código legível.
- Cinco tópicos do professor cobertos, incluindo Uvicorn e requisição real no Swagger.
- Custos apresentados como simulados e diferença entre treino e teste explicada.
- Upload no YouTube como não listado ou no Google Drive com acesso de leitura.
- Link aberto com sucesso em uma janela sem login.

Depois da publicação, registre o link no README e envie o endereço do repositório e o do vídeo na atividade Projeto Avaliativo Final do AVA.
