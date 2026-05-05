# Guia de Gravação do Vídeo - Método STAR (5 Minutos)

Este é um roteiro estruturado para ajudar o seu grupo a gravar o vídeo de 5 minutos exigido pelo Tech Challenge, utilizando o *Método STAR* (Situation, Task, Action, Result). 

O tempo alvo é de *5 minutos* no total (aproximadamente 600 a 700 palavras se lido em um ritmo normal). Divida as falas entre os integrantes ou escolha um porta-voz.

---

## ⏱️ Distribuição do Tempo Sugerida
- *Introdução:* ~20 segundos
- *S - Situation (Situação):* ~50 segundos
- *T - Task (Tarefa):* ~30 segundos
- *A - Action (Ação):* ~2 min 20 seg (5 bullets — cortar monitoramento se apertar)
- *R - Result (Resultado):* ~1 minuto

---

## 🎬 Roteiro Prático (Script)

### 1. Introdução (0:00 - 0:20)
*Objetivo:* Apresentar o grupo e o tema rapidamente.
> "Olá, nós somos o grupo [Nome/Número do Grupo] e hoje vamos apresentar o nosso Tech Challenge da Fase 1 — entrega final. Desenvolvemos um pipeline end-to-end de Machine Learning para a previsão de Churn (evasão de clientes) em uma operadora de telecomunicações."

### 2. S - Situation / Situação (0:20 - 1:20)
*Objetivo:* Qual o problema de negócio e o contexto do dataset?
> "O nosso contexto de negócio gira em torno de uma operadora de telecomunicações que está perdendo clientes em um ritmo muito acelerado. O 'Churn', que é essa taxa de cancelamento, gera uma enorme perda de receita, e a diretoria precisava de uma solução baseada em dados para agir preventivamente."
> 
> "Para atacar esse problema, utilizamos o dataset 'Telco Customer Churn' da IBM. Ele contém informações cruciais sobre os clientes, como serviços contratados, dados demográficos, tempo de permanência (tenure) e os encargos mensais. A nossa variável alvo (target) era identificar quem cancelaria ou não o serviço no próximo mês."

### 3. T - Task / Tarefa (1:20 - 2:00)
*Objetivo:* Qual a tarefa do grupo e os objetivos técnicos?
> "Nossa tarefa central foi construir um projeto profissional de ponta a ponta: saindo da análise exploratória de dados até colocar um modelo em produção com monitoramento completo."
> 
> "Os objetivos técnicos eram: desenvolver e comparar uma Rede Neural MLP em PyTorch com 11 baselines sklearn; implementar uma API de produção com FastAPI; e, principalmente, adicionar um stack completo de observabilidade — métricas com Prometheus, dashboards em Grafana e detecção de data drift em tempo real."

### 4. A - Action / Ação (2:00 - 4:00)
*Objetivo:* Quais decisões técnicas foram tomadas (arquitetura, features, modelo, métricas)?
(Nota: Essa é a parte mais longa. Pode ser ideal mostrar a tela com a arquitetura ou o MLflow rodando durante essa fala).

> "Para solucionar o desafio, dividimos o projeto em 4 grandes etapas:"
> 
> - *Preparação e EDA:* "Fizemos uma análise exploratória completa, tratamos valores nulos e aplicamos técnicas de Feature Engineering como One-Hot Encoding e tratamento de outliers com Winsorize. Avaliamos a viabilidade dos dados e definimos nossas métricas de sucesso: ROC-AUC, Recall (para focar nos falsos negativos) e a métrica de negócio baseada no custo do churn."
> - *Modelagem e Baselines:* "Treinamos pipelines robustos em Scikit-Learn com 11 modelos diferentes, incluindo Dummy Classifier e Logistic Regression como baselines."
> - *Rede Neural (PyTorch):* "Construímos nossa arquitetura principal: uma MLP (Multi-Layer Perceptron) com PyTorch, utilizando camadas ocultas, função de perda BCEWithLogitsLoss, Dropout para evitar overfitting e Early Stopping no loop de treinamento."
> - *Engenharia e API (Arquitetura):* "Refatoramos todo o código adotando modularização (pasta src/). Utilizamos o pyproject.toml para gerenciar dependências, aplicamos linting com ruff e implementamos testes automatizados (unitários, de esquema e API) com pytest. Todo o rastreamento foi feito com o MLflow. Servimos o modelo em uma API REST com FastAPI e Pydantic, rodando tudo em containers via Docker Compose, com deploy no Railway."
> - *Monitoramento em Produção (Step 4):* "Instrumentamos a API com Prometheus — coletando latência, taxa de erro, volume de predições e churn rate em tempo real. Criamos dashboards no Grafana conectados a essas métricas. Implementamos detecção de data drift: a API compara as distribuições dos dados recebidos em produção com as estatísticas do treino, expondo um endpoint GET /drift que alerta quando a distribuição deriva. O model hot-reload permite atualizar o modelo sem derrubar o serviço."

### 5. R - Result / Resultado (4:00 - 5:00)
*Objetivo:* Quais os resultados obtidos e as lições aprendidas?
> "Como resultado, treinamos e comparamos 12 modelos — 11 sklearn mais a MLP PyTorch — usando 5 métricas simultâneas: ROC-AUC, PR-AUC, Recall, F1 e custo de negócio. O **Gradient Boosting** liderou a comparação com **ROC-AUC de 0.843**, F1 de 0.588 e accuracy de 0.803. Ele foi então tunado com random search de orçamento de tempo de 5 minutos, alcançando accuracy final de **0.810**."
>
> "Vale destacar uma descoberta importante: a MLP PyTorch teve o ROC-AUC de 0.828 — competitivo — mas o menor recall entre os modelos úteis (0.404), confirmando que métricas únicas enganam. A análise de custo de negócio mostrou que, no threshold ótimo de ~0.14, a Regressão Logística minimizou a perda total (R$48.880), provando que o melhor modelo estatístico não é sempre o melhor para o negócio."
>
> "Como entrega final, temos além do modelo: a API em produção no Railway com Prometheus, Grafana e drift detection ativo. O Model Card documenta limitações, vieses e limiares de retreinamento. A maior lição foi que MLOps completo — rastreamento, monitoramento, observabilidade e deploy reprodutível — é tão crítico quanto a acurácia do modelo."

---

## 🧠 Como o Código Funciona: Decisões Técnicas de Treinamento e Deploy

Esta seção descreve, em detalhe, as decisões que o pipeline toma automaticamente — útil tanto para a gravação quanto para quem quiser entender o projeto a fundo.

---

### 1. Seleção de Features (`train_models.py`)

Antes de qualquer treinamento, o pipeline aplica `SelectKBest` com a função `f_classif` (ANOVA F-score) para selecionar as **25 features mais relevantes** do dataset pré-processado (que possui 34 colunas após o one-hot encoding). O número `k=25` é fixo, mas limitado ao total de colunas disponíveis:

```python
k = min(25, X_train.shape[1])
X_train_sel, X_test_sel, selector = trainer.feature_selection(X_train, X_test, y_train, k=k)
```

Essa seleção é aplicada de forma consistente em treino e teste para evitar data leakage.

---

### 2. Treinamento em Paralelo: 11 Modelos + MLP (`train_models.py` + `mlp_trainer.py`)

O pipeline treina **11 modelos Scikit-Learn** com configurações padrão (sem tuning) em sequência. Para cada modelo:
- Executa validação cruzada com 5 folds estratificados (`StratifiedKFold`) e métrica `roc_auc`.
- Treina no conjunto completo de treino e avalia no conjunto de teste.
- Registra automaticamente todos os parâmetros e métricas no **MLflow** em runs separadas e aninhadas.

Após os modelos sklearn, a **MLP PyTorch** é treinada com:
- Arquitetura fixa: `[128, 64, 32]` neurônios, dropout de 30%.
- Otimizador Adam com `lr=0.001`, batch size 64, até 200 épocas.
- **Early stopping** com paciência de 15 épocas — o treinamento para automaticamente ao detectar estagnação na `val_loss`.

---

### 3. Comparação e Identificação do Melhor Modelo (`train_models.py:85-88`)

Ao final do loop de treinamento, o código identifica o modelo com maior **test accuracy**:

```python
if acc > best_score:
    best_score, best_model, best_name = acc, model, name

logger.info(f"Best model: {best_name} (accuracy={best_score:.4f})")
```

Na última execução confirmada (run MLflow `be59b925`), o **Gradient Boosting** venceu com os seguintes resultados no conjunto de teste:

| Métrica | Valor |
|---------|-------|
| ROC-AUC | **0.8432** |
| PR-AUC | 0.6619 |
| F1 (churn) | 0.5875 |
| Recall (churn) | 0.5294 |
| Precision (churn) | 0.6600 |
| Accuracy | **0.8027** |
| TP / FP / FN / TN | 198 / 102 / 176 / 933 |

Para referência, a **MLP PyTorch** ficou com AUC=0.828 mas recall=0.404 — o menor entre os modelos não-triviais, reforçando que AUC alto não garante recall adequado para o problema de churn.

A comparação completa entre todos os modelos (incluindo a MLP) usa 5 métricas simultâneas: `roc_auc`, `pr_auc`, `recall`, `f1` e **custo de negócio** (calculado com base nos custos diferenciados de Falsos Positivos e Falsos Negativos). No **threshold ótimo de negócio (~0.14)**, a **Regressão Logística** minimizou o custo total com R$48.880 de perda — abaixo do Gradient Boosting (R$50.480) — mostrando que o vencedor estatístico pode não ser o ótimo de negócio.

---

### 4. Decisão de Qual Modelo Tunar (`main.py` + `tuning.py`)

O pipeline faz tuning no **modelo vencedor da comparação** — no nosso caso, o Gradient Boosting. A chamada em `main.py` passa explicitamente o melhor modelo sklearn:

```python
tuned_model, tuning_metrics = time_budget_search(
    best_sklearn_model, X_train_sel, y_train, X_test_sel, y_test,
    time_limit_minutes=5,
    acc_target=0.90,
    model_name=best_sklearn_name,
)
```

Dentro de `tuning.py`, a função `time_budget_search` usa o tipo do modelo recebido para selecionar automaticamente o espaço de hiperparâmetros correto via `_PARAM_DISTRIBUTIONS`:

```python
model_class_name = type(model).__name__
param_distributions = _PARAM_DISTRIBUTIONS.get(model_class_name)
# Se não houver distribuição definida, faz fallback para RandomForest
```

Para o **GradientBoostingClassifier**, o espaço inclui `n_estimators`, `max_depth`, `learning_rate`, `subsample` e `min_samples_split`. O pipeline constrói e avalia dinamicamente cada combinação com `clone(model)`, garantindo reprodutibilidade.

---

### 5. Estratégia de Tuning: Random Search com Orçamento de Tempo (`tuning.py`)

Em vez de GridSearchCV (que explora um grid fixo), o pipeline usa `ParameterSampler` com **busca aleatória com orçamento de tempo**:

- Define um prazo de **5 minutos** (configurável).
- Amostra configurações aleatórias de um espaço contínuo (ex: `n_estimators` entre 50 e 300).
- Para cada candidato: executa cross-validation com 5 folds, treina no conjunto completo e mede accuracy no teste.
- Salva apenas o melhor encontrado até o momento.
- Interrompe se atingir o tempo limite **ou** se o `acc_target` (0.90) for alcançado.

```python
if test_acc >= acc_target:
    logger.info(f"Target accuracy {acc_target} reached — stopping early.")
    break
```

Na execução do Step 4, o algoritmo rodou durante **5 minutos** tunando o Gradient Boosting (baseline `accuracy=0.8027`) e atingiu `accuracy≈0.810` — melhora consistente com o ganho esperado do espaço de busca do GB. O espaço foi bem explorado e o ganho marginal tende a estabilizar após as primeiras dezenas de iterações.

---

### 6. Deploy: Como o Modelo Vai para Produção (`main.py` + `api/`)

Após o tuning, o pipeline executa três ações de persistência:

1. **Salva o artefato local** em `/models/tuned_model.pkl` via `joblib`.
2. **Registra no MLflow Model Registry** com nome dinâmico baseado no vencedor (ex: `TelcoChurn_Tuned_GradientBoosting_Model`), criando a versão 1.
3. **Salva estatísticas de referência** (`reference_stats.json`) com as distribuições do dado de treino — usadas pela API para detecção de **data drift** em produção.

A API (`api/main.py`) carrega o modelo do `.pkl` na inicialização e expõe dois endpoints principais:
- `POST /predict` — recebe os dados do cliente em JSON, valida com **Pydantic**, aplica a mesma feature engineering do treino e retorna a probabilidade de churn.
- `GET /drift` — compara a distribuição dos dados recebidos desde o início do serviço com as estatísticas de referência, alertando para desvios significativos.

---


## 💡 Dicas Visuais para a Gravação
Para não ficar apenas "falando para a câmera" (talking head), enriqueça o vídeo intercalando sua fala com as seguintes imagens (Screen Recording):
1. *[0:40] Dataset/EDA:* Mostre rapidamente um gráfico da EDA gerado pelo código.
2. *[2:30] MLflow:* Mostre a interface do MLflow com as dezenas de runs registradas e gráficos de perda.
3. *[3:20] Código PyTorch/API:* Mostre um trecho limpo do modelo PyTorch e o arquivo da API FastAPI.
4. *[4:10] Avaliação:* Mostre a tabela de comparação de modelos ou os gráficos de Curva ROC/PR e Análise de Custo Trade-off.
5. *[4:30] Swagger FastAPI:* Mostre o /docs da API em funcionamento, realizando um .predict().