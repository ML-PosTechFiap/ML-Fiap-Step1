# Etapa 1 — Entendimento e Preparação

> Tech Challenge Fase 1 — ML PósTech FIAP
> Tema: **Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End**

Esta etapa cobre **formulação do problema**, **EDA completa** e **baselines** (DummyClassifier + Logistic Regression) com **tracking no MLflow**, conforme o PDF do Tech Challenge.

---

## Sumário

- [Objetivos da Etapa 1](#objetivos-da-etapa-1)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Pré-requisitos](#pré-requisitos)
- [Como executar (modo automático)](#como-executar-modo-automático)
- [Como executar (modo manual)](#como-executar-modo-manual)
- [O que cada componente faz](#o-que-cada-componente-faz)
- [Saídas e artefatos esperados](#saídas-e-artefatos-esperados)
- [Solução de problemas (FAQ)](#solução-de-problemas-faq)

---

## Objetivos da Etapa 1

Conforme o PDF do desafio, a Etapa 1 exige:

| Tarefa | Status |
|---|---|
| Preencher ML Canvas (stakeholders, métricas de negócio, SLOs) | ✅ [`docs/ML_CANVAS.md`](docs/ML_CANVAS.md) |
| EDA completa (volume, qualidade, distribuição, data readiness) | ✅ [`notebooks/01_eda_analysis.ipynb`](notebooks/01_eda_analysis.ipynb) |
| Definir métrica técnica (AUC-ROC, PR-AUC, F1) e métrica de negócio (custo de churn evitado) | ✅ ML Canvas + notebook 02 |
| Treinar baselines (DummyClassifier + LogisticRegression) | ✅ [`notebooks/02_baselines.ipynb`](notebooks/02_baselines.ipynb) |
| Registrar experimentos no MLflow (parâmetros, métricas, dataset version) | ✅ Experimento `TelcoChurn_Step1_Baselines` |

**Entregável da etapa:** notebook de EDA + baselines registrados no MLflow.

---

## Estrutura de pastas

```
Tech_Challenge/
├── pyproject.toml                       # Dependências (gerenciado pelo uv)
├── uv.lock                              # Lock file do uv
├── .python-version                      # Versão do Python (3.12)
├── data/
│   └── Telco-Customer-Churn.csv         # Dataset (deve estar presente)
└── step1/
    ├── docs/
    │   └── ML_CANVAS.md                 # ML Canvas — formulação do problema
    ├── notebooks/
    │   ├── 01_eda_analysis.ipynb        # EDA completa
    │   └── 02_baselines.ipynb           # Baselines + MLflow tracking
    ├── docker-compose.yml               # Servidor MLflow em container
    ├── run_step1.py                    # Orquestrador (sobe MLflow + executa notebooks)
    ├── run.bat                          # Wrapper para Windows
    ├── .gitignore
    └── README.md                        # Este arquivo
```

Ao executar, o script gera automaticamente:

```
step1/
├── mlruns/        # Backend store do MLflow (volumes Docker)
└── mlartifacts/   # Artefatos: modelos pickle, plots PNG, etc.
```

---

## Pré-requisitos

| Ferramenta | Versão | Por quê |
|---|---|---|
| **Python** | 3.12+ | Definido em `.python-version` |
| **uv** | 0.4+ | Gerenciador de pacotes e ambientes Python |
| **Docker Desktop** | 4.x+ | Subir o servidor MLflow em container |
| **Docker Compose** | v2 (`docker compose`) | Orquestração do container MLflow |

### Instalação do `uv`

```bash
# Via pip (qualquer plataforma)
pip install uv

# Ou via PowerShell (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Ou via curl (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> ⚠️ Em **Windows**, abra o **Docker Desktop** e aguarde-o ficar com status _"Running"_ antes de executar o script.

**Dataset**: o arquivo `Telco-Customer-Churn.csv` precisa estar em `../data/` (pasta `data/` na raiz do projeto, irmã de `step1/`). Ele já está incluído no repositório.

**Dependências**: definidas no [`pyproject.toml`](../pyproject.toml) na **raiz do projeto** (não em `step1/`). O `uv` cria um `.venv` automaticamente na primeira execução.

---

## Como executar (modo automático)

A forma recomendada é executar o orquestrador, que cuida de **tudo** automaticamente:

### Windows (CMD ou PowerShell)

```cmd
cd step1
run.bat
```

### Linux / macOS / Git Bash

```bash
cd step1
uv run python run_step1.py
```

### O que o script faz, em ordem

1. **Verifica o dataset** em `../data/Telco-Customer-Churn.csv`
2. **Verifica `pyproject.toml`** na raiz do projeto
3. **Verifica que `uv` está instalado**
4. **Sincroniza o ambiente** (`uv sync`) — cria/atualiza `.venv` conforme `pyproject.toml` + `uv.lock`
5. **Sobe o servidor MLflow** em container (`docker compose up -d mlflow`)
6. **Aguarda o MLflow ficar disponível** em `http://localhost:5000` (health check, timeout 240s)
7. **Executa os dois notebooks** em ordem usando o venv do uv:
   - `notebooks/01_eda_analysis.ipynb`
   - `notebooks/02_baselines.ipynb`
8. **Imprime um resumo** com a URL do MLflow UI

Após a conclusão, basta abrir [http://localhost:5000](http://localhost:5000) no navegador para visualizar os experimentos registrados.

### Flags disponíveis

```bash
uv run python run_step1.py --help
```

| Flag | Quando usar |
|---|---|
| `--skip-deps` | Pula `uv sync` (ambiente já sincronizado) |
| `--skip-docker` | Assume que MLflow já está rodando em :5000 |
| `--skip-notebooks` | Apenas sobe a infra (útil para abrir notebooks manualmente depois) |
| `--shutdown` | Encerra o container MLflow ao final da execução |

Exemplos:

```bash
# Re-executar apenas os notebooks (MLflow e venv já prontos)
uv run python run_step1.py --skip-deps --skip-docker

# Subir só a infra para depois trabalhar interativamente no Jupyter
uv run python run_step1.py --skip-notebooks

# Pipeline completo + cleanup ao final
uv run python run_step1.py --shutdown
```

---

## Como executar (modo manual)

Se preferir controle granular, execute as etapas manualmente:

### 1. Sincronizar dependências

A partir da **raiz do projeto** (`Tech_Challenge/`):

```bash
uv sync
```

Isso cria `.venv/` e instala tudo que está em `pyproject.toml` + `uv.lock`.

### 2. Subir o MLflow

```bash
cd step1
docker compose up -d mlflow
```

Verifique se está rodando:

```bash
docker compose ps
curl http://localhost:5000/health
```

### 3. Executar os notebooks

**Opção A — via Jupyter Lab (interativo):**

```bash
uv run jupyter lab notebooks/
```

E execute as células de `01_eda_analysis.ipynb` e `02_baselines.ipynb` na ordem.

**Opção B — via nbconvert (não-interativo):**

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_analysis.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb
```

### 4. Abrir o MLflow UI

```
http://localhost:5000
```

Procure pelo experimento **`TelcoChurn_Step1_Baselines`**.

### 5. Encerrar

```bash
docker compose down
```

---

## O que cada componente faz

### `docs/ML_CANVAS.md`

Documento de formulação do problema seguindo o framework ML Canvas. Cobre:

- **Problema de negócio** e contexto da operadora
- **Stakeholders** (Diretoria, CRM, Marketing, Data Science, Engenharia)
- **Dados**: dataset Telco Customer Churn (IBM), 7.043 registros × 21 colunas
- **Métricas técnicas**: AUC-ROC (principal), PR-AUC, F1, Precision, Recall
- **Métrica de negócio**: custo de churn evitado (CAC R$400 vs. retenção R$80)
- **Trade-off**: priorizar Recall (FN custa R$400, FP custa R$80)
- **SLOs do modelo** (AUC ≥ 0.80, Recall ≥ 0.75) e **da API** (latência P95 ≤ 200ms)
- **Vieses, limitações e cenários de falha**

### `notebooks/01_eda_analysis.ipynb`

Análise Exploratória de Dados completa, com 11 seções:

1. Carregamento e versionamento (SHA256 do arquivo)
2. Análise de **volume** (rows × cols, memória, tipos)
3. Análise de **qualidade** (nulos, duplicatas, inconsistências)
4. Estatísticas descritivas
5. Distribuição do **target** (Churn 26.5% / 73.5%)
6. Distribuição de features numéricas (tenure, MonthlyCharges, TotalCharges)
7. Distribuição de features categóricas (15 features)
8. Matriz de correlação
9. Análise de outliers (boxplots + Z-score)
10. Churn rate por segmento (Contract, InternetService, etc.)
11. **Data Readiness Assessment** (checklist de prontidão para ML)

### `notebooks/02_baselines.ipynb`

Treinamento dos baselines com tracking completo no MLflow:

- **Pré-processamento**: drop `customerID`, fix `TotalCharges`, encode target, one-hot nas categóricas
- **Split estratificado** 80/20 com `random_state=42`
- **Validação cruzada** estratificada (5 folds)
- **Baseline 1** — `DummyClassifier(strategy='most_frequent')`: piso absoluto
- **Baseline 2** — `LogisticRegression(solver='saga', max_iter=5000)`: baseline competitivo
- **Métricas registradas no MLflow** (por run):
  - `accuracy`, `f1_score`, `precision`, `recall`, `auc_roc`, `pr_auc`
  - `business_cost_avoided`, `true_positives`, `false_positives`, `false_negatives`, `true_negatives`
  - `cv_accuracy_mean/std`, `cv_auc_roc_mean/std`
- **Parâmetros registrados**: nome do dataset, versão (SHA256), hiperparâmetros
- **Artefatos**: modelo serializado, confusion matrix, ROC curve, PR curve
- **Análise de custo**: detalhamento de TP/FP/FN/TN em valor monetário

### `docker-compose.yml`

Serviço único: container Python 3.12-slim que instala MLflow 3.10.1 (alinhado ao cliente em `pyproject.toml`) e expõe o servidor em `:5000`. Volumes persistentes:

- `./mlruns` → backend store (metadados de experimentos)
- `./mlartifacts` → artefatos (modelos, plots, etc.)

### `pyproject.toml` (na raiz)

Single source of truth das dependências, gerenciado pelo `uv`:

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `mlflow`, `jupyter`, `ipykernel`, `scipy`
- + bibliotecas para etapas futuras: `fastapi`, `pydantic`, `pytorch` (a adicionar), `lightgbm`, `xgboost`, `evidently`, `fairlearn`, `slowapi`

### `run_step1.py`

Orquestrador Python (cross-platform) que executa todo o pipeline da Etapa 1 sem intervenção manual. Usa apenas a stdlib (`subprocess`, `urllib`, `argparse`, `logging`) e delega gerenciamento de ambiente para o `uv`.

### `run.bat`

Wrapper conveniente para Windows: detecta o `uv` no PATH, muda para o diretório correto e repassa todos os argumentos para `run_step1.py` via `uv run python`.

---

## Saídas e artefatos esperados

### Notebooks executados

Os arquivos `.ipynb` em `notebooks/` são reescritos in-place com os outputs preenchidos (gráficos, tabelas, prints).

### MLflow UI (http://localhost:5000)

Experimento **`TelcoChurn_Step1_Baselines`** com 2 runs:

| Run | AUC-ROC | F1 | Recall | Custo Evitado (R$) |
|---|---|---|---|---|
| `Baseline_DummyClassifier`  | ~0.50 | 0.00 | 0.00 | R$ 0 |
| `Baseline_LogisticRegression` | ~0.84 | ~0.59 | ~0.55 | positivo |

Cada run contém:

- **Parameters**: dataset_name, dataset_version (SHA256), model_type, hiperparâmetros, random_seed
- **Metrics**: todas as métricas técnicas + métrica de negócio + CV scores
- **Artifacts**:
  - `model/` — sklearn pickle
  - `plots/confusion_matrix.png`
  - `plots/roc_curve.png` (apenas LogReg)
  - `plots/precision_recall_curve.png` (apenas LogReg)

### Pastas geradas

```
step1/
├── mlruns/         # criada pelo MLflow (gitignored)
└── mlartifacts/    # criada pelo MLflow (gitignored)
```

---

## Solução de problemas (FAQ)

### `uv não está instalado`

Instale com `pip install uv` ou siga as instruções em [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).

### `Docker não está instalado` ou `docker compose: command not found`

Instale o **Docker Desktop** (Windows/macOS) ou Docker Engine + Compose v2 (Linux). Em Windows, abra o Docker Desktop e aguarde "Engine running".

### `MLflow não ficou disponível em 240s`

- Verifique se o container está de pé: `docker compose ps`
- Veja os logs: `docker compose logs mlflow`
- A primeira execução demora ~30-60s porque o container precisa baixar a imagem e instalar `mlflow==3.10.1`. Re-execute o script.

### `Dataset não encontrado em ../data/Telco-Customer-Churn.csv`

O dataset deve estar em `Tech_Challenge/data/Telco-Customer-Churn.csv` (já incluído no repositório). Confirme que executou o script de dentro da pasta `step1/`.

### Notebooks falham ao executar (`ModuleNotFoundError`)

Rode `uv sync` na raiz do projeto, ou execute o orquestrador sem `--skip-deps`.

### Quero re-rodar do zero limpando todos os experimentos MLflow

```bash
docker compose down
rm -rf mlruns mlartifacts          # Linux/Mac
rmdir /s /q mlruns mlartifacts     # Windows
uv run python run_step1.py
```

### Porta 5000 já está em uso

Edite `docker-compose.yml` e troque `"5000:5000"` por `"5050:5000"` (ou outra porta livre). Também atualize a variável `MLFLOW_URI` no notebook `02_baselines.ipynb` (cell 11) para `http://localhost:5050`.

### Posso usar um servidor MLflow já existente?

Sim. Use `--skip-docker` e ajuste a `MLFLOW_URI` no notebook 02.

### Cliente MLflow incompatível com servidor

Se você ver erros como `MlflowException: API request failed` ou problemas de versão, verifique:
- Cliente: `uv run python -c "import mlflow; print(mlflow.__version__)"` deve retornar `3.10.1` ou superior
- Servidor: o `docker-compose.yml` instala `mlflow==3.10.1` (alinhado ao cliente)

### MLflow 3.x bind workaround (não mexa sem entender)

O `docker-compose.yml` usa um **workaround** para um bug do MLflow 3.x:

- O wrapper `mlflow server --host 0.0.0.0` invoca internamente
  `uvicorn --host 127.0.0.1` (a flag externa é ignorada).
- Resultado: o servidor escuta apenas em loopback dentro do container e o
  port-forward do Docker não funciona (`Empty reply from server`).

**Workaround aplicado** em [docker-compose.yml](docker-compose.yml):

```yaml
command:
  - bash
  - -c
  - |
    set -e
    pip install --quiet --root-user-action=ignore 'mlflow==3.10.1'
    exec mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --workers 1 \
      --allowed-hosts '*' \
      --backend-store-uri /mlflow/mlruns \
      --default-artifact-root /mlflow/mlartifacts \
      --uvicorn-opts '--host 0.0.0.0'
```

Pontos críticos:
- `--uvicorn-opts '--host 0.0.0.0'` — força o uvicorn a bindar em todas as interfaces (o último `--host` vence)
- `--workers 1` — múltiplos workers no Docker Desktop causam instabilidade
- `--allowed-hosts '*'` — desabilita proteção DNS rebinding (apropriado em dev local)
- Formato **array** de `command:` (e não `>` folded scalar) — para garantir que o `bash -c` receba o script como string única, não como múltiplos argumentos

---

## Próximas etapas

A Etapa 1 estabelece o **piso de performance** (DummyClassifier) e o **baseline competitivo** (LogisticRegression com AUC ≈ 0.84). As próximas etapas devem **superar** esses números:

- **Etapa 2** — Treinar MLP em PyTorch com early stopping, comparar com baselines, registrar no MLflow.
- **Etapa 3** — Refatoração em `src/`, pipeline reprodutível, API FastAPI, testes pytest, Makefile.
- **Etapa 4** — Model Card completo, plano de monitoramento, README final, vídeo STAR.

---

*Tech Challenge Fase 1 — ML PósTech FIAP*
