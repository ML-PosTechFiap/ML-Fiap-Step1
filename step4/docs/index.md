# Telco Churn Prediction

**Pipeline end-to-end de Machine Learning — Tech Challenge Fase 1 · FIAP POS-TECH**

---

## Visão Geral

Previsão de churn (evasão de clientes) em tempo real para uma operadora de telecomunicações.  
O modelo é servido via API REST com stack completo de observabilidade.

| Componente | Tecnologia |
|------------|-----------|
| Treinamento | scikit-learn · PyTorch · MLflow |
| API | FastAPI · Pydantic · slowapi |
| Monitoramento | Prometheus · Grafana · drift detection |
| Deploy | Docker Compose · Railway |
| CI | GitHub Actions (lint → test → build) |

---

## Resultados do Modelo

Comparação de 12 modelos (11 sklearn + MLP PyTorch) no conjunto de teste:

| Modelo | ROC-AUC | F1 | Recall | Accuracy |
|--------|---------|----|--------|----------|
| **Gradient Boosting** *(vencedor)* | **0.843** | **0.588** | 0.529 | **0.803** |
| AdaBoost | 0.843 | 0.580 | 0.529 | 0.796 |
| Logistic Regression | 0.840 | 0.594 | 0.545 | 0.802 |
| LightGBM | 0.834 | 0.577 | 0.529 | 0.794 |
| MLP (PyTorch) | 0.828 | 0.503 | 0.404 | 0.789 |
| XGBoost | 0.821 | 0.574 | 0.524 | 0.793 |
| Random Forest | 0.816 | 0.556 | 0.505 | 0.786 |

> O Gradient Boosting foi tunado com random search de orçamento de tempo (5 min), atingindo **accuracy ≈ 0.810**.  
> No threshold ótimo de negócio (~0.14), a **Regressão Logística** minimiza o custo total (R$48.880).

---

## Progressão do Projeto

| Step | Foco | Entrega principal |
|------|------|-------------------|
| 1 | EDA + ML Canvas + Baselines | Notebook + MLflow tracking |
| 2 | Pipeline sklearn + feature engineering | Módulo `src/` + MLflow |
| 3 | MLP PyTorch + 11 modelos + FastAPI | API REST + hot-reload |
| **4** | **Monitoramento + Testes + Documentação** | **Prometheus · Grafana · Model Card · deploy** |

---

## Quick Start (Docker)

```bash
docker compose up --build
```

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| API docs (Swagger) | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## Estrutura do Projeto

O Tech Challenge Fase 1 é composto por 4 steps evolutivos, cada um construindo sobre o anterior:

```
Tech_Challenge/
├── data/
│   └── Telco-Customer-Churn.csv     ← Dataset compartilhado entre todos os steps
│
├── step1/                           ← EDA + Baselines
│   ├── notebooks/
│   │   ├── 01_eda_analysis.ipynb    ← Análise exploratória completa (11 seções)
│   │   └── 02_baselines.ipynb       ← Modelos baseline + MLflow tracking
│   └── docs/ML_CANVAS.md
│
├── step2/                           ← Pipeline sklearn modularizado
│   ├── src/
│   │   ├── eda/                     ← ShowWithPandas
│   │   ├── features/                ← Feature engineering
│   │   ├── train/                   ← MLTrainer (sklearn)
│   │   └── main.py
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── step3/                           ← MLP PyTorch + FastAPI
│   ├── src/
│   │   ├── api/                     ← FastAPI + hot-reload
│   │   ├── train/                   ← 11 modelos sklearn + MLP PyTorch
│   │   └── evaluation/              ← Comparação de modelos + custo de negócio
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── step4/                           ← Entrega final: monitoramento + documentação
    ├── docs/                        ← Esta documentação (MkDocs)
    ├── monitoring/                  ← prometheus.yml + Grafana dashboards
    ├── src/
    │   ├── api/                     ← FastAPI + Prometheus + drift detection
    │   ├── train/                   ← 11 modelos sklearn + MLP PyTorch + tuning
    │   ├── eda/                     ← ShowWithPandas
    │   ├── features/                ← Feature engineering
    │   └── evaluation/              ← Comparação de modelos + custo de negócio
    ├── tests/                       ← smoke · schema · API (24 testes)
    ├── models/                      ← tuned_model.pkl + reference_stats.json
    ├── Dockerfile
    ├── docker-compose.yml
    └── Makefile
```

---

## Documentação

| Documento | Descrição |
|-----------|-----------|
| [ML Canvas](ML_CANVAS.md) | Formulação do problema, stakeholders, métricas e SLOs |
| [Model Card](MODEL_CARD.md) | Performance, limitações, vieses e modos de falha |
| [Arquitetura de Deploy](DEPLOY_ARCHITECTURE.md) | Decisões de infraestrutura e diagrama de componentes |
| [Plano de Monitoramento](MONITORING_PLAN.md) | Alertas, playbook de incidentes e estratégia de retreino |
| [Roteiro do Vídeo](roteiro.md) | Script STAR para apresentação de 5 minutos |
