# ML Canvas — Previsão de Churn em Telecomunicações

> Documento de formulação do problema de Machine Learning seguindo o framework ML Canvas.

---

## 1. Problema de Negócio

### Contexto
Uma operadora de telecomunicações está enfrentando uma taxa elevada de cancelamento de clientes (churn). A perda de clientes impacta diretamente a receita recorrente e aumenta os custos de aquisição de novos clientes para compensar o déficit.

### Problema
Identificar, com antecedência, quais clientes possuem maior probabilidade de cancelar o serviço, permitindo ações proativas de retenção.

### Hipótese
Padrões no comportamento contratual e de uso dos clientes (tipo de contrato, tempo de permanência, serviços contratados, método de pagamento) são preditivos do risco de churn.

---

## 2. Stakeholders

| Stakeholder | Papel | Interesse |
|-------------|-------|-----------|
| **Diretoria Executiva** | Decisor estratégico | Redução de churn rate e impacto no faturamento |
| **Equipe de Retenção / CRM** | Usuário direto do modelo | Lista priorizada de clientes em risco para ações de retenção |
| **Equipe de Marketing** | Planejamento de campanhas | Segmentação de clientes para ofertas personalizadas |
| **Equipe de Data Science** | Desenvolvimento e manutenção | Modelo confiável, reprodutível e monitorável |
| **Equipe de Engenharia** | Infraestrutura | API estável, latência dentro dos SLOs |

---

## 3. Dados

### Dataset
- **Nome**: Telco Customer Churn (IBM)
- **Fonte**: Kaggle / IBM Sample Datasets
- **Registros**: 7.043 clientes
- **Features**: 21 colunas (20 features + 1 target)
- **Target**: `Churn` (Yes/No) — classificação binária

### Categorias de Features

| Categoria | Features | Tipo |
|-----------|----------|------|
| **Demográficas** | gender, SeniorCitizen, Partner, Dependents | Categóricas |
| **Serviços Contratados** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Categóricas |
| **Contrato e Pagamento** | Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges | Mistas |
| **Tempo** | tenure | Numérica |
| **Identificação** | customerID | ID (removido no pré-processamento) |

### Qualidade dos Dados
- **Valores nulos**: `TotalCharges` contém 11 valores em branco (clientes com tenure=0)
- **Desbalanceamento**: ~26.5% Churn=Yes vs. ~73.5% Churn=No
- **Inconsistências**: `TotalCharges` armazenado como string no dataset original

---

## 4. Métricas

### 4.1 Métrica Técnica Principal: **AUC-ROC**

**Justificativa**: AUC-ROC é a métrica mais adequada para classificação binária com desbalanceamento moderado. Ela avalia a capacidade discriminativa do modelo independente do threshold de decisão.

### 4.2 Métricas Técnicas Secundárias

| Métrica | Propósito |
|---------|-----------|
| **F1-Score** | Equilíbrio entre precision e recall |
| **Precision** | Proporção de verdadeiros positivos entre predições positivas |
| **Recall (Sensibilidade)** | Proporção de churns reais detectados |
| **PR-AUC** | Mais informativa que ROC-AUC em cenários de alto desbalanceamento |
| **Accuracy** | Referência geral |

### 4.3 Métrica de Negócio: **Custo de Churn Evitado**

| Tipo de Erro | Consequência | Custo |
|-------------|-------------|-------|
| **TP** (churn detectado, retenção aplicada) | Economiza CAC | **+R$320** |
| **FP** (falso alarme, retenção desnecessária) | Custo da oferta | **−R$80** |
| **FN** (churn não detectado) | Perde cliente + CAC | **−R$400** |
| **TN** | Sem ação necessária | R$0 |

---

## 5. SLOs

### Modelo
| SLO | Objetivo |
|-----|----------|
| AUC-ROC mínima | ≥ 0.80 |
| Recall mínimo | ≥ 0.75 |
| F1-Score mínimo | ≥ 0.65 |
| Model freshness | Retreino a cada 90 dias ou drift detectado |

### API
| SLO | Objetivo |
|-----|----------|
| Latência P95 | ≤ 200 ms |
| Disponibilidade | ≥ 99.5% |
| Error rate (5xx) | < 0.5% |

---

## 6. Pipeline de ML

```
[Dados Brutos CSV]
    → EDA (ShowWithPandas)
    → Feature Engineering (apply_feature_engineering)
    → Pré-processamento (winsorize, median impute, one-hot)
    → SelectKBest (top 25 features)
    → 11 sklearn models + PyTorch MLP
    → Compare (≥ 6 metrics + business cost)
    → Time-budget RF tuning
    → tuned_model.pkl → MLflow registry
    → FastAPI /predict endpoint
    → Prometheus /metrics → Grafana dashboards
```

---

## 7. Considerações Éticas e Limitações

- `gender` e `SeniorCitizen` presentes — auditoria de equidade recomendada antes de produção em jurisdições com obrigações anti-discriminatórias.
- Dataset estático de uma única operadora — generalização não garantida.
- Cold start: clientes com `tenure=0` têm predições não confiáveis.
- Concept drift esperado: modelo deve ser retreinado ao menos trimestralmente.

---

*Documento criado como parte do Tech Challenge Fase 1 — ML-PosTech-FIAP*
