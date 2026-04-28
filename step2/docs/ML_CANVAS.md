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
| **Recall (Sensibilidade)** | Proporção de churns reais detectados — minimiza churn não detectado |
| **Accuracy** | Referência geral (menos útil com dados desbalanceados) |

### 4.3 Trade-off: Falso Positivo vs. Falso Negativo

| Tipo de Erro | Consequência | Custo |
|-------------|-------------|-------|
| **Falso Positivo** (prediz churn, mas cliente não ia sair) | Gasta-se em retenção desnecessária | Baixo |
| **Falso Negativo** (não prediz churn, mas cliente sai) | Perde-se o cliente + custo de aquisição de substituto | **Alto** |

> **Conclusão**: Devemos priorizar **Recall** — é preferível gastar em retenção desnecessária do que perder o cliente.

---

## 5. Meta de Desempenho do Modelo

| Meta | Valor-alvo | Justificativa |
|------|-----------|---------------|
| **AUC-ROC** | **≥ 0.85** | Métrica principal — mede discriminação entre churners e não-churners. |
| **Recall** | **≥ 0.75** | Meta de negócio crítica — detectar ao menos 75% dos churns reais. |
| **F1-Score** | **≥ 0.65** | Equilíbrio mínimo aceitável entre precision e recall. |

---

## 6. SLOs (Service Level Objectives)

### 6.1 Modelo

| SLO | Objetivo |
|-----|----------|
| **AUC-ROC mínima** | ≥ 0.80 |
| **Recall mínimo** | ≥ 0.75 |
| **F1-Score mínimo** | ≥ 0.65 |
| **Model freshness** | Retreino mensal |

### 6.2 API de Inferência

| SLO | Objetivo |
|-----|----------|
| **Latência P95** | ≤ 200ms |
| **Disponibilidade** | ≥ 99.5% |
| **Throughput** | ≥ 50 req/s |

---

## 7. Pipeline de ML

```
[Dados Brutos] → [EDA] → [Feature Engineering] → [Pré-processamento]
                                                         ↓
                                              [Train/Test Split]
                                                         ↓
                                          [5 Modelos: Dummy, LR, RF, XGB, LGBM]
                                                         ↓
                                            [Avaliação: AUC, F1, Recall]
                                                         ↓
                                              [Registro no MLflow]
                                                         ↓
                                            [API FastAPI + Deploy]
```

---

## 8. Considerações Éticas e Limitações

### Vieses Conhecidos
- **Viés demográfico**: O modelo usa `gender` e `SeniorCitizen` como features. Isso pode levar a discriminação na oferta de retenção.
- **Viés de amostra**: Dataset contém apenas clientes de uma operadora específica.
- **Viés temporal**: Dataset é um snapshot estático.

### Limitações
- Modelo treinado com dados estáticos, sem features comportamentais em tempo real.
- Binary classification simplifica o problema — na realidade, churn pode ser parcial.

---

*Documento criado como parte do Tech Challenge Fase 1 — ML-PosTech-FIAP*
