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
| **Precision** | Proporção de verdadeiros positivos entre predições positivas — minimiza "alarmes falsos" |
| **Recall (Sensibilidade)** | Proporção de churns reais detectados — minimiza churn não detectado |
| **PR-AUC** | Mais informativa que ROC-AUC em cenários de alto desbalanceamento |
| **Accuracy** | Referência geral (menos útil com dados desbalanceados) |

### 4.3 Métrica de Negócio: **Custo de Churn Evitado**

**Premissas** (baseadas em benchmarks do setor de telecomunicações):
- **Custo de aquisição de cliente (CAC)**: ~R$ 400 por novo cliente
- **Custo médio de retenção (ação proativa)**: ~R$ 80 por cliente contatado
- **Receita mensal média por cliente**: ~R$ 65 (baseado em `MonthlyCharges` do dataset)

**Fórmula**:
```
Custo Evitado = (True Positives × CAC) − (True Positives × Custo_Retenção) − (False Positives × Custo_Retenção)

Custo Evitado = TP × (CAC − Custo_Retenção) − FP × Custo_Retenção
Custo Evitado = TP × R$320 − FP × R$80
```

**Interpretação**: Cada churn corretamente previsto e evitado economiza R$320 (diferença entre adquirir um novo cliente e reter o atual). Cada falso positivo custa R$80 (custo da ação de retenção em vão).

### 4.4 Trade-off: Falso Positivo vs. Falso Negativo

| Tipo de Erro | Consequência | Custo |
|-------------|-------------|-------|
| **Falso Positivo** (prediz churn, mas cliente não ia sair) | Gasta-se R$80 em retenção desnecessária | Baixo |
| **Falso Negativo** (não prediz churn, mas cliente sai) | Perde-se o cliente + gasta R$400 para adquirir substituto | **Alto** |

> **Conclusão**: Devemos priorizar **Recall** sobre Precision — é preferível gastar R$80 em retenção desnecessária do que perder R$400 em aquisição.

---

## 5. Meta de Desempenho do Modelo

### 5.1 Por que não usamos Acurácia como meta

Acurácia é uma métrica **enganosa** para este problema. O dataset é desbalanceado: 73,5% dos clientes não cancelam. Um modelo que **nunca** detecta churn já atinge 73,5% de acurácia — portanto, atingir 85% de acurácia seria uma meta facilmente alcançada sem nenhum valor de negócio real.

Exemplo: um modelo com 85% de acurácia pode estar detectando apenas 20% dos churns reais (recall baixo), enquanto o custo de cada cliente perdido é R$400. A acurácia mascara esse prejuízo.

### 5.2 Metas do Modelo (Targets)

| Meta | Valor-alvo | Justificativa |
|------|-----------|---------------|
| **AUC-ROC** | **≥ 0.85** | Métrica principal — mede discriminação entre churners e não-churners independente do threshold. O baseline LogisticRegression atinge ~0.84, portanto o modelo final deve superá-lo. |
| **Recall** | **≥ 0.75** | Meta de negócio crítica — detectar ao menos 75% dos churns reais. Cada falso negativo custa R$400 (CAC), enquanto um falso positivo custa apenas R$80 (retenção). Priorizar recall sobre precisão é a decisão correta dado o custo assimétrico dos erros. |
| **F1-Score** | **≥ 0.65** | Equilíbrio mínimo aceitável entre precision e recall — garante que o modelo não sacrifica totalmente a precisão em prol do recall. |
| **PR-AUC** | **≥ 0.60** | Complementa a AUC-ROC em cenários de desbalanceamento — mais sensível à performance na classe minoritária (Churn=Yes). |

> **Referência de baseline**: LogisticRegression atinge AUC-ROC ≈ 0.84 e Recall ≈ 0.55. O modelo final deve superar ambos os valores.

---

## 6. SLOs (Service Level Objectives)

### 6.1 Modelo

| SLO | Objetivo | Justificativa |
|-----|----------|---------------|
| **AUC-ROC mínima** | ≥ 0.80 | Piso de discriminação aceitável para produção |
| **Recall mínimo** | ≥ 0.75 | Detectar pelo menos 75% dos churns reais |
| **F1-Score mínimo** | ≥ 0.65 | Equilíbrio mínimo aceitável |
| **Model freshness** | Retreino mensal | Dados de churn mudam com sazonalidade |

### 6.2 API de Inferência

| SLO | Objetivo | Justificativa |
|-----|----------|---------------|
| **Latência P95** | ≤ 200ms | Resposta rápida para integração com CRM |
| **Disponibilidade** | ≥ 99.5% | Serviço confiável para equipe de retenção |
| **Throughput** | ≥ 50 req/s | Suporte a batch scoring diário |

---

## 7. Pipeline de ML

```
[Dados Brutos] → [EDA] → [Feature Engineering] → [Pré-processamento]
       ↓                                                   ↓
[ML Canvas]                                    [Train/Test Split]
                                                        ↓
                                            [Baselines (Dummy + LogReg)]
                                                        ↓
                                              [Modelos Avançados (MLP)]
                                                        ↓
                                            [Avaliação + Comparação]
                                                        ↓
                                              [Registro no MLflow]
                                                        ↓
                                            [API FastAPI + Deploy]
```

---

## 8. Considerações Éticas e Limitações

### Vieses Conhecidos
- **Viés demográfico**: O modelo usa `gender` e `SeniorCitizen` como features. Isso pode levar a discriminação na oferta de retenção. Deve-se monitorar equidade entre grupos.
- **Viés de amostra**: Dataset contém apenas clientes de uma operadora específica. Generalização para outras operadoras não é garantida.
- **Viés temporal**: Dataset é um snapshot estático. Comportamento de churn pode variar com condições de mercado.

### Limitações
- Modelo treinado com dados estáticos, sem features comportamentais em tempo real (ex.: chamadas ao SAC, reclamações).
- Sem dados de concorrência (ofertas de competidores que poderiam causar churn).
- Binary classification simplifica o problema — na realidade, churn pode ser parcial (downgrade de plano).

### Cenários de Falha
- **Data drift**: Mudanças na base de clientes ou nos planos oferecidos podem degradar o modelo.
- **Concept drift**: O conceito de "risco de churn" pode mudar com novas políticas da empresa.
- **Distribuição de features**: Novos tipos de contrato ou serviço não presentes no training set.

---

## 9. Abordagem de Monitoramento (Planejado)

| Aspecto | Ferramenta/Método | Frequência |
|---------|-------------------|------------|
| **Performance do modelo** | MLflow + métricas em produção | Semanal |
| **Data drift** | Evidently AI | Diário |
| **Latência da API** | Middleware de latência (já implementado) | Contínuo |
| **Alertas** | Threshold-based (AUC-ROC < 0.75) | Automático |

---

*Documento criado como parte do Tech Challenge Fase 1 — ML-PosTech-FIAP*
