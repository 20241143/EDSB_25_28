## EDSB_25_28
### 👥 Team Members

| Name | Student ID |
|------|------------|
| Diogo Ferreira | 20241143 |
| Francisco Moutinho | 20240818 |
| Pedro Boto | 20241144 |
| João Carvalho | 20241340 |

**Navigation**
- [Home](README.MD)
- [How to Setup](How_to_setup_git_hugging.MD)
- [Hugging Face README](./96-huggingface_space/README.md)

### Requirements
- See [requirements.txt](requirements.txt) for dependencies.

---

# 📞 Predicting the Success of Bank Telemarketing Campaigns

Este projeto tem como objetivo prever o sucesso de chamadas de telemarketing realizadas por um banco português, utilizando técnicas de Machine Learning e análise exploratória de dados (EDA).  
O dataset contém informação demográfica, socioeconómica e histórica das campanhas, permitindo construir modelos preditivos e identificar os fatores mais relevantes para a conversão.  

> Baseado no documento de descrição do projeto fornecido (Bank Telemarketing Campaigns) *Bank Telemarketing Campaigns (1).pdf*


### 🧠 Objetivos do Projeto

- Criar modelos de machine learning para prever se uma chamada de telemarketing resultará numa subscrição de depósito a prazo.
- Explorar o dataset através de gráficos e análises estatísticas.
- Interpretar modelos usando SHAP.
- Avaliar desempenho com métricas adequadas e matrizes de confusão.
- Identificar insights acionáveis para melhorar campanhas futuras.
- Efetuar o deploy do modelo selecionado para um Space de Hugging Face

## 📊 Exploratory Data Analysis (EDA)

A análise exploratória incluiu:

### **Univariate Analysis**
- Distribuições de variáveis numéricas (ex.: idade, taxa Euribor, duração de chamada).
- Contagem das classes (desbalanceamento entre "sucesso" e "insucesso").

### **Bivariate / Multivariate Analysis**
- Correlações entre variáveis.
- Relação entre features e a variável alvo.
- Scatter plots, boxplots e heatmaps.

### **Segmentação**
- Comportamento por grupos etários.
- Efeitos de campanhas anteriores.
- Impacto de variáveis macroeconómicas.

Os principais gráficos gerados incluíram:
- Histogramas
- Boxplots
- Heatmap de correlação
- Scatter plots relevantes

## 🤖 Modelos Desenvolvidos

Foram treinados e avaliados três modelos principais:

### **1️⃣ Baseline: Regressor Linear Simples**
- Criado para estabelecer um ponto de comparação.
- Permite avaliar se modelos mais complexos trazem melhorias reais.

### **2️⃣ Regressão Linear**
- Modelo interpretável com coeficientes diretos.
- Útil para entender relações lineares e verificar multicolinearidade.
- Avaliado com métricas como Accuracy, Recall, Precisão e F1 (após binarização).

### **3️⃣ XGBoost Classifier**
- Modelo principal devido ao melhor desempenho.
- Captura relações não lineares e interações entre features.
- Inclui tuning de hiperparâmetros.

## 📈 Avaliação dos Modelos

As métricas utilizadas incluíram:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score** - *Métrica alvo*
- **ROC-AUC**

Também foram geradas **matrizes de confusão** para interpretar os erros de classificação em cada modelo.

## 🔍 Interpretação com SHAP

Para explicar o comportamento dos modelos, especialmente o XGBoost, foram usadas:

- **SHAP Summary Plots**
- **SHAP Bar Importance**
- **SHAP Dependence Plots**

Estes gráficos permitiram:
- Identificar features com maior impacto.
- Avaliar a direção e força da influência de cada variável.
- Comunicar resultados a stakeholders de forma clara e transparente.

## 🏁 Conclusões

- O modelo **XGBoost** apresentou o melhor desempenho global.
- Variáveis como histórico de campanhas anteriores e indicadores económicos mostraram elevada importância.
- As análises SHAP forneceram insights claros e justificáveis para utilização prática.
- A EDA permitiu compreender padrões úteis e confirmar hipóteses sobre comportamentos dos clientes.

## 🗂 Estrutura do Repositório
```
📦 EDSB_25_28
│
├── .github/
│   └── workflows/
│       └── deploy_to_hf.yml
│
├── 01-data/
│   ├── 01-bronze/
│   ├── 02-silver/
│   └── 03-gold/
│
├── 02-notebooks/
│   ├── __pycache__/
│   ├── 01-EDA.ipynb
│   ├── 02-Data Cleaning and Preprocessing.ipynb
│   ├── 03-Baseline.ipynb
│   ├── 04-Logistic Regression.ipynb
│   ├── 05-XGBoost.ipynb
│   ├── 06-best_model.ipynb
│   └── utils.py
│
├── 96-huggingface_space/
│   ├── app.py
│   ├── model.pkl
│   ├── README.md
│   └── requirements.txt
│
├── 97-model_metrics/
│   ├── Baseline/
│   ├── Logistic_Regression_Model/
│   ├── Logistic_Regression_Model_yyyyMMdd_hhmmss
│   ├── XGBoost_Model/
│   └── XGBoost_Model_yyyyMMdd_hhmmss
│
├── 98-reports/
│
├── 99-old/
│
├── .env
├── .gitignore
├── Analysis_baseline.md
├── How_to_setup_git_hugging.md
├── README.md
└── requirements.txt
```