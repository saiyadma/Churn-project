# Customer Churn Prediction

**Predicting which telecom customers will leave — and explaining why**

!\[Python](https://img.shields.io/badge/Python-3.10-blue) !\[scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-green) !\[SHAP](https://img.shields.io/badge/Explainability-SHAP-orange) !\[ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.84-brightgreen)

\---

## The Business Problem

A telecom company loses revenue every time a customer cancels their service. The goal of this project is to predict which customers are likely to churn *before* they leave — giving the business time to intervene with targeted retention offers.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 20 features, 26% churn rate.

\---

## Results

|Metric|Score|
|-|-|
|**ROC-AUC**|**0.84**|
|Precision (Churned)|0.65|
|Recall (Churned)|0.52|
|F1 (Churned)|0.58|
|Overall Accuracy|80%|

> The model correctly identifies the majority of churners while maintaining high precision on retained customers (0.84 precision, 0.90 recall on "Stayed"). The ROC-AUC of 0.84 means the model is a strong ranker — it can correctly order a churner above a non-churner 84% of the time.

\---

## Key Findings

### 1\. Contract type is the #1 churn driver

Month-to-month customers churn at **42.7%** vs just **2.8%** for two-year contracts — a **15× difference**.

**Business recommendation:** Prioritize annual plan incentives at the point of sign-up. Even a modest discount to move new customers off month-to-month contracts could significantly reduce churn.

### 2\. New customers are the highest-risk segment

Customers in their **first 12 months** churn at **47.4%** — nearly half.

**Business recommendation:** Invest in onboarding and early engagement programs. A proactive check-in or loyalty reward at the 3-month mark targets the highest-risk window.

### 3\. High monthly charges correlate with churn

Customers paying more are more likely to leave, likely because they feel the value doesn't match the cost.

**Business recommendation:** Trigger loyalty discount outreach for high-spend customers who also have month-to-month contracts — the intersection of the two highest-risk factors.

\---

## Technical Approach

### Pipeline Architecture

Built using scikit-learn's `Pipeline` + `ColumnTransformer` to prevent data leakage and ensure reproducibility:

```python
preprocessor = ColumnTransformer(\\\[
    ('num', StandardScaler(), numeric\\\_cols),
    ('cat', OneHotEncoder(handle\\\_unknown='ignore'), categorical\\\_cols)
])

pipeline = Pipeline(\\\[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(
        n\\\_estimators=200, learning\\\_rate=0.05,
        max\\\_depth=4, random\\\_state=42
    ))
])
```

### Model Comparison (5-fold cross-validated ROC-AUC)

|Model|CV ROC-AUC|
|-|-|
|Logistic Regression (baseline)|\~0.82|
|Random Forest|\~0.84|
|**Gradient Boosting (final)**|**\~0.84**|

`class\\\_weight='balanced'` applied to account for the 26/74 class imbalance.

### Explainability with SHAP

Used `shap.TreeExplainer` to generate feature-level explanations for every prediction. The SHAP summary plot confirms that **contract type, tenure, and monthly charges** are the three dominant drivers — consistent with the EDA findings.

\---

## How to Run

```bash
# Clone the repo
git clone https://github.com/manarsaiyad/churn-prediction.git
cd churn-prediction

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place in data/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Launch the notebook
jupyter notebook notebooks/churn\\\_prediction\\\_starter.ipynb
```

**Requirements:** `pandas numpy scikit-learn matplotlib seaborn shap joblib jupyter`

\---

\---

## About

Built by **Manar Saiyad** — Computational Data Science, Michigan State University.
Relocating to Minneapolis, MN | [LinkedIn](https://linkedin.com/in/manarsaiyad) | [GitHub](https://github.com/manarsaiyad)

