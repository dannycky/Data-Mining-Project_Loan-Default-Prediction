
# High-Throughput Credit Risk & Loan Default Processing Pipeline

# High-Throughput Credit Risk & Loan Default Processing Pipeline
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.2+-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE--NC-yellowgreen.svg)](https://imbalanced-learn.org/)

An end-to-end data processing and predictive analytical engineering pipeline built to intake, clean, balance, transform, and evaluate large-scale credit application data. This project showcases a production-ready approach to handling raw financial records, optimizing data pipelines for deterministic machine learning downstream, engineering mathematical features, and implementing performance scoring configurations.

---

## 🏗️ Pipeline Architecture & Data Flow

The architecture focuses on structural idempotency, data consistency, and robust class-based preprocessing to ensure that data flows seamlessly from a raw operational format to an analytical matrix optimized for high-performance modeling.

```text
 ┌──────────────────────┐      ┌────────────────────────────────────────────────────────┐
 │      DATA INPUT      │      │             PREPROCESSING & ETL PIPELINE               │
 │                      │      │                                                        │
 │  Raw Credit Ledger   │ ───> │  1. Stratified Splitting (Train/Val/Test Isolation)   │
 │   (21k+ Loan Rows)   │      │  2. Deterministic Imputation (Median / Most Frequent)  │
 └──────────────────────┘      │  3. Standardized Feature Scaling & One-Hot Encoding   │
                               └────────────────────────────────────────────────────────┘
                                                           │
                                                           ▼
 ┌──────────────────────┐      ┌────────────────────────────────────────────────────────┐
 │   MODEL EVALUATION   │      │            DATA RESAMPLING & ALIGNED TESTING           │
 │                      │      │                                                        │
 │   Ensemble, LogReg   │ <─── │  5. Threshold Calibration (Optimizing Val/Test F1)     │
 │  & Custom Thresholds │      │  4. Synthetic Resampling (SMOTE-NC on Train Segment)   │
 └──────────────────────┘      └────────────────────────────────────────────────────────┘

```

1. **Deterministic Separation:** Avoids target leakage by isolating data before any stateful transformations are performed.
2. **Missing Value Imputation:** Implements robust fallbacks using statistical metrics tailored by feature typography.
3. **Imbalanced Class Mitigation:** Handles skewed financial target vectors safely at the pipeline level using synthetic minority over-sampling.
4. **Vector Alignment & Serialization:** Merges complex independent categorical matrices and scaled statistical numerical representations into compact array representations.

---

## 🛠️ Technical Stack & Data Engineering Competencies

* **Languages & Paradigms:** Python, Functional & Object-Oriented Pipeline Structuring, Automated ETL Matrix Transformations.
* **Core Frameworks:** Pandas, NumPy, Scikit-Learn.
* **Resampling Engineering:** Imbalanced-Learn (`SMOTENC` - Synthetic Minority Over-sampling Technique for Nominal and Continuous features).
* **Statistical Operations:** Standard Scaling, Median and Mode-Based Vector Imputation, Multi-Class One-Hot Matrix Formats.
* **Pipeline Metrics Architecture:** Hyperparameter Optimization (GridSearch), Stratification, Threshold Calibration, Matrix Evaluation (ROC-AUC, Confusion Matrices, Complex F1 Optimization).

---

## ⚡ Core Engineering Workflows & Implementations

### 1. Robust Anti-Leakage Data Partitioning

To prevent information leakage from the valuation and testing spaces into training sets, a multi-tier stratified splitting workflow isolates data structures prior to stateful execution.

```python
# Multi-stage stratified data isolation pipeline
X = df.drop(columns=['LoanID', 'Default'])
y = df['Default']

# 80/20 Train-Test split preserves relational balance
X_train0, X_test, y_train0, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sub-isolation of validation sets 
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.2, random_state=42, stratify=y_train0)

```

### 2. High-Performance Synthesized Matrix Balancing (`SMOTE-NC`)

Financial risk sets suffer from strong target element class skews (non-defaulting vs defaulting applicants). To resolve this while managing a combination of numerical variables and non-numeric identifiers, the data ingestion pipeline embeds `SMOTENC`. This ensures artificial records are constructed using valid structural mathematical parameters rather than arbitrary calculations.

```python
# Precise column-type segregation for structural resampling alignment
num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Interest_to_Credit_Ratio']
cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Identifying index boundaries of categorical parameters
cat_index = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

# Constructing synthetic samples exclusively on isolated training sets
X_train_smotenc, y_train_smotenc = SMOTENC(categorical_features=cat_index, random_state=42).fit_resample(X_train_pre, y_train)

```

### 3. Production Preprocessing & Feature Engineering

Rather than manually manipulating parameters, data manipulation processes are bundled into clean mathematical transforms. This includes a custom-engineered financial interaction feature (`Interest_to_Credit_Ratio`) to capture nonlinear risk indicators.

```python
def add_new_features(dataset):
    dataset = dataset.copy()
    # Dynamic mathematical risk metric generation
    dataset['Interest_to_Credit_Ratio'] = dataset['InterestRate'] / (dataset['CreditScore'] + 1)
    return dataset

X_train = add_new_features(X_train)
X_test = add_new_features(X_test)
X_val = add_new_features(X_val)

```

### 4. Dynamic Probability Threshold Calibration

Standard classifiers default to a standard `0.50` probability cutoff. In credit risk pipelines, catching actual defaults is highly critical. The evaluation phase runs automated step-wise thresholds across prediction scores to maximize validation performance metrics.

```python
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold_dt = 0
best_f1_dt = 0

for threshold in thresholds:
    y_val_pred_temp = (y_val_proba >= threshold).astype(int)
    current_f1 = f1_score(y_val, y_val_pred_temp)
    if current_f1 > best_f1_dt:
        best_f1_dt = current_f1
        best_threshold_dt = threshold

```

---

## 📊 Empirical Transformation & Model Performance Outcomes

The pipeline successfully structures data across 5 production models: Decision Trees, Naive Bayes, K-Nearest Neighbors, Logistic Regression, and Ensemble Random Forests.

* **Data Compression Scale:** Safely processed, missing-imputed, and transformed over **21,500+ production application profiles**.
* **Imbalance Resolution:** Correctly balanced the highly skewed training target array into symmetric matrices (**288,886 synthetic balanced profiles**).
* **Pipeline Results:** * **Logistic Regression:** Achieved balanced classification outputs with a test set Accuracy of **77.7%** and a stable ROC-AUC score of **0.752**.
* **Ensemble Learning (Random Forest):** Delivered the highest structural prediction Accuracy at **78.5%**, providing strong, low-variance risk scores.



---

## 🚀 Pipeline Initialization & Local Setup

### Prerequisites

* Python 3.10+
* Virtual Environment Configuration Tool (`venv` or `conda`)

### Local Environment Setup

1. Clone the analytics asset engine repository to your station:
```bash
git clone [https://github.com/dannycky/credit-risk-pipeline.git](https://github.com/dannycky/credit-risk-pipeline.git)
cd credit-risk-pipeline

```


2. Establish an isolated python environment and trigger package synchronization:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib

```


3. Execute the pipeline notebook using your preferred platform or your shell terminal:
```bash
jupyter notebook Data_Mining_Project_LoanDefault_Code.ipynb

```



