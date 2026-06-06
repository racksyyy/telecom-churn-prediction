# Telco Customer Churn Prediction

This project predicts customer churn using an ensemble of machine learning models trained on a real-world telco dataset. The goal is to help businesses proactively retain customers by identifying those at risk of leaving.

## Problem Statement

Customer churn is costly for any subscription-based business. Predicting it allows companies to intervene before a customer leaves. This project benchmarks multiple classifiers, handles class imbalance via SMOTE, and combines the best models into a Voting Classifier — evaluated with metrics suited to imbalanced classification.

## Dataset

- **Source**: IBM Sample Dataset — Telco Customer Churn
- **File**: `Telco-Customer-Churn.csv`
- **Samples**: ~7,043 customers
- **Target Variable**: `Churn` (binary: Yes / No, ~26% positive class)
- **Features include**:
  - Demographics (gender, senior citizen, dependents)
  - Services (internet service, phone lines, streaming, tech support)
  - Account info (contract type, tenure, monthly charges, total charges)

## Technologies Used

- **Language**: Python (Jupyter / Google Colab)
- **Data**: pandas, NumPy
- **Visualization**: seaborn, matplotlib
- **Modeling**: scikit-learn, XGBoost
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Serialization**: joblib

## Project Structure

```
├── Data/
│   └── Telco-Customer-Churn.csv
├── Models/
│   ├── final_churn_model.pkl
│   └── label_encoders.pkl
└── churn_prediction.ipynb
```

## Workflow

### 1. Data Preprocessing
- Converted `TotalCharges` to numeric and dropped rows with nulls (~11 rows)
- Label-encoded all categorical columns using `LabelEncoder`
- Scaled numerical features using `StandardScaler` for models that require it

### 2. Exploratory Data Analysis
- Visualized churn distribution (~26% churn rate, confirming class imbalance)
- Computed and plotted a full correlation heatmap to identify feature relationships

### 3. Feature Selection
- Trained a baseline `RandomForestClassifier` to extract feature importances
- Selected the top 10 most important features for downstream modeling
- Key features: `tenure`, `TotalCharges`, `MonthlyCharges`, `Contract`, `InternetService`

### 4. Baseline Model Comparison
Benchmarked five classifiers using 5-fold cross-validation (accuracy):

| Model | CV Accuracy |
|---|---|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.80 |
| XGBoost | ~0.81 |
| SVM | ~0.79 |
| Naive Bayes | ~0.75 |

### 5. Hyperparameter Tuning
Used `GridSearchCV` (5-fold, F1 scoring) to tune:
- **Logistic Regression**: `C`, `solver`, `class_weight`
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `scale_pos_weight`

### 6. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set only to avoid data leakage
- Balanced the minority churn class before fitting final models

### 7. Ensemble Modeling
- Built a **soft Voting Classifier** combining tuned Logistic Regression, Random Forest, and XGBoost
- Soft voting uses predicted probabilities, giving more nuanced aggregation than hard voting

### 8. Evaluation
Final model evaluated on a held-out test set (20%) with:
- **Classification Report** — precision, recall, F1 per class
- **Confusion Matrix** — visualized with `ConfusionMatrixDisplay`
- **ROC Curve + AUC Score** — threshold-independent discrimination ability
- **Precision-Recall Curve + Average Precision** — more informative than ROC for imbalanced classes

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
   ```
3. Place the dataset in the `Data/` folder
4. Run `churn_prediction.ipynb` top to bottom

## Results

- The **Voting Classifier** (LR + RF + XGBoost) achieved the best overall performance
- SMOTE improved recall on the minority (churn) class significantly
- ROC-AUC and Average Precision scores confirm strong discrimination ability on the positive class
