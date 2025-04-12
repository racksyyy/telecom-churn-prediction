# Telco Customer Churn Prediction

This project focuses on predicting customer churn using machine learning models. The dataset is based on a real-world telco company’s customer records. The goal is to help businesses proactively retain customers by identifying those at risk of leaving.

## Problem Statement

Customer churn is costly for any subscription-based business. Predicting churn allows companies to intervene before a customer leaves. This project explores multiple classification models to predict churn and compares their performance. It also addresses class imbalance and selects the most impactful features.

## Dataset

- **File**: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Target Variable**: `Churn` (binary: Yes/No)
- Features include:
  - Demographics (e.g., gender, senior citizen)
  - Services signed up (e.g., internet service, phone lines)
  - Account information (e.g., contract type, tenure, charges)

## Technologies Used

- Python (Colab)
- pandas, NumPy, seaborn, matplotlib
- scikit-learn, XGBoost
- imbalanced-learn (SMOTE)
- joblib (model serialization)

## Workflow

### 1. Data Preprocessing
- Removed missing values in `TotalCharges`
- Encoded categorical features using `LabelEncoder`
- Scaled features using `StandardScaler`

### 2. Feature Selection
- Used Random Forest to extract feature importances
- Selected top 10 important features for model training

### 3. Model Training
Trained and cross-validated five ML models:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine
- Naive Bayes

### 4. Hyperparameter Tuning
- Used GridSearchCV to tune:
  - Logistic Regression (`C`, `solver`)
  - Random Forest (`n_estimators`, `max_depth`, etc.)
  - XGBoost (`learning_rate`, `max_depth`, `n_estimators`, etc.)

### 5. Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Over-sampling Technique) for balanced training

### 6. Ensemble Modeling
- Built a **Voting Classifier** using Logistic Regression, Random Forest, and XGBoost for better performance

### 7. Model Export
- Saved the final tuned model and encoders using `joblib`

## Results

- **XGBoost** and **VotingClassifier** achieved the best F1-scores on test data
- Applied SMOTE significantly improved minority class recall
- Classification report and confusion matrices show improved balance in predictions

## How to Run

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
