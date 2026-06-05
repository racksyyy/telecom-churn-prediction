# -*- coding: utf-8 -*-
"""hack_churn.ipynb

Original file is located at
    https://colab.research.google.com/drive/1ON_58xWvFk0rWfhNyEI-OD2sF_ofgkON
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('/content/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data

data.drop('customerID',axis=1,inplace=True)
data.describe()

print(data.info())

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

unique_elements = data.apply(pd.Series.unique)
print(unique_elements)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data)

# Columns to exclude
exclude_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in df.columns:
    if col not in exclude_cols:
        value_counts = df[col].value_counts()
        value_counts_df = value_counts.reset_index()
        value_counts_df.columns = [col, "Count"]

        sns.barplot(x=col, y="Count", data=value_counts_df, palette="pastel")
        plt.title(f"Unique Value Counts for {col}")
        plt.xlabel("Unique Values")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

from sklearn.preprocessing import LabelEncoder

categorical_cols = data.select_dtypes(include=['object', 'category']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data

correlation_matrix = data.corr()
print(correlation_matrix['Churn'].sort_values(ascending=False))

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title("Top 10 Important Features from Random Forest")
plt.show()

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

important_features = feature_importance_df[feature_importance_df['Importance'] >= 0.01]['Feature'].tolist()
important_features = feature_importance_df.head(10)['Feature'].tolist()
X_reduced = X[important_features]
df_reduced = data[important_features + ['Churn']]

df_reduced

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

X = df_reduced.drop('Churn', axis=1)
y = df_reduced['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X.isnull().sum())  # To check for missing values in each feature of X

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB()
}

for name, model in models.items():

    X_input = X_scaled if name in ["Logistic Regression", "SVM", "Naive Bayes"] else X
    scores = cross_val_score(model, X_input, y, cv=5, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}")

print(y.value_counts(normalize=True))

from sklearn.model_selection import GridSearchCV
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'class_weight': ['balanced']
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_scaled, y)

print("Best Logistic Regression Params:", grid_lr.best_params_)
print("Best F1 Score:", grid_lr.best_score_)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1')
grid_rf.fit(X, y)

print("Best Random Forest Params:", grid_rf.best_params_)
print("Best F1 Score:", grid_rf.best_score_)

scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
    param_grid_xgb,
    cv=5,
    scoring='f1'
)
grid_xgb.fit(X, y)

print("Best XGBoost Params:", grid_xgb.best_params_)
print("Best F1 Score:", grid_xgb.best_score_)

scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid_xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
    param_grid_xgb,
    cv=5,
    scoring='f1'
)
grid_xgb.fit(X, y)

print("Best XGBoost Params:", grid_xgb.best_params_)
print("Best F1 Score:", grid_xgb.best_score_)

from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = LogisticRegression(**grid_lr.best_params_)
scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1')
print(f"Logistic Regression (SMOTE) F1 Score: {scores.mean():.4f}")

rf = RandomForestClassifier(**grid_rf.best_params_)
scores = cross_val_score(rf, X_train_smote, y_train_smote, cv=5, scoring='f1')
print(f"Random Forest (SMOTE) F1 Score: {scores.mean():.4f}")

model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(**grid_rf.best_params_)),
    ('lr', LogisticRegression(**grid_lr.best_params_)),
    ('xgb', XGBClassifier(**grid_xgb.best_params_, use_label_encoder=False, eval_metric='logloss'))
], voting='soft')

voting_clf.fit(X_train_smote, y_train_smote)
y_pred = voting_clf.predict(X_test)
print(classification_report(y_test, y_pred))

import joblib

joblib.dump(model, 'final_churn_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

model = joblib.load('final_churn_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
