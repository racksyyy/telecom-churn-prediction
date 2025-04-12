# telecom-churn-prediction
# Telecom Churn Prediction

This project aims to predict customer churn using various classification models, comparing their performance, and visualizing the results through a dashboard.

## Problem Statement

Customer churn is a major challenge in the telecom (or banking) industry. This project uses machine learning to predict whether a customer will subscribe to a term deposit offer, which is a proxy for customer engagement or retention.

## Dataset

- **Source**: `bank.csv`
- The dataset includes attributes such as age, job type, marital status, education, contact method, previous outcome, and more.
- Target variable: `y` (binary: yes/no)

## Technologies Used

- Google Colab (Python)
- pandas, seaborn, matplotlib
- scikit-learn (for model training and evaluation)
- Streamlit (for dashboard)

## Workflow

1. **Data Cleaning**
   - Handled "unknown" values
   - Label encoding for binary features
   - One-hot encoding for categorical features

2. **Model Training**
   - Trained five classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Gradient Boosting

3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Classification Report

4. **Hyperparameter Tuning**
   - Grid Search on Random Forest model

5. **Dashboard**
   - Built using Streamlit
   - Horizontal bar chart showing F1-scores
   - Highlights best performing model

## Results

| Model              | F1 Score |
|--------------------|----------|
| Logistic Regression| 0.87     |
| Decision Tree      | 0.85     |
| Random Forest      | 0.87     |
| Gradient Boosting  | 0.88     |
| SVM                | 0.83     |

- **Best Model**: Gradient Boosting (F1-score: 0.92 after tuning)

## How to Run

1. Clone this repository or open the notebook in Google Colab.
2. Run `hackathon.py` to train and evaluate models.
3. To launch the dashboard:
   ```bash
   streamlit run app.py
