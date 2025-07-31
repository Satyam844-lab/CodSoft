**Bank Customer Churn Prediction**

A machine learning project to predict which bank customers are likely to leave ("churn") using logistic regression on real customer data. This repository includes all steps from data preparation to evaluation and a user-friendly way to enter new customer details for live predictions.

**🚀 Project Overview**

**Problem**

Customer churn (leaving the bank) is costly for businesses. Predicting who is likely to churn helps the bank retain valuable customers through targeted interventions.

**Objective **

Use historical customer data to build a model that predicts churn, allowing for early retention efforts.

Approach: Step-by-step workflow using Python, pandas, and scikit-learn, focused on clarity and learning.

**📂 Dataset**


Name: Bank Customer Churn Prediction (from Kaggle)
Link:[Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)


Rows: 10,000

Features: Demographic, financial, and behavioral attributes

Target: Exited (1 = churned/left, 0 = stayed)

Key features:

CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender
**
🛠️ Workflow & Methods**


Data Cleaning:

Dropped irrelevant columns: RowNumber, CustomerId, Surname

Exploratory Data Analysis:

Inspected missing values, got data overview

Data Preprocessing:

One-hot encoded categorical features (Geography, Gender)

Scaled numerical features for uniformity

Modeling:

Split data into train/test sets (80/20 split)

Trained logistic regression for interpretability and baseline performance

Evaluation:

Accuracy, precision, recall, F1-score, confusion matrix

Found model is strong at identifying non-churners, less so at catching actual churners (common for simple models on imbalanced data)

User Interaction:

Built an interactive function to input new customer data and predict churn likelihood in real time

📈 Results
Accuracy: ~81%

Recall (for churners): ~20% (model catches some, but not all true churners—typical for this dataset and simple models)

Business insight: Customers with higher account activity, more products, or larger balances are less likely to churn, while certain demographics or low engagement raise risk.

**📝 Limitations & Future Improvements**
Logistic regression gives a strong baseline but struggles with imbalanced data (low recall for churners).

Possible improvements:

Use advanced models (Random Forest, XGBoost)

Balance the dataset or adjust class weights

Feature engineering

Build a GUI or deploy as a web app
