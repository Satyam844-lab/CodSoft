import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv("fraudTest.csv")

cols_to_drop = [
    'Unnamed: 0', 'cc_num', 'merchant', 'first', 'last',
    'street', 'job', 'dob', 'trans_num', 'unix_time'
]
df.drop(columns=cols_to_drop, inplace=True)

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())

sns.countplot(x="is_fraud", data=df)
plt.title("Class Distribution (0 = Legitimate, 1 = Fraud)")
plt.show()

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
df.drop('trans_date_trans_time', axis=1, inplace=True)

categorical_columns = ['category', 'gender', 'state', 'city']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

numeric_columns = ['amt', 'lat', 'long', 'hour', 'dayofweek', 'city_pop', 'merch_lat', 'merch_long']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    solver='liblinear'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

original_df = pd.read_csv("fraudTest.csv")
ohe_reference = {}
for col in categorical_columns:
    ohe_reference[col] = sorted(original_df[col].dropna().unique())

feature_columns = list(X.columns)

def user_transaction_prediction(model, scaler, feature_cols, ohe_ref):
    print("\nEnter transaction details to predict fraud:")

    amt = float(input("Transaction amount: "))
    lat = float(input("Merchant latitude: "))
    long = float(input("Merchant longitude: "))
    hour = int(input("Transaction hour (0-23): "))
    dayofweek = int(input("Day of week (0=Mon, 6=Sun): "))

    category = input(f"Category ({', '.join(ohe_ref['category'])}): ").strip()
    gender = input(f"Gender ({', '.join(ohe_ref['gender'])}): ").strip()
    state = input(f"State ({', '.join(ohe_ref['state'])}): ").strip()
    city = input(f"City ({', '.join(ohe_ref['city'])}): ").strip()

    user_data = pd.DataFrame(
        [[amt, lat, long, hour, dayofweek, category, gender, state, city]],
        columns=['amt', 'lat', 'long', 'hour', 'dayofweek', 'category', 'gender', 'state', 'city']
    )

    user_data = pd.get_dummies(user_data, columns=['category', 'gender', 'state', 'city'], drop_first=True)

    for col in feature_cols:
        if col not in user_data.columns:
            user_data[col] = 0

    user_data = user_data[feature_cols]

    user_data[numeric_columns] = scaler.transform(user_data[numeric_columns])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    print("\nPrediction Result:")
    print("⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction")
    print(f"Fraud Probability: {probability:.3f}")

user_transaction_prediction(model, scaler, feature_columns, ohe_reference)
