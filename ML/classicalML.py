# standalone script. Run the script before executing streamlit job.

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import joblib

# --- SECTION 1: Load Dataset ---
data = pd.read_csv(r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\archive\Financial_datasets_log.csv")

# --- SECTION 2: Preprocess ---
# Use the same features as st3.py
feature_cols = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'type_CASH_OUT', 'type_TRANSFER'
]

# One-hot encode 'type' for CASH_OUT and TRANSFER
for t in ['CASH_OUT', 'TRANSFER']:
    data[f'type_{t}'] = (data['type'] == t).astype(int)

X = data[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_OUT', 'type_TRANSFER']]
y = data['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, stratify=y, random_state=42
)

# --- SECTION 3: Train Model ---
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump((model, scaler), "fraud_model.joblib")

# --- SECTION 4: Metrics ---
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_score))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- SECTION 5: Example Single Prediction ---
example = X.iloc[[0]]
example_scaled = scaler.transform(example)
pred = model.predict(example_scaled)[0]
prob = model.predict_proba(example_scaled)[0][1]
print(f"Example prediction: {pred}, probability: {prob:.4f}")


# ROC-AUC: 0.9997839817611955
# Precision: 0.14492753623188406
# Recall: 0.9981740718198417
# Classification Report:
# precision    recall  f1-score   support
#
# 0       1.00      0.99      1.00   1270881
# 1       0.14      1.00      0.25      1643
#
# accuracy                           0.99   1272524
# macro avg       0.57      1.00      0.62   1272524
# weighted avg       1.00      0.99      1.00   1272524
#
# Example prediction: 0, probability: 0.0003
