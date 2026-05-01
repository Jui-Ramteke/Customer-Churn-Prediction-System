import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# --- Setup Directories ---
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("🚀 Starting Advanced Customer Churn Prediction Pipeline...\n")

# --------------------------------------------------
# 1. GENERATE REALISTIC, IMBALANCED SYNTHETIC DATA
# --------------------------------------------------
print("📊 Generating Synthetic Data (Imbalanced)...")
np.random.seed(42)
n_samples = 3000

data = {
    'customer_id': [f"CUST-{i}" for i in range(1, n_samples + 1)],
    'tenure_months': np.random.randint(1, 72, n_samples),
    'billing_amount': np.random.uniform(20.0, 150.0, n_samples),
    'support_tickets': np.random.randint(0, 10, n_samples),
    'sla_breaches': np.random.randint(0, 4, n_samples),
    'active_days': np.random.randint(1, 31, n_samples)
}
df = pd.DataFrame(data)

# Logic for Churn: High frustration + low usage = Churn
# We keep the base probability low to simulate a realistic ~15-20% churn rate
churn_prob = np.zeros(n_samples) + 0.05 
churn_prob += np.where(df['support_tickets'] > 5, 0.2, 0.0)
churn_prob += np.where(df['sla_breaches'] > 1, 0.3, 0.0)
churn_prob += np.where(df['active_days'] < 10, 0.25, 0.0)
churn_prob -= np.where(df['tenure_months'] > 24, 0.15, 0.0) # Loyalty reduces churn

churn_prob = np.clip(churn_prob, 0, 1)
df['Churn'] = np.random.binomial(1, churn_prob)

df.to_csv('data/advanced_churn_data.csv', index=False)
print(f"✅ Data saved to data/advanced_churn_data.csv")
print(f"📉 Base Churn Rate in raw data: {(df['Churn'].mean() * 100):.1f}%\n")

# --------------------------------------------------
# 2. FEATURE ENGINEERING ENGINE
# --------------------------------------------------
print("⚙️ Engineering Advanced Features...")
# These are the exact features your API and UI will use to explain churn
df['engagement_rate'] = df['active_days'] / 30
df['support_intensity'] = df['support_tickets'] + (3 * df['sla_breaches'])
df['price_to_tenure'] = df['billing_amount'] / (df['tenure_months'] + 1)

# Drop ID as it has no predictive power
X = df.drop(['customer_id', 'Churn'], axis=1)
y = df['Churn']

# --------------------------------------------------
# 3. IMBALANCED DATA HANDLING (SMOTE)
# --------------------------------------------------
print("⚖️ Splitting and Balancing Data with SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"   Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"   After SMOTE:  {y_train_smote.value_counts().to_dict()}\n")

# --------------------------------------------------
# 4. ADVANCED MODEL TRAINING (XGBoost)
# --------------------------------------------------
print("🧠 Training XGBoost Classifier...")
# XGBoost doesn't strictly require standard scaling, making our pipeline cleaner
model = XGBClassifier(
    n_estimators=150, 
    max_depth=4, 
    learning_rate=0.1, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_smote, y_train_smote)

# --------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------
print("\n📈 Evaluating Model...")
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
plt.title('XGBoost Confusion Matrix (with SMOTE)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/xgboost_confusion_matrix.png')
plt.close()
print("✅ Confusion matrix saved to outputs/xgboost_confusion_matrix.png")

# Save Feature Importance Plot
importances = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('outputs/xgboost_feature_importance.png')
plt.close()
print("✅ Feature importance saved to outputs/xgboost_feature_importance.png\n")

# --------------------------------------------------
# 6. SAVE ARTIFACTS FOR PRODUCTION API
# --------------------------------------------------
joblib.dump(model, 'models/xgboost_churn.pkl')
# Saving the feature columns so our API knows the exact order expected by XGBoost
joblib.dump(list(X.columns), 'models/model_features.pkl') 

print("💾 Model and Feature map saved to models/ folder.")
print("🎉 Advanced Pipeline Complete!")