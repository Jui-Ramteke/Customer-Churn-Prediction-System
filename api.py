from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Initialize the FastAPI app
app = FastAPI(title="Customer Churn Prediction API", version="1.0")

# 2. Load the trained model and expected feature names
print("Loading model...")
try:
    model = joblib.load('models/xgboost_churn.pkl')
    feature_columns = joblib.load('models/model_features.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 3. Define the Input Data Schema (What the API expects to receive)
class CustomerData(BaseModel):
    customer_id: str
    tenure_months: int
    billing_amount: float
    support_tickets: int
    sla_breaches: int
    active_days: int

# 4. Create the Prediction Endpoint
@app.post("/score")
def predict_churn(data: CustomerData):
    
    # Step A: Perform Feature Engineering (On-the-fly)
    engagement_rate = data.active_days / 30
    support_intensity = data.support_tickets + (3 * data.sla_breaches)
    price_to_tenure = data.billing_amount / (data.tenure_months + 1)
    
    # Step B: Create a DataFrame formatted exactly how XGBoost expects it
    input_dict = {
        'tenure_months': [data.tenure_months],
        'billing_amount': [data.billing_amount],
        'support_tickets': [data.support_tickets],
        'sla_breaches': [data.sla_breaches],
        'active_days': [data.active_days],
        'engagement_rate': [engagement_rate],
        'support_intensity': [support_intensity],
        'price_to_tenure': [price_to_tenure]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Ensure columns are in the exact order the model was trained on
    input_df = input_df[feature_columns]
    
    # Step C: Get Model Prediction
    # predict_proba returns an array like [[Prob_Class_0, Prob_Class_1]]
    churn_prob = float(model.predict_proba(input_df)[0][1])
    
    # Step D: Apply Business Logic
    if churn_prob >= 0.70:
        segment = "High Risk 🔴"
        action = "Priority Call & Discount Offer"
    elif churn_prob >= 0.40:
        segment = "Medium Risk 🟡"
        action = "Send Re-engagement Email"
    else:
        segment = "Low Risk 🟢"
        action = "No Action Needed"
        
    # Step E: Return the JSON response
    return {
        "customer_id": data.customer_id,
        "churn_probability": round(churn_prob, 3),
        "segment": segment,
        "recommended_action": action
    }

# Run the server automatically when the script is executed
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)