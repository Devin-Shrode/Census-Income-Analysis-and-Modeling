from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import joblib
import numpy as np
import pandas as pd

# Load trained CatBoost model
model_filename = "final_catboost_model.pkl"
model = joblib.load(model_filename)

# Load feature names
feature_names_filename = "feature_names.pkl"
feature_names = joblib.load(feature_names_filename)

# Define the optimized decision threshold
OPTIMAL_THRESHOLD = 0.41

# Initialize FastAPI app
app = FastAPI(title="Income Prediction API", description="Predicts income category based on user features.")

# Define request model for validation
class UserFeatures(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Income Prediction API"}

@app.post("/predict")
def predict_income(user: UserFeatures):
    try:
        # Validate input length
        if len(user.features) != len(feature_names):
            raise HTTPException(status_code=400, detail=f"Expected {len(feature_names)} features, but got {len(user.features)}.")

        # Convert input to DataFrame
        input_data = pd.DataFrame([user.features], columns=feature_names)

        # Predict probabilities
        prob = model.predict_proba(input_data)[:, 1]

        # Apply optimized threshold
        prediction = (prob >= OPTIMAL_THRESHOLD).astype(int)

        return {"predicted_income": ">50K" if prediction[0] == 1 else "<=50K"}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
