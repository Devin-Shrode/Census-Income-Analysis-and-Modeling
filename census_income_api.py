from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import joblib
import numpy as np
import pandas as pd

# Load trained CatBoost model
model_filename = "final_catboost_model.pkl"
model = joblib.load(model_filename)

# Corrected feature order retrieved from training data
expected_features = [
    "relationship_Unmarried", "occupation_Unknown_Occupation", "sex_Male", 
    "occupation_Prof-specialty", "marital-status_Widowed", "relationship_Not-in-family", 
    "race_White", "marital-status_Separated", "occupation_Farming-fishing", 
    "occupation_Exec-managerial", "education", "relationship_Other-relative", 
    "occupation_Handlers-cleaners", "marital-status_Never-married", "workclass_Unknown_Workclass", 
    "native-country_Mexico", "workclass_Private", "relationship_Wife", "age", "capital-gain", 
    "occupation_Other-service", "relationship_Own-child", "occupation_Machine-op-inspct", 
    "hours-per-week", "capital-loss", "workclass_Self-emp-inc", 
    "marital-status_Married-civ-spouse", "race_Black", "education-num"
]

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
        if len(user.features) != len(expected_features):
            raise HTTPException(status_code=400, detail=f"Expected {len(expected_features)} features, but got {len(user.features)}.")

        # Convert input to DataFrame
        input_data = pd.DataFrame([user.features], columns=expected_features)

        # Predict income category
        prediction = model.predict(input_data)
        return {"predicted_income": ">50K" if int(prediction[0]) == 1 else "<=50K"}

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Validation Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
