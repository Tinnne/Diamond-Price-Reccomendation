from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load saved model and preprocessing tools
model = joblib.load("xgb_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Define input schema
class DiamondInput(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float

# Initialize app
app = FastAPI()

@app.post("/predict")
def predict(data: DiamondInput):
    # Extract input and transform
    raw = [[
        data.carat, data.cut, data.color, data.clarity,
        data.depth, data.table, data.x, data.y, data.z
    ]]
    
    # Encode categorical values
    raw_np = np.array(raw, dtype=object)
    raw_np[:, 1:4] = encoder.transform(raw_np[:, 1:4])
    raw_np = raw_np.astype(float)

    # Predict
    prediction = model.predict(raw_np)
    return {"predicted_price": round(float(prediction[0]), 2)}
