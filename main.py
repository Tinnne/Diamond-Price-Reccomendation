from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and tools
model = joblib.load("xgb_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Input schema
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

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict")
def predict(data: DiamondInput):
    raw = [[
        data.carat, data.cut, data.color, data.clarity,
        data.depth, data.table, data.x, data.y, data.z
    ]]
    raw_np = np.array(raw, dtype=object)
    raw_np[:, 1:4] = encoder.transform(raw_np[:, 1:4])
    raw_np = raw_np.astype(float)
    prediction = model.predict(raw_np)
    return {"predicted_price": round(float(prediction[0]), 2)}
