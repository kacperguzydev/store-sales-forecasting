from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Store Sales Forecasting API")


class PredictionRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int


class PredictionResponse(BaseModel):
    predicted_sales: float


@app.get("/")
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Store Sales Forecasting API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Dummy prediction endpoint.
    Currently returns a placeholder value - will be replaced with the real
    trained model later in the project.
    """
    dummy_prediction = 100.0
    return PredictionResponse(predicted_sales=dummy_prediction)