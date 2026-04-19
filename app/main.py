"""FastAPI app for serving inflation nowcasting predictions."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inflation_nowcasting.config import get_settings
from inflation_nowcasting.pipeline import generate_latest_forecast, run_training_pipeline

app = FastAPI(title="Inflation Nowcasting API", version="0.1.0")


class TrainResponse(BaseModel):
    selected_model: str
    model_path: str
    metrics_path: str
    candidates_evaluated: int


class ForecastResponse(BaseModel):
    forecast_date: str
    prediction_mom_percent: float
    prediction_annualized_percent: float
    model_name: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
def train(refresh_data: bool = False) -> TrainResponse:
    settings = get_settings()
    metrics_df, best_model = run_training_pipeline(settings=settings, refresh_data=refresh_data)
    return TrainResponse(
        selected_model=best_model,
        model_path=str(settings.model_path),
        metrics_path=str(settings.metrics_path),
        candidates_evaluated=int(len(metrics_df)),
    )


@app.get("/forecast", response_model=ForecastResponse)
def latest_forecast(model_name: str = "best_model") -> ForecastResponse:
    settings = get_settings()
    if not settings.model_path.exists():
        raise HTTPException(status_code=400, detail="Model not found. Run training first.")
    forecast = generate_latest_forecast(settings=settings, model_name=model_name)
    return ForecastResponse(
        forecast_date=forecast.target_date.date().isoformat(),
        prediction_mom_percent=forecast.prediction_mom_percent,
        prediction_annualized_percent=forecast.prediction_annualized_percent,
        model_name=forecast.model_name,
    )
