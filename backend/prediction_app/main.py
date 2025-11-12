from fastapi import FastAPI, HTTPException
from . import schemas
from .service import PredictionService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="BGA-AI Prediction API",
    description="API for predicting BGA warpage.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PredictionService()

@app.post("/predict/convex", response_model=schemas.PredictionOutput, tags=["Prediction"])
async def predict_convex_warpage(inputs: schemas.PredictionInputC):
    try:
        result = predictor.run_prediction_c(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/concave", response_model=schemas.PredictionOutput, tags=["Prediction"])
async def predict_concave_warpage(inputs: schemas.PredictionInputS):
    try:
        result = predictor.run_prediction_s(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
