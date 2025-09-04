from fastapi import FastAPI, HTTPException
from . import schemas
from .prediction_service import PredictionService

# 建立 FastAPI 應用
app = FastAPI(
    title="BGA-AI Prediction API",
    description="API for predicting BGA warpage.",
    version="1.0.0"
)

# 實例化服務，模型會在此時載入記憶體
predictor = PredictionService()

@app.post("/predict/convex", response_model=schemas.PredictionOutput, tags=["Prediction"])
async def predict_convex_warpage(inputs: schemas.PredictionInputC):
    """接收製程參數，預測凸面 (Convex) 翹曲"""
    try:
        result = predictor.run_prediction_c(inputs)
        return result
    except Exception as e:
        # 捕捉潛在的錯誤，並以標準的 HTTP 錯誤格式回傳
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/concave", response_model=schemas.PredictionOutput, tags=["Prediction"])
async def predict_concave_warpage(inputs: schemas.PredictionInputS):
    """接收製程參數，預測凹面 (Concave) 翹曲"""
    try:
        result = predictor.run_prediction_s(inputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
