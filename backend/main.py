from fastapi import FastAPI, HTTPException
from . import schemas
from .prediction_service import PredictionService
from fastapi.middleware.cors import CORSMiddleware

# 建立 FastAPI 應用
app = FastAPI(
    title="BGA-AI Prediction API",
    description="API for predicting BGA warpage.",
    version="1.0.0"
)

# --- 設定 CORS 中介軟體 ---
# origins 列表可以是 "*" (允許所有來源)，或是一個包含您前端 URL 的列表
# 例如：["http://localhost", "http://localhost:8080"]
# 在開發階段，使用 "*" 是最方便的。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有 HTTP 標頭
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
