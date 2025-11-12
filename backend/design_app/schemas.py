from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DesignInput(BaseModel):
    target_warpage_um: float = Field(..., example=25.0, description="目標翹曲值 (μm)")
    substrate: int = Field(..., example=55, description="Substrate規格 (mm)")
    copper: int = Field(..., example=100, description="Copper Ratio (%)")
    sbthk_vals: List[float] = Field(..., min_items=33, max_items=33, description="Substrate層數厚度 (33個數值)")
    material_vals: List[float] = Field(..., min_items=7, max_items=7, description="Substrate材料參數 (7個數值)")

class BestParameters(BaseModel):
    tool_height: Optional[float] = Field(None, description="Tool高度 (mm), 僅在凸面設計中提供")
    magnet: int
    jig: float
    b1: int
    w1: int

class DesignOutput(BaseModel):
    achieved_warpage_um: float = Field(..., description="AI找到的最佳參數所對應的翹曲值 (μm)")
    best_parameters: BestParameters

class DesignTaskSubmission(BaseModel):
    task_id: str = Field(..., description="背景任務的唯一ID")
    message: str = Field(..., example="AI design task has been started in the background.")

class DesignTaskResult(BaseModel):
    status: str = Field(..., example="processing", description="任務狀態 (processing, completed, failed)")
    result: DesignOutput | None = Field(None, description="設計任務的結果，僅在 status 為 'completed' 時可用")
    error: str | None = Field(None, description="如果任務失敗，則為錯誤訊息")
