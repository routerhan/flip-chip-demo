from pydantic import BaseModel, Field
from typing import List, Dict

class PredictionInputC(BaseModel):
    tool_height: float = Field(..., example=0.0, description="Tool高度 (mm)")
    magnet: int = Field(..., example=10, description="磁鐵數量")
    jig: float = Field(..., example=1.0, description="Jig厚度 (mm)")
    copper: int = Field(..., example=100, description="Copper Ratio (%)")
    b1: int = Field(..., example=40, description="Jig中心矩形孔 B1 (mm)")
    w1: int = Field(..., example=47, description="Jig中心矩形孔 W1 (mm)")
    substrate: int = Field(..., example=55, description="Substrate規格 (mm)")
    sbthk_vals: List[float] = Field(..., min_items=33, max_items=33, description="Substrate層數厚度 (33個數值)")
    material_vals: List[float] = Field(..., min_items=7, max_items=7, description="Substrate材料參數 (7個數值)")

class PredictionInputS(BaseModel):
    magnet: int = Field(..., example=10, description="磁鐵數量")
    jig: float = Field(..., example=1.0, description="Jig厚度 (mm)")
    copper: int = Field(..., example=100, description="Copper Ratio (%)")
    b1: int = Field(..., example=40, description="Jig中心矩形孔 B1 (mm)")
    w1: int = Field(..., example=47, description="Jig中心矩形孔 W1 (mm)")
    substrate: int = Field(..., example=55, description="Substrate規格 (mm)")
    sbthk_vals: List[float] = Field(..., min_items=33, max_items=33, description="Substrate層數厚度 (33個數值)")
    material_vals: List[float] = Field(..., min_items=7, max_items=7, description="Substrate材料參數 (7個數值)")

class PlotData(BaseModel):
    x: List[float] = Field(..., description="X 軸網格座標 (1D)")
    y: List[float] = Field(..., description="Y 軸網格座標 (1D)")
    z: List[List[float | None]] = Field(..., description="Z 軸高度值 (2D 網格)")

class PredictionOutput(BaseModel):
    warpage_um: float = Field(..., description="預測的翹曲值 (μm)")
    input_summary: Dict[str, float | int | str]
    plot_data: PlotData
