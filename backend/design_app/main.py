from fastapi import FastAPI, HTTPException, BackgroundTasks
from . import schemas
from .service import DesignService
from fastapi.middleware.cors import CORSMiddleware
import uuid
from typing import Dict

app = FastAPI(
    title="BGA-AI Design API",
    description="API for finding optimal BGA process parameters using DDQN.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

designer = DesignService()
tasks: Dict[str, Dict] = {}

def run_design_c_in_background(task_id: str, inputs: schemas.DesignInput):
    try:
        result = designer.run_design_c(inputs)
        tasks[task_id] = {"status": "completed", "result": result.dict()}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

def run_design_s_in_background(task_id: str, inputs: schemas.DesignInput):
    try:
        result = designer.run_design_s(inputs)
        tasks[task_id] = {"status": "completed", "result": result.dict()}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/design/convex", response_model=schemas.DesignTaskSubmission, status_code=202, tags=["Design"])
async def design_convex_parameters(inputs: schemas.DesignInput, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(run_design_c_in_background, task_id, inputs)
    return {"task_id": task_id, "message": "AI design task has been started in the background."}

@app.get("/design/status/{task_id}", response_model=schemas.DesignTaskResult, tags=["Design"])
async def get_design_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/design/concave", response_model=schemas.DesignTaskSubmission, status_code=202, tags=["Design"])
async def design_concave_parameters(inputs: schemas.DesignInput, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(run_design_s_in_background, task_id, inputs)
    return {"task_id": task_id, "message": "AI design task has been started in the background."}
