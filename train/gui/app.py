import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Make repo root importable
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "train"))

from config import (
    OPTIMIZER_REGISTRY,
    PRETRAINED_MODEL_REGISTRY,
    SCHEDULER_REGISTRY,
    VALID_PRECISION,
    TrainConfig,
)

app = FastAPI(title="Flowers Training GUI")

gui_dir = BASE_DIR / "train" / "gui"
app.mount("/static", StaticFiles(directory=gui_dir / "static"), name="static")
templates = Jinja2Templates(directory=gui_dir / "templates")

# Basic flash message mockup via query parameters
def get_flashed_messages(request: Request, with_categories=False):
    msg = request.query_params.get("msg")
    cat = request.query_params.get("cat", "info")
    if not msg:
        return []
    if with_categories:
        return [(cat, msg)]
    return [msg]

templates.env.globals["get_flashed_messages"] = get_flashed_messages

DB_PATH = BASE_DIR / "mlflow.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def format_timestamp(ts):
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

@app.get("/", response_class=HTMLResponse)
def train_get(request: Request):
    defaults = TrainConfig()
    choices = {
        "models": list(PRETRAINED_MODEL_REGISTRY.keys()),
        "optimizers": list(OPTIMIZER_REGISTRY.keys()),
        "schedulers": list(SCHEDULER_REGISTRY.keys()),
        "precisions": VALID_PRECISION,
    }
    return templates.TemplateResponse(request=request, name="train.html", context={
        "active": "train",
        "defaults": defaults,
        "choices": choices,
    })

@app.post("/")
def train_post(
    request: Request,
    exp_name: str = Form(...),
    pretrained_model: str = Form(...),
    optimizer: str = Form(...),
    scheduler: str = Form(...),
    precision: str = Form(...),
    lr_head_stage_1: float = Form(...),
    lr_head_stage_2: float = Form(...),
    lr_backbone: float = Form(...),
    max_epochs: int = Form(...),
    batch_size: int = Form(...),
    unfreeze_at_epoch: int = Form(...),
    accumulate_grad_batches: int = Form(...),
    early_stopping_patience: int = Form(...),
):
    cmd = [
        sys.executable, str(BASE_DIR / "train" / "cli.py"), "train",
        "--exp-name", exp_name,
        "--pretrained-model", pretrained_model,
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--precision", precision,
        "--lr-head-stage-1", str(lr_head_stage_1),
        "--lr-head-stage-2", str(lr_head_stage_2),
        "--lr-backbone", str(lr_backbone),
        "--max-epochs", str(max_epochs),
        "--batch-size", str(batch_size),
        "--unfreeze-at-epoch", str(unfreeze_at_epoch),
        "--accumulate-grad-batches", str(accumulate_grad_batches),
        "--early-stopping-patience", str(early_stopping_patience),
    ]
    
    # Launch in background
    subprocess.Popen(cmd, cwd=str(BASE_DIR))
    
    return RedirectResponse(url="/runs?msg=Training+launched+successfully&cat=success", status_code=303)

@app.get("/runs", response_class=HTMLResponse)
def runs_list(request: Request):
    try:
        with get_db() as conn:
            runs = conn.execute(
                "SELECT run_uuid, name, status, start_time, end_time FROM runs ORDER BY start_time DESC LIMIT 100"
            ).fetchall()
            
        formatted_runs = []
        for r in runs:
            d = dict(r)
            d['start_time'] = format_timestamp(d['start_time'])
            d['end_time'] = format_timestamp(d['end_time'])
            formatted_runs.append(d)
    except sqlite3.OperationalError:
        formatted_runs = []
        
    return templates.TemplateResponse(request=request, name="runs.html", context={
        "active": "runs",
        "runs": formatted_runs,
    })

@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    with get_db() as conn:
        run = conn.execute("SELECT run_uuid, name, status FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
        
        params_rows = conn.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,)).fetchall()
        params = {r['key']: r['value'] for r in params_rows}
        
        metrics_rows = conn.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (run_id,)).fetchall()
        metrics = {r['key']: r['value'] for r in metrics_rows}
        
    return templates.TemplateResponse(request=request, name="run_detail.html", context={
        "active": "runs",
        "run": run,
        "params": params,
        "metrics": metrics,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
