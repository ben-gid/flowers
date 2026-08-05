from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core.config import state
from .core.logging_config import build_log_config, init_logger
from .core.middleware import log_responses
from .routers import classify, system

logger = init_logger("api_v2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load(logger=logger)
    yield

    state.clear(logger=logger)


app = FastAPI(
    title="Flower Classifier API V2",
    description="Classify 102 different flower types using a finetuned vision model",
    version="2.0.0",
    lifespan=lifespan,
)

app.middleware("http")(log_responses)
app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)
app.include_router(classify.router)
app.include_router(system.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=build_log_config())
