from fastapi import APIRouter

from ..core.config import state
from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_name=state.model_name,
        model_repo=state.model_repo,
        model_loaded=state.classifier is not None,
        class_names_loaded=state.class_names is not None,
        transform_loaded=state.transform is not None,
    )
