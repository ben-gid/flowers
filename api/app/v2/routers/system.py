from fastapi import APIRouter

from ..core.config import state

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(state, "ft_model"),
        "class_names_loaded": hasattr(state, "class_names"),
        "transform_loaded": hasattr(state, "transform"),
    }
