from PIL import Image
from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    filename: str
    content_type: str
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    # model_* fields collide with pydantic's protected namespace
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str | None
    model_repo: str | None
    model_loaded: bool
    class_names_loaded: bool
    transform_loaded: bool


class ValidatedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    filename: str
    content_type: str
