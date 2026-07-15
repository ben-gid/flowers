from PIL import Image
from pydantic import BaseModel, ConfigDict


class PredictionResponse(BaseModel):
    filename: str
    content_type: str
    prediction: str
    confidence: float
    
class ValidatedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    image: Image.Image
    filename: str
    content_type: str