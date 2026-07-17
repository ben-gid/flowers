import torch
from fastapi import APIRouter, Depends, HTTPException
from torch._tensor import Tensor
from torch.nn import functional as F

from ..core.config import state
from ..models import PredictionResponse, ValidatedImage
from ..utils.dependencies import validate_and_convert_file

router = APIRouter()

@router.post("/classify", response_model=PredictionResponse)
async def classify(
    valid_img: ValidatedImage = Depends(validate_and_convert_file)
):
    if state.transform is None:
        raise HTTPException(
            status_code=500,
            detail="app state transform wasn't loaded"
        )
        
    if state.classifier is None:
        raise HTTPException(
            status_code=500,
            detail="app state classifier wasn't loaded"
        )
        
    if state.class_names is None:
        raise HTTPException(
            status_code=500,
            detail="app state class_names weren't loaded"
        )
    
    # get device
    device = next(state.classifier.parameters()).device

    # transform image to tensor and add batch dimension and move to device
    transform_img = state.transform(valid_img.image).unsqueeze(0).to(device) # type: ignore

    # make prediction
    with torch.no_grad():
        prediction: Tensor = state.classifier(transform_img)

    # get class name
    classification = int(prediction.argmax().item())
    class_name = state.class_names[classification]

    # get confidence
    confidence = F.softmax(prediction, dim=1)[0][classification].item()
    
    return PredictionResponse(
        filename=valid_img.filename,
        content_type=valid_img.content_type,
        prediction=class_name,
        confidence=round(confidence, 4),
    )