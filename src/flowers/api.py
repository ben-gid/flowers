from contextlib import asynccontextmanager
from fastapi import FastAPI, Body, File, UploadFile, HTTPException
from pathlib import Path
import os
from PIL import Image
import io
import torch
from torch.nn import functional as F
from torch._tensor import Tensor
from pydantic import BaseModel
import logging
import sys

# Add 'src' directory to sys.path
# Path(__file__).parent is src/flowers/
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .train import init_model, get_transforms
except (ImportError, ValueError):
    from train import init_model, get_transforms

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

MIN_IMG_SIZE = 224
MAX_FILE_SIZE = 5 * 1024 * 1024 # 5mb limit
IMG_EXT = {"image/jpeg", "image/png", "image/jpg"}

class PredictionResponse(BaseModel):
    filename: str
    content_type: str
    prediction: str
    confidence: float

@asynccontextmanager
async def lifespan(app:FastAPI):
    # Get paths from env or use defaults
    # Path(__file__).parent is src/flowers/
    # We want root which is parent.parent.parent
    project_root = Path(__file__).parent.parent.parent
    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
    model_path = Path(os.getenv("MODEL_PATH", project_root / "flower_model_weights.pth"))
    
    # get model classes
    logger.info(f"Loading Class Names from {data_root}")
    # We only need get_classes, we don't need the full Dataset instance logic
    try:
        classes_path = data_root / "Oxford-102_Flower_dataset_labels.txt"
        if not classes_path.exists():
            logger.error(f"Class names file NOT FOUND at {classes_path}. Please ensure it is present.")
            app.state.class_names = [f"Class {i}" for i in range(102)]
        else:
            with open(classes_path) as f:
                app.state.class_names = f.read().splitlines()
            logger.info("Class Names loaded")
    except Exception as e:
        logger.error(f"Failed to load class names: {e}")
        app.state.class_names = [f"Class {i}" for i in range(102)]
    
    # load model on startup
    logger.info(f"Loading CNN model from {model_path}...")
    model, _, _ = init_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    # set model to eval mode
    model.eval()
    
    logger.info("Loading Transform...")
    _, val_transform = get_transforms()
    
    app.state.model = model
    logger.info("Model loaded")
    app.state.transform = val_transform
    logger.info("Transform loaded")
    yield
    
    del app.state.class_names
    del app.state.model
    del app.state.transform
    logger.info("Model, Transform, and Class Names cleared from memory.")    

app = FastAPI(lifespan=lifespan)   

@app.post(
    "/predict", 
    summary="Predict flower image", 
    description="Upload an image of a flower and get the predicted class", 
    response_model=PredictionResponse,
    responses={200: {
        "description": "Prediction successful", 
        "content": {
            "application/json": {
                "example": {
                    "filename": "rose.jpg", "content_type": "image/jpeg", "prediction": "rose"
                    }
                }
            }
        }, 
        400: {"description": "Invalid file type"}, 
        413: {"description": "File too large"},
        500: {"description": "Internal server error"}
    },
    
    )
async def predict(file: UploadFile = File(description="Upload an image of a flower", content_type="image/jpeg, image/png, image/jpg")):
    # validate file extension
    if file.content_type not in IMG_EXT:
        logger.warning(f"Invalid file type. Allowed: {IMG_EXT}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {IMG_EXT}"
        )
        
    # validate file size
    if file.size is not None and file.size > MAX_FILE_SIZE:
        logger.warning(f"File too large. Maximum size: {MAX_FILE_SIZE}")
        raise HTTPException(status_code=413, detail="File too large")
    
    # read file to memory
    file_contents = await file.read()
    
    # convert to PIL image
    try:
        img = Image.open(io.BytesIO(file_contents)).convert("RGB")
        
        # validate image size
        if img.size[0] < MIN_IMG_SIZE or img.size[1] < MIN_IMG_SIZE:
            logger.warning(f"Image too small. Minimum size: {MIN_IMG_SIZE}x{MIN_IMG_SIZE}")
            raise HTTPException(status_code=400, detail=f"Image too small. Minimum size: {MIN_IMG_SIZE}x{MIN_IMG_SIZE}")
    except Exception as e:
        logger.warning(f"Invalid image file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # get device
    device = next(app.state.model.parameters()).device
    
    # transform image to tensor and add batch dimension and move to device
    transform_img = app.state.transform(img).unsqueeze(0).to(device)
    
    # make prediction
    with torch.no_grad():
        prediction: Tensor = app.state.model(transform_img)
    
    # get class name
    classification = int(prediction.argmax().item())
    class_name = app.state.class_names[classification]
    
    # get confidence
    confidence = F.softmax(prediction, dim=1)[0][classification].item()
    
    return PredictionResponse(
        filename=file.filename or "unknown",
        content_type=file.content_type,
        prediction=class_name,
        confidence=confidence
    )
    
@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": hasattr(app.state, "model"), 
        "class_names_loaded": hasattr(app.state, "class_names"), 
        "transform_loaded": hasattr(app.state, "transform")
    }
