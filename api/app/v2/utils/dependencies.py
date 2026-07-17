import io

from fastapi import File, HTTPException, UploadFile
from PIL import Image

import logging

from ..models import ValidatedImage

logger = logging.getLogger("api_v2")


MIN_IMG_SIZE = 224
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5mb limit
IMG_EXT = ["image/jpeg", "image/png", "image/jpg"]

async def validate_and_convert_file(
    file: UploadFile = File(
        description="Upload an image of a flower",
        content_type=IMG_EXT,
    ),
) -> ValidatedImage:
    
    # validate file extension
    if file.content_type not in IMG_EXT:
        logger.warning(f"Invalid file type. Allowed: {IMG_EXT}")
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {IMG_EXT}"
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
    except Exception as e:
        logger.warning(f"Invalid image file: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {IMG_EXT}"
        ) from e
        
    # validate image size
    if img.size[0] < MIN_IMG_SIZE or img.size[1] < MIN_IMG_SIZE:
        logger.warning(
            f"Image too small. Minimum size: {MIN_IMG_SIZE}x{MIN_IMG_SIZE}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Image too small. Minimum size: {MIN_IMG_SIZE}x{MIN_IMG_SIZE}",
        )
        
    return ValidatedImage(
        image=img,
        filename=file.filename or "unknown",
        content_type=file.content_type,
    )
