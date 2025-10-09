#!/usr/bin/env python3
"""
AutoSOS YOLOv8 Service
Cloud-deployed motorcycle diagnostic detection using YOLOv8 + Supabase model download
Optimized for Render free services
"""

import os
import time
import base64
import logging
from typing import Dict, Any, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# Redis + Supabase
import redis
from supabase import create_client

# =========================
# Logging Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("autosos-yolov8")

# =========================
# Config
# =========================
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
MODEL_FILENAME = "motorcycle_diagnostics.pt"
MODEL_PATH = os.path.join(MODEL_CACHE_DIR, MODEL_FILENAME)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# =========================
# Globals
# =========================
yolo_model = None
supabase_client = None
redis_client = None

# Detection classes
CLASS_NAMES = {
    0: "broken_headlights_tail_lights",
    1: "broken_side_mirror",
    2: "flat_tire",
    3: "oil_leak"
}

CLASS_DISPLAY_NAMES = {
    0: "Broken Headlights/Tail Lights",
    1: "Broken Side Mirror",
    2: "Flat Tire",
    3: "Oil Leak"
}

CLASS_COLORS = {
    0: (255, 255, 0),
    1: (255, 165, 0),
    2: (0, 0, 255),
    3: (128, 0, 128)
}

# =========================
# Utility: Supabase download
# =========================
def download_model_from_supabase():
    """Download custom YOLOv8 model from Supabase storage if not found locally."""
    global supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not set, skipping download")
        return False

    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Downloading YOLOv8 model from Supabase Storage...")

    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        response = supabase_client.storage.from_("autosos").download(
            f"models/yolov8/{MODEL_FILENAME}"
        )
        if response:
            with open(MODEL_PATH, "wb") as f:
                f.write(response)
            logger.info(f"✅ Model downloaded successfully to {MODEL_PATH}")
            return True
        else:
            logger.error("❌ Failed to download model: No response")
            return False
    except Exception as e:
        logger.error(f"❌ Error downloading model from Supabase: {e}")
        return False

# =========================
# Utility: Process detection results
# =========================
def process_detections(results, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    detections = []
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].cpu().numpy().item()
                cls = int(box.cls[0].cpu().numpy().item())

                if conf >= confidence_threshold:
                    detections.append({
                        "bbox": bbox,
                        "confidence": float(conf),
                        "class": cls,
                        "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                        "display_name": CLASS_DISPLAY_NAMES.get(cls, f"Class {cls}"),
                        "color": CLASS_COLORS.get(cls, (255, 255, 255))
                    })
    return detections

# =========================
# Utility: Annotated image
# =========================
def create_annotated_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    import cv2
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = det["color"]
        label = f"{det['display_name']}: {det['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            annotated,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    _, buffer = cv2.imencode('.jpg', annotated)
    return base64.b64encode(buffer).decode('utf-8')

# =========================
# Lifespan - startup/shutdown
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, redis_client

    logger.info("🚀 Starting YOLOv8 motorcycle diagnostic service")

    # Init Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis client initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}")
        redis_client = None

    # Load YOLO model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Custom model not found at {MODEL_PATH}, attempting download...")
        if not download_model_from_supabase():
            logger.warning("Falling back to YOLOv8 nano model")
            yolo_model = YOLO("yolov8n.pt")
        else:
            yolo_model = YOLO(MODEL_PATH)
    else:
        yolo_model = YOLO(MODEL_PATH)

    logger.info(f"✅ YOLO model loaded: {MODEL_PATH if os.path.exists(MODEL_PATH) else 'yolov8n.pt'}")
    yield
    logger.info("🛑 Shutting down YOLOv8 service")

# =========================
# FastAPI App
# =========================
app = FastAPI(
    title="AutoSOS YOLOv8 Motorcycle Diagnostics",
    description="Detects motorcycle issues using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Routes
# =========================
@app.get("/health")
async def health_check():
    if yolo_model is None:
        return JSONResponse(status_code=503, content={"status": "unhealthy"})
    return {"status": "healthy", "model_loaded": True, "timestamp": time.time()}

@app.get("/model-info")
async def model_info():
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_loaded": True,
        "model_type": str(type(yolo_model)),
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_classes": getattr(yolo_model, 'names', None)
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...), confidence: float = 0.5, include_annotated_image: bool = True):
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import cv2
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        start = time.time()
        results = yolo_model(image, conf=confidence, verbose=False)
        detections = process_detections(results, confidence)
        detection_time = time.time() - start

        response = {
            "success": True,
            "detection_count": len(detections),
            "detections": detections,
            "detection_time": detection_time,
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }

        if include_annotated_image and detections:
            response["annotated_image"] = create_annotated_image(image, detections)

        return response
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# =========================
# Entrypoint for local dev
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
