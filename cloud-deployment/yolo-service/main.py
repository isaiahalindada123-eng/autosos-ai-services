#!/usr/bin/env python3
"""
YOLOv8 Service for AutoSOS
Cloud-deployed motorcycle diagnostic service
Aligned with local test structure
"""

import os
import time
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV imported successfully")
except ImportError as e:
    print(f"OpenCV import failed: {e}")
    OPENCV_AVAILABLE = False
    # Create a dummy cv2 module for fallback
    class DummyCV2:
        @staticmethod
        def imdecode(data, flags):
            return None
        @staticmethod
        def imencode(ext, img):
            return (False, None)
        @staticmethod
        def cvtColor(img, code):
            return img
        COLOR_BGR2RGB = 4
        IMREAD_COLOR = 1
        FONT_HERSHEY_SIMPLEX = 0
    cv2 = DummyCV2()

import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from supabase import create_client, Client

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Redis with error handling
redis_client = None
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Test the connection
    redis_client.ping()
    logger.info("Redis client initialized successfully")
except Exception as e:
    logger.warning(f"Redis client initialization failed: {e}")
    redis_client = None

# Class names for motorcycle issues (aligned with local test)
CLASS_NAMES = {
    0: "broken_headlights_tail_lights",
    1: "broken_side_mirror", 
    2: "flat_tire",
    3: "oil_leak"
}

# Class display names
CLASS_DISPLAY_NAMES = {
    0: "Broken Headlights/Tail Lights",
    1: "Broken Side Mirror",
    2: "Flat Tire", 
    3: "Oil Leak"
}

# Class colors for visualization
CLASS_COLORS = {
    0: (255, 255, 0),    # Yellow
    1: (255, 165, 0),    # Orange
    2: (0, 0, 255),      # Red
    3: (128, 0, 128)     # Purple
}

# Global YOLO model instance
yolo_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global yolo_model
    
    logger.info("Starting YOLOv8 Service")
    
    try:
        # Create models directory
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        
        # Initialize YOLOv8 model
        model_path = os.path.join(MODEL_CACHE_DIR, "motorcycle_diagnostic.pt")
        local_model_path = "motorcycle_diagnostic.pt"
        
        # Try to load local model first
        if os.path.exists(local_model_path):
            yolo_model = YOLO(local_model_path)
            logger.info("YOLOv8 model loaded from local file")
        elif os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            logger.info("YOLOv8 model loaded from cache")
        else:
            # Download model from Supabase Storage if not exists locally
            download_yolo_model_from_supabase(model_path)
            
            if os.path.exists(model_path):
                yolo_model = YOLO(model_path)
                logger.info("YOLOv8 model loaded from Supabase Storage")
            else:
                logger.info("Downloading fallback YOLOv8 model...")
                yolo_model = YOLO("yolov8n.pt")  # Use YOLOv8 nano as fallback
                yolo_model.save(model_path)
                # Upload to Supabase Storage
                upload_yolo_model_to_supabase(model_path)
        
        logger.info("YOLOv8 model initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize YOLOv8 model", error=str(e))
        # Fallback to nano model
        yolo_model = YOLO("yolov8n.pt")
        logger.info("Using fallback YOLOv8 nano model")
    
    yield
    
    logger.info("Shutting down YOLOv8 Service")

# Supabase Storage functions
def download_yolo_model_from_supabase(model_path: str):
    """Download YOLOv8 model from Supabase Storage"""
    if not supabase_client:
        logger.warning("Supabase client not initialized, skipping model download")
        return
    
    try:
        # Try to download the model from Supabase Storage
        response = supabase_client.storage.from_("autosos").download("yolo_models/motorcycle_diagnostic.pt")
        
        if response:
            # Save the model to local path
            with open(model_path, 'wb') as f:
                f.write(response)
            logger.info("YOLOv8 model downloaded from Supabase Storage")
        else:
            logger.warning("No YOLOv8 model found in Supabase Storage")
        
    except Exception as e:
        logger.error(f"Failed to download YOLOv8 model from Supabase: {e}")

def upload_yolo_model_to_supabase(model_path: str):
    """Upload YOLOv8 model to Supabase Storage"""
    if not supabase_client:
        logger.warning("Supabase client not initialized, skipping model upload")
        return
    
    try:
        # Read the model file
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Upload to Supabase Storage
        supabase_client.storage.from_("autosos").upload(
            "yolo_models/motorcycle_diagnostic.pt",
            model_data,
            {"content-type": "application/octet-stream"}
        )
        logger.info("YOLOv8 model uploaded to Supabase Storage")
        
    except Exception as e:
        logger.error(f"Failed to upload YOLOv8 model to Supabase: {e}")

# Prometheus metrics
REQUEST_COUNT = Counter('yolo_requests_total', 'Total YOLO requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('yolo_request_duration_seconds', 'YOLO request duration')

# FastAPI app
app = FastAPI(
    title="AutoSOS YOLOv8 Service",
    description="Motorcycle diagnostic detection service using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_detection_results(results, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Process YOLOv8 detection results"""
    detections = []
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                # Get box data
                if hasattr(box, 'xyxy'):
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    
                    # Filter by confidence
                    if conf >= confidence_threshold:
                        # Get class name
                        class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                        display_name = CLASS_DISPLAY_NAMES.get(cls, f"Class {cls}")
                        
                        detection = {
                            "bbox": bbox,
                            "confidence": float(conf),
                            "class": cls,
                            "class_name": class_name,
                            "display_name": display_name,
                            "color": CLASS_COLORS.get(cls, (255, 255, 255))
                        }
                        detections.append(detection)
    
    return detections

def create_annotated_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Create annotated image with detection boxes"""
    annotated_image = image.copy()
    
    for detection in detections:
        bbox = detection["bbox"]
        conf = detection["confidence"]
        class_name = detection["display_name"]
        color = detection["color"]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/health")
async def health_check():
    """Service health check"""
    if yolo_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "YOLOv8 model not initialized"}
        )
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": yolo_model is not None,
        "model_name": getattr(yolo_model, 'model_name', 'motorcycle_diagnostic_v1') if yolo_model else None
    }

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    if yolo_model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "YOLOv8 model not initialized"}
        )
    
    try:
        info = {
            "model_loaded": True,
            "model_type": str(type(yolo_model)),
            "class_names": CLASS_NAMES,
            "class_display_names": CLASS_DISPLAY_NAMES,
            "num_classes": len(CLASS_NAMES)
        }
        
        # Try to get model names/classes
        if hasattr(yolo_model, 'names'):
            info['model_classes'] = yolo_model.names
            info['model_num_classes'] = len(yolo_model.names)
        
        return info
        
    except Exception as e:
        logger.error("Failed to get model information", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get model information"}
        )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/detect")
async def detect_motorcycle_issues(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    include_annotated_image: bool = True
):
    """Detect motorcycle issues using YOLOv8"""
    REQUEST_COUNT.labels(method="POST", endpoint="/detect", status="200").inc()
    
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 model not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run YOLOv8 detection
        start_time = time.time()
        results = yolo_model(image, conf=confidence, verbose=False)
        detection_time = time.time() - start_time
        
        # Process results
        detections = process_detection_results(results, confidence)
        
        # Create response
        response_data = {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "confidence_threshold": confidence,
            "detection_time": detection_time,
            "processing_time": time.time()
        }
        
        # Add annotated image if requested
        if include_annotated_image and detections:
            annotated_image = create_annotated_image(image, detections)
            response_data["annotated_image"] = annotated_image
        
        # Cache the result (with error handling)
        if redis_client is not None:
            try:
                if hasattr(image_data, 'tobytes'):
                    cache_key = f"yolo:detect:{hash(image_data.tobytes())}"
                else:
                    cache_key = f"yolo:detect:{hash(image_data)}"
                
                redis_client.setex(
                    cache_key,
                    3600,  # 1 hour
                    json.dumps(response_data)
                )
            except Exception as cache_error:
                logger.warning("Failed to cache result", error=str(cache_error))
        
        logger.info("YOLOv8 detection completed", 
                   detection_count=len(detections),
                   confidence_threshold=confidence,
                   detection_time=detection_time)
        
        return response_data
        
    except Exception as e:
        logger.error("YOLOv8 detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/detect-base64")
async def detect_motorcycle_issues_base64(
    data: Dict[str, Any]
):
    """Detect motorcycle issues from base64 encoded image"""
    REQUEST_COUNT.labels(method="POST", endpoint="/detect-base64", status="200").inc()
    
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 model not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image_data"])
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        confidence = data.get("confidence", 0.5)
        include_annotated_image = data.get("include_annotated_image", True)
        
        # Run YOLOv8 detection
        start_time = time.time()
        results = yolo_model(image, conf=confidence, verbose=False)
        detection_time = time.time() - start_time
        
        # Process results
        detections = process_detection_results(results, confidence)
        
        # Create response
        response_data = {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": {
                "width": image.shape[1],
                "height": image.shape[0]
            },
            "confidence_threshold": confidence,
            "detection_time": detection_time,
            "processing_time": time.time()
        }
        
        # Add annotated image if requested
        if include_annotated_image and detections:
            annotated_image = create_annotated_image(image, detections)
            response_data["annotated_image"] = annotated_image
        
        logger.info("YOLOv8 detection completed", 
                   detection_count=len(detections),
                   confidence_threshold=confidence,
                   detection_time=detection_time)
        
        return response_data
        
    except Exception as e:
        logger.error("YOLOv8 detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/classes")
async def get_class_info():
    """Get information about detection classes"""
    return {
        "classes": CLASS_NAMES,
        "display_names": CLASS_DISPLAY_NAMES,
        "colors": CLASS_COLORS,
        "total_classes": len(CLASS_NAMES)
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if yolo_model is None:
        raise HTTPException(status_code=503, detail="YOLOv8 model not initialized")
    
    return {
        "model_loaded": True,
        "model_type": str(type(yolo_model)),
        "class_names": CLASS_NAMES,
        "class_display_names": CLASS_DISPLAY_NAMES,
        "num_classes": len(CLASS_NAMES),
        "model_classes": yolo_model.names if hasattr(yolo_model, 'names') else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)