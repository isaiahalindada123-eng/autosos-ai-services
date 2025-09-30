#!/usr/bin/env python3
"""
YOLOv8 Service for AutoSOS
Cloud-deployed motorcycle diagnostic service
"""

import os
import time
import base64
import json
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import cv2
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

# Prometheus metrics
REQUEST_COUNT = Counter('yolo_requests_total', 'Total YOLOv8 requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('yolo_request_duration_seconds', 'YOLOv8 request duration', ['method', 'endpoint'])

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase_client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Class names for motorcycle issues
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
        
        # Download model from Supabase Storage if not exists locally
        if not os.path.exists(model_path):
            await download_yolo_model_from_supabase(model_path)
        
        # Load model
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            logger.info("YOLOv8 model loaded from Supabase Storage")
        else:
            logger.info("Downloading fallback YOLOv8 model...")
            yolo_model = YOLO("yolov8n.pt")  # Use nano model for cloud deployment
            yolo_model.save(model_path)
            # Upload to Supabase Storage
            await upload_yolo_model_to_supabase(model_path)
        
        logger.info("YOLOv8 model initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize YOLOv8 model", error=str(e))
        # Fallback to nano model
        yolo_model = YOLO("yolov8n.pt")
        logger.info("Using fallback YOLOv8 nano model")
    
    yield
    
    logger.info("Shutting down YOLOv8 Service")

# Supabase Storage functions
async def download_yolo_model_from_supabase(model_path: str):
    """Download YOLOv8 model from Supabase Storage"""
    if not supabase_client:
        logger.warning("Supabase client not initialized, skipping model download")
        return
    
    try:
        logger.info("Downloading YOLOv8 model from Supabase Storage...")
        
        # Try to download custom trained model first
        model_files = [
            "autosos/models/yolov8/motorcycle_diagnostic.pt",
            "autosos/models/yolov8/best.pt",
            "autosos/models/yolov8/yolov8n.pt"
        ]
        
        for model_file in model_files:
            try:
                response = supabase_client.storage.from_("autosos").download(model_file)
                if response:
                    with open(model_path, 'wb') as f:
                        f.write(response)
                    logger.info(f"Downloaded {model_file} to {model_path}")
                    return
            except Exception as e:
                logger.warning(f"Failed to download {model_file}: {e}")
                continue
        
        logger.warning("No YOLOv8 model found in Supabase Storage")
        
    except Exception as e:
        logger.error(f"Failed to download YOLOv8 model from Supabase: {e}")

async def upload_yolo_model_to_supabase(model_path: str):
    """Upload YOLOv8 model to Supabase Storage"""
    if not supabase_client:
        logger.warning("Supabase client not initialized, skipping model upload")
        return
    
    try:
        logger.info("Uploading YOLOv8 model to Supabase Storage...")
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            supabase_client.storage.from_("autosos").upload(
                "autosos/models/yolov8/motorcycle_diagnostic.pt",
                model_data,
                {"content-type": "application/octet-stream"}
            )
            logger.info("Uploaded YOLOv8 model to Supabase Storage")
            
    except Exception as e:
        logger.error(f"Failed to upload YOLOv8 model to Supabase: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="AutoSOS YOLOv8 Service",
    description="Motorcycle diagnostic service using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_detection_results(results, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Process YOLOv8 detection results"""
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                if confidence >= confidence_threshold:
                    detection = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                        "class_display_name": CLASS_DISPLAY_NAMES.get(class_id, f"Class {class_id}"),
                        "severity": get_severity_level(class_id, confidence)
                    }
                    detections.append(detection)
    
    return detections

def get_severity_level(class_id: int, confidence: float) -> str:
    """Determine severity level based on class and confidence"""
    # Critical issues
    if class_id in [2, 3]:  # flat_tire, oil_leak
        return "Critical" if confidence > 0.8 else "High"
    
    # High severity issues
    if class_id == 0:  # broken_headlights_tail_lights
        return "High" if confidence > 0.7 else "Medium"
    
    # Medium severity issues
    if class_id == 1:  # broken_side_mirror
        return "Medium" if confidence > 0.6 else "Low"
    
    return "Low"

def create_annotated_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> str:
    """Create annotated image with bounding boxes"""
    annotated_image = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        class_name = detection["class_display_name"]
        
        # Get color for this class
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(annotated_image, label, (int(x1), int(y1) - 5), 
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
        "service": "yolo",
        "timestamp": time.time(),
        "model_loaded": yolo_model is not None
    }

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
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run YOLOv8 detection
        results = yolo_model(image, conf=confidence, verbose=False)
        
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
            "processing_time": time.time()
        }
        
        # Add annotated image if requested
        if include_annotated_image and detections:
            annotated_image = create_annotated_image(image, detections)
            response_data["annotated_image"] = annotated_image
        
        # Cache the result
        cache_key = f"yolo:detect:{hash(image_data.tobytes())}"
        await redis_client.setex(
            cache_key,
            3600,  # 1 hour
            json.dumps(response_data)
        )
        
        logger.info("YOLOv8 detection completed", 
                   detection_count=len(detections),
                   confidence_threshold=confidence)
        
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
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        confidence = data.get("confidence", 0.5)
        include_annotated_image = data.get("include_annotated_image", True)
        
        # Run YOLOv8 detection
        results = yolo_model(image, conf=confidence, verbose=False)
        
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
            "processing_time": time.time()
        }
        
        # Add annotated image if requested
        if include_annotated_image and detections:
            annotated_image = create_annotated_image(image, detections)
            response_data["annotated_image"] = annotated_image
        
        logger.info("YOLOv8 detection completed", 
                   detection_count=len(detections),
                   confidence_threshold=confidence)
        
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
        "model_type": "YOLOv8",
        "classes": list(CLASS_NAMES.values()),
        "input_size": "640x640",
        "framework": "PyTorch"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
