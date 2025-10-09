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
        logger.info("🚀 === YOLOv8 MODEL LOADING PROCESS STARTED ===")
        
        # Create models directory
        logger.info(f"📁 Creating models directory: {MODEL_CACHE_DIR}")
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        logger.info("✅ Models directory created successfully")
        
        # Initialize YOLOv8 model
        model_path = os.path.join(MODEL_CACHE_DIR, "motorcycle_diagnostic.pt")
        local_model_path = "motorcycle_diagnostic.pt"
        
        logger.info(f"🔍 Checking for local model at: {local_model_path}")
        logger.info(f"🔍 Checking for cached model at: {model_path}")
        
        # Try to load local model first
        if os.path.exists(local_model_path):
            logger.info(f"✅ Found local model file: {local_model_path}")
            logger.info(f"📊 Local model file size: {os.path.getsize(local_model_path)} bytes")
            yolo_model = YOLO(local_model_path)
            logger.info("✅ YOLOv8 motorcycle diagnostic model loaded from local file")
            logger.info(f"Model path: {local_model_path}")
        elif os.path.exists(model_path):
            logger.info(f"✅ Found cached model file: {model_path}")
            logger.info(f"📊 Cached model file size: {os.path.getsize(model_path)} bytes")
            yolo_model = YOLO(model_path)
            logger.info("✅ YOLOv8 motorcycle diagnostic model loaded from cache")
            logger.info(f"Model path: {model_path}")
        else:
            logger.info("❌ No local or cached model found")
            logger.info("🔄 Custom motorcycle model not found locally, downloading from Supabase Storage...")
            logger.info(f"📥 Attempting to download from: models/yolov8/motorcycle_diagnostic.pt")
            
            download_yolo_model_from_supabase(model_path)
            
            if os.path.exists(model_path):
                logger.info(f"✅ Model downloaded successfully to: {model_path}")
                logger.info(f"📊 Downloaded model file size: {os.path.getsize(model_path)} bytes")
                yolo_model = YOLO(model_path)
                logger.info("✅ YOLOv8 motorcycle diagnostic model loaded from Supabase Storage")
                logger.info(f"Model path: {model_path}")
            else:
                logger.error("❌ Failed to download model from Supabase Storage")
                logger.warning("⚠️ Custom motorcycle model not available, using fallback YOLOv8 nano model")
                logger.warning("⚠️ This model will NOT detect motorcycle issues - only general objects")
                logger.info("📥 Downloading default YOLOv8 nano model...")
                yolo_model = YOLO("yolov8n.pt")  # Use YOLOv8 nano as fallback
                logger.info("💾 Saving fallback model to cache...")
                yolo_model.save(model_path)
                logger.info("📤 Uploading fallback model to Supabase Storage...")
                upload_yolo_model_to_supabase(model_path)
        
        # Log model information
        logger.info("YOLOv8 model initialized successfully")
        logger.info(f"Model type: {type(yolo_model)}")
        if hasattr(yolo_model, 'names'):
            logger.info(f"Model classes: {yolo_model.names}")
            logger.info(f"Number of classes: {len(yolo_model.names)}")
            
            # Check if this is the custom motorcycle model
            expected_classes = ['broken_headlights_tail_lights', 'broken_side_mirror', 'flat_tire', 'oil_leak']
            if any(class_name in str(yolo_model.names) for class_name in expected_classes):
                logger.info("✅ CUSTOM MOTORCYCLE DIAGNOSTIC MODEL IS ACTIVE")
                logger.info("✅ This model can detect motorcycle issues")
            else:
                logger.warning("⚠️ DEFAULT YOLOv8 MODEL IS ACTIVE")
                logger.warning("⚠️ This model CANNOT detect motorcycle issues - only general objects")
                logger.warning("⚠️ Expected motorcycle classes not found in model")
        else:
            logger.warning("Model does not have 'names' attribute")
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
    logger.info("🔗 === SUPABASE MODEL DOWNLOAD STARTED ===")
    
    if not supabase_client:
        logger.error("❌ Supabase client not initialized, skipping model download")
        logger.error("❌ This means the service will fall back to the default YOLOv8 nano model")
        return
    
    try:
        logger.info("✅ Supabase client is initialized")
        logger.info("📥 Attempting to download motorcycle_diagnostic.pt from Supabase Storage...")
        logger.info("🔗 Storage bucket: autosos")
        logger.info("📁 File path: models/yolov8/motorcycle_diagnostic.pt")
        
        response = supabase_client.storage.from_("autosos").download("models/yolov8/motorcycle_diagnostic.pt")
        
        if response:
            logger.info("✅ Successfully received response from Supabase Storage")
            logger.info(f"📊 Response type: {type(response)}")
            logger.info(f"📊 Response size: {len(response)} bytes")
            
            # Save the model to local path
            logger.info(f"💾 Saving model to: {model_path}")
            with open(model_path, 'wb') as f:
                f.write(response)
            
            # Verify the file was saved
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                logger.info("✅ YOLOv8 motorcycle diagnostic model downloaded from Supabase Storage")
                logger.info(f"✅ Model saved to: {model_path}")
                logger.info(f"✅ Model size: {file_size} bytes")
                logger.info("✅ File verification: Model file exists and is readable")
            else:
                logger.error("❌ Model file was not saved successfully")
        else:
            logger.error("❌ No response received from Supabase Storage")
            logger.error("❌ This could mean:")
            logger.error("   - File does not exist at models/yolov8/motorcycle_diagnostic.pt")
            logger.error("   - Storage bucket 'autosos' does not exist")
            logger.error("   - Permission issues with the storage bucket")
            logger.error("   - Network connectivity issues")
        
    except Exception as e:
        logger.error(f"❌ Failed to download YOLOv8 model from Supabase: {e}")
        logger.error(f"❌ Exception type: {type(e)}")
        logger.error("❌ This means the service will fall back to the default YOLOv8 nano model")
        logger.error("❌ The custom motorcycle diagnostic model will NOT be available")

def upload_yolo_model_to_supabase(model_path: str):
    """Upload YOLOv8 model to Supabase Storage"""
    if not supabase_client:
        logger.warning("Supabase client not initialized, skipping model upload")
        return
    
    try:
        # Read the model file
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Upload to Supabase Storage using the correct path
        supabase_client.storage.from_("autosos").upload(
            "models/yolov8/motorcycle_diagnostic.pt",
            model_data,
            {"content-type": "application/octet-stream"}
        )
        logger.info("YOLOv8 model uploaded to Supabase Storage at models/yolov8/motorcycle_diagnostic.pt")
        
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
    
    logger.info(f"Processing detection results with confidence threshold: {confidence_threshold}")
    logger.info(f"Number of results to process: {len(results) if results else 0}")
    
    for i, result in enumerate(results):
        logger.info(f"Processing result {i}")
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            logger.info(f"Result {i} has {len(boxes)} boxes")
            for j, box in enumerate(boxes):
                # Get box data
                if hasattr(box, 'xyxy'):
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    
                    logger.info(f"Box {j}: class={cls}, confidence={conf:.3f}, bbox={bbox}")
                    
                    # Filter by confidence
                    if conf >= confidence_threshold:
                        # Get class name
                        class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                        display_name = CLASS_DISPLAY_NAMES.get(cls, f"Class {cls}")
                        
                        logger.info(f"Box {j} passed confidence filter: {class_name} ({display_name})")
                        
                        detection = {
                            "bbox": bbox,
                            "confidence": float(conf),
                            "class": cls,
                            "class_name": class_name,
                            "display_name": display_name,
                            "color": CLASS_COLORS.get(cls, (255, 255, 255))
                        }
                        detections.append(detection)
                    else:
                        logger.info(f"Box {j} filtered out due to low confidence: {conf:.3f} < {confidence_threshold}")
        else:
            logger.info(f"Result {i} has no boxes or boxes is None")
    
    logger.info(f"Final detections count: {len(detections)}")
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
        
        # Debug logging
        logger.info(f"YOLOv8 detection completed in {detection_time:.3f}s")
        logger.info(f"Model type: {type(yolo_model)}")
        logger.info(f"Model names: {getattr(yolo_model, 'names', 'No names attribute')}")
        logger.info(f"Number of results: {len(results) if results else 0}")
        
        if results:
            for i, result in enumerate(results):
                logger.info(f"Result {i}: boxes={getattr(result, 'boxes', None)}")
                if hasattr(result, 'boxes') and result.boxes is not None:
                    logger.info(f"Result {i}: {len(result.boxes)} detections found")
                else:
                    logger.info(f"Result {i}: No detections found")
        
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
            "processing_time": time.time(),
            "model_info": {
                "model_type": "custom_motorcycle_diagnostic" if hasattr(yolo_model, 'names') and any(class_name in str(yolo_model.names) for class_name in ['broken_headlights_tail_lights', 'broken_side_mirror', 'flat_tire', 'oil_leak']) else "default_yolov8",
                "model_classes": yolo_model.names if hasattr(yolo_model, 'names') else "unknown",
                "total_classes": len(yolo_model.names) if hasattr(yolo_model, 'names') else 0
            }
        }
        
        # Enhanced logging for response
        logger.info(f"🎯 DETECTION RESPONSE SUMMARY:")
        logger.info(f"   - Success: {response_data['success']}")
        logger.info(f"   - Detections found: {response_data['detection_count']}")
        logger.info(f"   - Image size: {response_data['image_size']['width']}x{response_data['image_size']['height']}")
        logger.info(f"   - Confidence threshold: {response_data['confidence_threshold']}")
        logger.info(f"   - Detection time: {response_data['detection_time']:.3f}s")
        logger.info(f"   - Model type: {response_data['model_info']['model_type']}")
        logger.info(f"   - Model classes: {response_data['model_info']['model_classes']}")
        
        if detections:
            logger.info(f"   - Detection details:")
            for i, detection in enumerate(detections):
                logger.info(f"     Detection {i+1}: {detection['class_name']} ({detection['display_name']}) - {detection['confidence']:.3f} confidence")
        else:
            logger.info(f"   - No detections found above confidence threshold {confidence}")
        
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