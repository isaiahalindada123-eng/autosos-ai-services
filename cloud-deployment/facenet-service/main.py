#!/usr/bin/env python3
"""
FaceNet Service for AutoSOS
Cloud-deployed facial recognition service
"""

import os
import time
import base64
import json
import logging
from typing import Dict, Any, Optional
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
    cv2 = DummyCV2()

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import your existing FaceNet service
from facial_recognition_service import FaceNetService

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
REQUEST_COUNT = Counter('facenet_requests_total', 'Total FaceNet requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('facenet_request_duration_seconds', 'FaceNet request duration', ['method', 'endpoint'])

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Global FaceNet service instance
facenet_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global facenet_service
    
    logger.info("Starting FaceNet Service")
    
    try:
        # Initialize FaceNet service
        facenet_service = FaceNetService()
        await facenet_service.initialize()
        logger.info("FaceNet service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize FaceNet service", error=str(e))
        raise
    
    yield
    
    logger.info("Shutting down FaceNet Service")

# Initialize FastAPI app
app = FastAPI(
    title="AutoSOS FaceNet Service",
    description="Facial recognition service for AutoSOS payment system",
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

@app.get("/health")
async def health_check():
    """Service health check"""
    if facenet_service is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "FaceNet service not initialized"}
        )
    
    return {
        "status": "healthy",
        "service": "facenet",
        "timestamp": time.time(),
        "face_count": facenet_service.get_face_count()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/register-face")
async def register_face(
    user_id: str = Form(...),
    user_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Register a new face for recognition"""
    REQUEST_COUNT.labels(method="POST", endpoint="/register-face", status="200").inc()
    
    if facenet_service is None:
        raise HTTPException(status_code=503, detail="FaceNet service not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Register face
        result = facenet_service.register_face(image, user_id, user_name)
        
        if result["success"]:
            # Cache the result
            await redis_client.setex(
                f"facenet:register:{user_id}",
                3600,  # 1 hour
                json.dumps(result)
            )
            
            logger.info("Face registered successfully", user_id=user_id, user_name=user_name)
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
            
    except Exception as e:
        logger.error("Face registration failed", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...)):
    """Recognize a face from uploaded image"""
    REQUEST_COUNT.labels(method="POST", endpoint="/recognize-face", status="200").inc()
    
    if facenet_service is None:
        raise HTTPException(status_code=503, detail="FaceNet service not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Recognize face
        result = facenet_service.recognize_face(image)
        
        if result:
            # Cache the result
            await redis_client.setex(
                f"facenet:recognize:{result['user_id']}",
                300,  # 5 minutes
                json.dumps(result)
            )
            
            logger.info("Face recognized successfully", user_id=result.get('user_id'))
            return {
                "success": True,
                "recognized": True,
                "user_id": result["user_id"],
                "user_name": result["user_name"],
                "confidence": result["confidence"],
                "similarity": result["similarity"]
            }
        else:
            return {
                "success": True,
                "recognized": False,
                "message": "No matching face found"
            }
            
    except Exception as e:
        logger.error("Face recognition failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/recognize-base64")
async def recognize_face_base64(data: Dict[str, str]):
    """Recognize a face from base64 encoded image"""
    REQUEST_COUNT.labels(method="POST", endpoint="/recognize-base64", status="200").inc()
    
    if facenet_service is None:
        raise HTTPException(status_code=503, detail="FaceNet service not initialized")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image_data"])
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Recognize face
        result = facenet_service.recognize_face(image)
        
        if result:
            logger.info("Face recognized successfully", user_id=result.get('user_id'))
            return {
                "success": True,
                "recognized": True,
                "user_id": result["user_id"],
                "user_name": result["user_name"],
                "confidence": result["confidence"],
                "similarity": result["similarity"]
            }
        else:
            return {
                "success": True,
                "recognized": False,
                "message": "No matching face found"
            }
            
    except Exception as e:
        logger.error("Face recognition failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/process-payment")
async def process_payment(
    client_id: str = Form(...),
    mechanic_id: str = Form(...),
    booking_id: str = Form(...),
    amount: float = Form(...),
    file: UploadFile = File(...)
):
    """Process payment using facial recognition"""
    REQUEST_COUNT.labels(method="POST", endpoint="/process-payment", status="200").inc()
    
    if facenet_service is None:
        raise HTTPException(status_code=503, detail="FaceNet service not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        if not OPENCV_AVAILABLE:
            raise HTTPException(status_code=500, detail="OpenCV not available for image processing")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Recognize face
        recognition_result = facenet_service.recognize_face(image)
        
        if not recognition_result:
            return {
                "success": False,
                "verified": False,
                "message": "Face not recognized",
                "error": "No matching face found in database"
            }
        
        # Verify the recognized user matches the expected client
        if recognition_result["user_id"] != client_id:
            return {
                "success": False,
                "verified": False,
                "message": "Face verification failed",
                "error": "Recognized user does not match expected client"
            }
        
        # Check confidence threshold
        if recognition_result["confidence"] < 0.8:  # 80% confidence threshold
            return {
                "success": False,
                "verified": False,
                "message": "Low confidence match",
                "error": f"Confidence {recognition_result['confidence']:.2f} below threshold"
            }
        
        # Payment verification successful
        payment_data = {
            "client_id": client_id,
            "mechanic_id": mechanic_id,
            "booking_id": booking_id,
            "amount": amount,
            "verification_photo": base64.b64encode(image_data).decode('utf-8'),
            "facial_verification_data": {
                "user_id": recognition_result["user_id"],
                "user_name": recognition_result["user_name"],
                "confidence": recognition_result["confidence"],
                "verified_at": time.time()
            }
        }
        
        # Cache the payment data
        await redis_client.setex(
            f"payment:{booking_id}",
            3600,  # 1 hour
            json.dumps(payment_data)
        )
        
        logger.info("Payment verification successful", 
                   client_id=client_id, 
                   booking_id=booking_id,
                   confidence=recognition_result["confidence"])
        
        return {
            "success": True,
            "verified": True,
            "message": "Face verification successful",
            "payment_data": payment_data
        }
        
    except Exception as e:
        logger.error("Payment processing failed", error=str(e), client_id=client_id)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/face-count")
async def get_face_count():
    """Get the number of registered faces"""
    if facenet_service is None:
        raise HTTPException(status_code=503, detail="FaceNet service not initialized")
    
    count = facenet_service.get_face_count()
    return {"face_count": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
