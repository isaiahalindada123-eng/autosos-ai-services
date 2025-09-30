#!/usr/bin/env python3
"""
AutoSOS API Gateway
Central entry point for all AI services (FaceNet, YOLOv8, Ollama)
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import httpx
import redis
import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

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
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

# Service URLs
FACENET_SERVICE_URL = os.getenv("FACENET_SERVICE_URL", "http://facenet-service:8001")
YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "http://yolo-service:8002")
OLLAMA_SERVICE_URL = os.getenv("OLLAMA_SERVICE_URL", "http://ollama-service:11434")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# HTTP client for service calls
http_client = httpx.AsyncClient(timeout=30.0)

# Service health status
service_health = {
    "facenet": False,
    "yolo": False,
    "ollama": False
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting AutoSOS API Gateway")
    
    # Check service health on startup
    await check_all_services()
    
    yield
    
    logger.info("Shutting down AutoSOS API Gateway")
    await http_client.aclose()

# Initialize FastAPI app
app = FastAPI(
    title="AutoSOS API Gateway",
    description="Central API gateway for AutoSOS AI services",
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

# Pydantic models
class DiagnosticRequest(BaseModel):
    user_message: str
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = "llama3.2:3b"

class PaymentRequest(BaseModel):
    client_id: str
    mechanic_id: str
    booking_id: str
    amount: float

class ServiceResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    service: str
    processing_time: float

# Dependency for request timing
async def time_request():
    start_time = time.time()
    yield start_time
    # The timing is handled in the endpoint

# Health check functions
async def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is healthy"""
    try:
        response = await http_client.get(f"{url}/health", timeout=5.0)
        is_healthy = response.status_code == 200
        service_health[service_name] = is_healthy
        return is_healthy
    except Exception as e:
        logger.warning(f"Service {service_name} health check failed", error=str(e))
        service_health[service_name] = False
        return False

async def check_all_services():
    """Check health of all services"""
    tasks = [
        check_service_health("facenet", FACENET_SERVICE_URL),
        check_service_health("yolo", YOLO_SERVICE_URL),
        check_service_health("ollama", OLLAMA_SERVICE_URL)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

# API Endpoints

@app.get("/health")
async def health_check():
    """Gateway health check"""
    await check_all_services()
    
    return {
        "status": "healthy",
        "gateway": True,
        "services": service_health,
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# FaceNet Endpoints

@app.post("/api/facenet/register")
async def register_face(
    user_id: str = Form(...),
    user_name: str = Form(...),
    file: UploadFile = File(...),
    start_time: float = Depends(time_request)
):
    """Register a new face for facial recognition"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/facenet/register", status="200").inc()
    
    if not service_health["facenet"]:
        raise HTTPException(status_code=503, detail="FaceNet service unavailable")
    
    try:
        # Forward request to FaceNet service
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {"user_id": user_id, "user_name": user_name}
        
        response = await http_client.post(
            f"{FACENET_SERVICE_URL}/register-face",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Cache the result
            await redis_client.setex(
                f"facenet:register:{user_id}",
                3600,  # 1 hour
                str(result)
            )
            
            return ServiceResponse(
                success=True,
                data=result,
                service="facenet",
                processing_time=time.time() - start_time
            )
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="FaceNet service timeout")
    except Exception as e:
        logger.error("FaceNet registration failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/facenet/recognize")
async def recognize_face(
    file: UploadFile = File(...),
    start_time: float = Depends(time_request)
):
    """Recognize a face from uploaded image"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/facenet/recognize", status="200").inc()
    
    if not service_health["facenet"]:
        raise HTTPException(status_code=503, detail="FaceNet service unavailable")
    
    try:
        # Forward request to FaceNet service
        files = {"file": (file.filename, await file.read(), file.content_type)}
        
        response = await http_client.post(
            f"{FACENET_SERVICE_URL}/recognize-face",
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            return ServiceResponse(
                success=True,
                data=result,
                service="facenet",
                processing_time=time.time() - start_time
            )
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="FaceNet service timeout")
    except Exception as e:
        logger.error("FaceNet recognition failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/facenet/process-payment")
async def process_facial_payment(
    client_id: str = Form(...),
    mechanic_id: str = Form(...),
    booking_id: str = Form(...),
    amount: float = Form(...),
    file: UploadFile = File(...),
    start_time: float = Depends(time_request)
):
    """Process payment using facial recognition"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/facenet/process-payment", status="200").inc()
    
    if not service_health["facenet"]:
        raise HTTPException(status_code=503, detail="FaceNet service unavailable")
    
    try:
        # Forward request to FaceNet service
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {
            "client_id": client_id,
            "mechanic_id": mechanic_id,
            "booking_id": booking_id,
            "amount": str(amount)
        }
        
        response = await http_client.post(
            f"{FACENET_SERVICE_URL}/process-payment",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return ServiceResponse(
                success=True,
                data=result,
                service="facenet",
                processing_time=time.time() - start_time
            )
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="FaceNet service timeout")
    except Exception as e:
        logger.error("Facial payment processing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# YOLOv8 Endpoints

@app.post("/api/yolo/detect")
async def detect_motorcycle_issues(
    file: UploadFile = File(...),
    start_time: float = Depends(time_request)
):
    """Detect motorcycle issues using YOLOv8"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/yolo/detect", status="200").inc()
    
    if not service_health["yolo"]:
        raise HTTPException(status_code=503, detail="YOLOv8 service unavailable")
    
    try:
        # Forward request to YOLOv8 service
        files = {"file": (file.filename, await file.read(), file.content_type)}
        
        response = await http_client.post(
            f"{YOLO_SERVICE_URL}/detect",
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            return ServiceResponse(
                success=True,
                data=result,
                service="yolo",
                processing_time=time.time() - start_time
            )
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="YOLOv8 service timeout")
    except Exception as e:
        logger.error("YOLOv8 detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Ollama Endpoints

@app.post("/api/ollama/diagnostic")
async def generate_diagnostic_response(
    request: DiagnosticRequest,
    start_time: float = Depends(time_request)
):
    """Generate AI diagnostic response using Ollama"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/ollama/diagnostic", status="200").inc()
    
    if not service_health["ollama"]:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    
    try:
        # Forward request to Ollama service
        response = await http_client.post(
            f"{OLLAMA_SERVICE_URL}/api/generate",
            json={
                "model": request.model,
                "prompt": request.user_message,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return ServiceResponse(
                success=True,
                data=result,
                service="ollama",
                processing_time=time.time() - start_time
            )
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama service timeout")
    except Exception as e:
        logger.error("Ollama diagnostic failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/ollama/models")
async def get_available_models():
    """Get available Ollama models"""
    if not service_health["ollama"]:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    
    try:
        response = await http_client.get(f"{OLLAMA_SERVICE_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        logger.error("Failed to get Ollama models", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

# Combined Diagnostic Endpoint

@app.post("/api/diagnostic/complete")
async def complete_diagnostic(
    image_file: UploadFile = File(...),
    user_message: str = Form(...),
    start_time: float = Depends(time_request)
):
    """Complete diagnostic using both YOLOv8 and Ollama"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/diagnostic/complete", status="200").inc()
    
    if not service_health["yolo"] or not service_health["ollama"]:
        raise HTTPException(status_code=503, detail="Required services unavailable")
    
    try:
        # Run YOLOv8 detection
        yolo_files = {"file": (image_file.filename, await image_file.read(), image_file.content_type)}
        yolo_response = await http_client.post(
            f"{YOLO_SERVICE_URL}/detect",
            files=yolo_files
        )
        
        yolo_result = yolo_response.json() if yolo_response.status_code == 200 else None
        
        # Generate enhanced prompt for Ollama
        enhanced_prompt = f"""
        Based on the following motorcycle diagnostic information:
        
        Visual Analysis (YOLOv8): {yolo_result}
        User Description: {user_message}
        
        Please provide a comprehensive diagnostic analysis including:
        1. Issue identification
        2. Severity assessment
        3. Immediate actions
        4. Long-term solutions
        5. Safety warnings
        """
        
        # Run Ollama analysis
        ollama_response = await http_client.post(
            f"{OLLAMA_SERVICE_URL}/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1
                }
            }
        )
        
        ollama_result = ollama_response.json() if ollama_response.status_code == 200 else None
        
        return ServiceResponse(
            success=True,
            data={
                "visual_analysis": yolo_result,
                "ai_analysis": ollama_result,
                "combined_diagnostic": True
            },
            service="combined",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Complete diagnostic failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
