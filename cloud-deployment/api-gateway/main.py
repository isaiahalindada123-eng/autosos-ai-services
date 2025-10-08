#!/usr/bin/env python3
"""
AutoSOS API Gateway with GPT-5 Integration
Central entry point for all AI services (FaceNet, YOLOv8, Ollama, GPT-5)
"""

import os
import time
import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

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
GPT5_COST_TRACKER = Counter('gpt5_cost_total', 'Total GPT-5 cost in USD', ['model', 'endpoint'])

# Service URLs
FACENET_SERVICE_URL = os.getenv("FACENET_SERVICE_URL", "http://facenet-service:8001")
YOLO_SERVICE_URL = os.getenv("YOLO_SERVICE_URL", "http://yolo-service:8002")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# GPT-5 Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MAX_MONTHLY_COST = float(os.getenv("MAX_MONTHLY_COST", "200.0"))
COST_ALERT_THRESHOLD = float(os.getenv("COST_ALERT_THRESHOLD", "150.0"))
CACHE_DURATION = int(os.getenv("CACHE_DURATION", "3600"))  # 1 hour

# Initialize Redis
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# HTTP client for service calls
http_client = httpx.AsyncClient(timeout=30.0)

# Service health status
service_health = {
    "facenet": False,
    "yolo": False,
    "gpt5": False
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

class GPT5DiagnosticRequest(BaseModel):
    user_message: str
    yolo_detections: Optional[List[Dict[str, Any]]] = None
    user_tier: str = "free"
    emergency_level: str = "normal"
    context: Optional[Dict[str, Any]] = None

class GPT5DiagnosticResponse(BaseModel):
    success: bool
    response: str
    model_used: str
    cost: float
    cached: bool
    processing_time: float
    recommendations: List[str]
    severity: str
    immediate_actions: List[str]

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

class CostMetrics(BaseModel):
    monthly_cost: float
    daily_cost: float
    requests_today: int
    average_cost_per_request: float
    cost_alert: bool

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
        check_gpt5_health()
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

# GPT-5 Helper Functions
async def check_gpt5_health() -> bool:
    """Check GPT-5 service health"""
    try:
        if not OPENAI_API_KEY:
            service_health["gpt5"] = False
            return False
        
        # Simple test request to check API key validity
        response = await http_client.get(
            f"{OPENAI_BASE_URL}/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=5.0
        )
        is_healthy = response.status_code == 200
        service_health["gpt5"] = is_healthy
        return is_healthy
    except Exception as e:
        logger.warning("GPT-5 health check failed", error=str(e))
        service_health["gpt5"] = False
        return False

def track_gpt5_cost(model: str, endpoint: str, cost: float):
    """Track GPT-5 API costs"""
    GPT5_COST_TRACKER.labels(model=model, endpoint=endpoint).inc(cost)
    
    # Store in Redis for monthly tracking
    today = datetime.now().strftime("%Y-%m-%d")
    redis_client.hincrbyfloat(f"gpt5_costs:{today}", f"{model}:{endpoint}", cost)
    redis_client.expire(f"gpt5_costs:{today}", 86400 * 31)  # 31 days

def get_monthly_gpt5_cost() -> float:
    """Get current month's total GPT-5 cost"""
    total_cost = 0.0
    current_month = datetime.now().strftime("%Y-%m")
    
    # Get all cost entries for current month
    keys = redis_client.keys(f"gpt5_costs:{current_month}-*")
    for key in keys:
        costs = redis_client.hgetall(key)
        for cost_key, cost_value in costs.items():
            total_cost += float(cost_value)
    
    return total_cost


def generate_cache_key(request: GPT5DiagnosticRequest) -> str:
    """Generate cache key for GPT-5 request"""
    content = f"{request.user_message}:{json.dumps(request.yolo_detections or [])}:{request.user_tier}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cached_gpt5_response(cache_key: str) -> Optional[GPT5DiagnosticResponse]:
    """Get cached GPT-5 response if available"""
    try:
        cached_data = redis_client.get(f"gpt5_cache:{cache_key}")
        if cached_data:
            data = json.loads(cached_data)
            return GPT5DiagnosticResponse(**data)
    except Exception as e:
        logger.error(f"Error retrieving cached GPT-5 response: {e}")
    return None

def cache_gpt5_response(cache_key: str, response: GPT5DiagnosticResponse, duration: int = CACHE_DURATION):
    """Cache GPT-5 response"""
    try:
        redis_client.setex(
            f"gpt5_cache:{cache_key}",
            duration,
            json.dumps(response.dict())
        )
    except Exception as e:
        logger.error(f"Error caching GPT-5 response: {e}")

async def call_gpt5_api(request: GPT5DiagnosticRequest) -> GPT5DiagnosticResponse:
    """Call GPT-5 API"""
    start_time = time.time()
    
    # Build enhanced prompt with YOLOv8 context
    prompt = build_diagnostic_prompt(request)
    
    try:
        response = await http_client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional motorcycle mechanic with 15+ years of experience. You specialize in diagnosing and repairing all types of motorcycles including sport bikes, cruisers, touring bikes, and dirt bikes. You have extensive knowledge of engines, transmissions, electrical systems, brakes, suspension, and safety systems. You always prioritize rider safety and provide clear, step-by-step guidance. You speak in a friendly, professional manner and explain technical concepts in terms that motorcycle owners can understand. Your goal is to help riders identify issues, understand their severity, and provide practical solutions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="GPT-5 API error")
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Estimate cost (rough calculation)
        cost = estimate_gpt5_cost(result["usage"])
        
        # Track cost
        track_gpt5_cost("gpt-5", "diagnostic", cost)
        
        processing_time = time.time() - start_time
        
        return GPT5DiagnosticResponse(
            success=True,
            response=content,
            model_used="gpt-5",
            cost=cost,
            cached=False,
            processing_time=processing_time,
            recommendations=extract_recommendations(content),
            severity=determine_severity(request, content),
            immediate_actions=extract_immediate_actions(content)
        )
        
    except Exception as e:
        logger.error(f"GPT-5 API error: {e}")
        raise HTTPException(status_code=500, detail=f"GPT-5 service error: {str(e)}")

def build_diagnostic_prompt(request: GPT5DiagnosticRequest) -> str:
    """Build enhanced diagnostic prompt with YOLOv8 context"""
    prompt = f"""
    Hey there! I'm your motorcycle mechanic, and I'm here to help you with your bike. Let me take a look at what's going on.
    
    **What you're telling me:** {request.user_message}
    
    """
    
    if request.yolo_detections:
        prompt += f"""
    **What I can see from the visual inspection:**
    """
        for i, detection in enumerate(request.yolo_detections, 1):
            prompt += f"""
    Issue #{i}: {detection.get('class_display_name', 'Unknown')}
    - How confident I am: {detection.get('confidence', 0):.1%}
    - How serious this looks: {detection.get('severity', 'Unknown')}
    - Where I spotted it: {detection.get('bbox', 'Unknown')}
    """
    
    prompt += f"""
    
    Please answer in this EXACT format:
    
    **What is the problem?**
    [Explain the specific issue you've identified]
    
    **How to fix or remedy the problem?**
    [Give step-by-step instructions to fix it]
    
    **Is mechanic needed?**
    [Yes/No - and explain why]
    
    **Can I ride the motorcycle?**
    [Yes/No - and explain the safety implications]
    
    Keep your response concise and practical. Focus on what the rider needs to know right now.
    """
    
    return prompt

def estimate_gpt5_cost(usage: Dict[str, Any]) -> float:
    """Estimate GPT-5 API cost"""
    # Rough estimation - adjust based on actual GPT-5 pricing
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Estimated pricing (adjust when GPT-5 pricing is released)
    prompt_cost = prompt_tokens * 0.00001  # $0.01 per 1K tokens
    completion_cost = completion_tokens * 0.00003  # $0.03 per 1K tokens
    
    return prompt_cost + completion_cost

def extract_recommendations(content: str) -> List[str]:
    """Extract recommendations from response"""
    recommendations = []
    lines = content.split('\n')
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'need to']):
            recommendations.append(line.strip())
    
    return recommendations[:5]  # Limit to 5 recommendations

def determine_severity(request: GPT5DiagnosticRequest, content: str) -> str:
    """Determine severity level"""
    if request.emergency_level == "critical":
        return "critical"
    
    content_lower = content.lower()
    if any(word in content_lower for word in ['critical', 'dangerous', 'emergency', 'urgent']):
        return "high"
    elif any(word in content_lower for word in ['serious', 'important', 'attention']):
        return "medium"
    else:
        return "low"

def extract_immediate_actions(content: str) -> List[str]:
    """Extract immediate actions from response"""
    actions = []
    lines = content.split('\n')
    
    for line in lines:
        if any(keyword in line.lower() for keyword in ['immediately', 'urgent', 'stop', 'do not', 'avoid']):
            actions.append(line.strip())
    
    return actions[:3]  # Limit to 3 immediate actions

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AutoSOS API Gateway",
        "version": "1.0.0",
        "status": "running",
        "description": "Central API gateway for AutoSOS AI services",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "facenet": "/api/facenet/*",
            "yolo": "/api/yolo/*",
            "gpt5": "/api/gpt5/*",
            "diagnostic": "/api/diagnostic/*"
        },
        "services": {
            "facenet": "Facial recognition for payments",
            "yolo": "Motorcycle diagnostic detection",
            "gpt5": "Advanced AI diagnostics with cost optimization"
        }
    }

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

@app.get("/api")
async def api_info():
    """API documentation endpoint"""
    return {
        "title": "AutoSOS API Gateway",
        "version": "1.0.0",
        "description": "Central API gateway for AutoSOS motorcycle diagnostic services",
        "documentation": {
            "facenet": {
                "description": "Facial recognition service for secure payments",
                "endpoints": {
                    "register": "POST /api/facenet/register",
                    "recognize": "POST /api/facenet/recognize", 
                    "process-payment": "POST /api/facenet/process-payment",
                    "check-registration": "GET /api/facenet/check-face-registration/{user_id}",
                    "remove-face": "DELETE /api/facenet/remove-face/{user_id}"
                }
            },
            "yolo": {
                "description": "YOLOv8 motorcycle diagnostic detection",
                "endpoints": {
                    "detect": "POST /api/yolo/detect",
                    "detect-base64": "POST /api/yolo/detect-base64"
                }
            },
            "gpt5": {
                "description": "Advanced AI diagnostics with GPT-5 and cost optimization",
                "endpoints": {
                    "diagnostic": "POST /api/gpt5/diagnostic",
                    "cost-metrics": "GET /api/gpt5/cost-metrics"
                }
            },
            "diagnostic": {
                "description": "Combined diagnostic service (YOLOv8 + AI)",
                "endpoints": {
                    "complete": "POST /api/diagnostic/complete"
                }
            }
        },
        "status_endpoints": {
            "health": "GET /health",
            "metrics": "GET /metrics",
            "root": "GET /"
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Direct YOLOv8 endpoints for backward compatibility
@app.post("/detect-base64")
async def detect_base64_direct(
    data: Dict[str, Any],
    start_time: float = Depends(time_request)
):
    """Direct YOLOv8 detection from base64 (backward compatibility)"""
    REQUEST_COUNT.labels(method="POST", endpoint="/detect-base64", status="200").inc()
    
    if not service_health["yolo"]:
        raise HTTPException(status_code=503, detail="YOLOv8 service unavailable")
    
    try:
        # Forward request to YOLOv8 service
        response = await http_client.post(
            f"{YOLO_SERVICE_URL}/detect-base64",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result  # Return raw result for backward compatibility
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="YOLOv8 service timeout")
    except Exception as e:
        logger.error("YOLOv8 direct base64 detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

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

@app.get("/api/facenet/check-face-registration/{user_id}")
async def check_face_registration(
    user_id: str,
    start_time: float = Depends(time_request)
):
    """Check if a user has a registered face"""
    REQUEST_COUNT.labels(method="GET", endpoint="/api/facenet/check-face-registration", status="200").inc()
    
    if not service_health["facenet"]:
        raise HTTPException(status_code=503, detail="FaceNet service unavailable")
    
    try:
        # Forward request to FaceNet service
        response = await http_client.get(
            f"{FACENET_SERVICE_URL}/check-face-registration/{user_id}"
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
        logger.error("Face registration check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/api/facenet/remove-face/{user_id}")
async def remove_face(
    user_id: str,
    start_time: float = Depends(time_request)
):
    """Remove a registered face"""
    REQUEST_COUNT.labels(method="DELETE", endpoint="/api/facenet/remove-face", status="200").inc()
    
    if not service_health["facenet"]:
        raise HTTPException(status_code=503, detail="FaceNet service unavailable")
    
    try:
        # Forward request to FaceNet service
        response = await http_client.delete(
            f"{FACENET_SERVICE_URL}/remove-face/{user_id}"
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
        logger.error("Face removal failed", error=str(e))
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

@app.post("/api/yolo/detect-base64")
async def detect_motorcycle_issues_base64(
    data: Dict[str, Any],
    start_time: float = Depends(time_request)
):
    """Detect motorcycle issues using YOLOv8 from base64 encoded image"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/yolo/detect-base64", status="200").inc()
    
    if not service_health["yolo"]:
        raise HTTPException(status_code=503, detail="YOLOv8 service unavailable")
    
    try:
        # Forward request to YOLOv8 service
        response = await http_client.post(
            f"{YOLO_SERVICE_URL}/detect-base64",
            json=data
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
        logger.error("YOLOv8 base64 detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# GPT-5 Endpoints

@app.post("/api/gpt5/diagnostic", response_model=GPT5DiagnosticResponse)
async def generate_gpt5_diagnostic(
    request: GPT5DiagnosticRequest,
    start_time: float = Depends(time_request)
):
    """Generate diagnostic response using GPT-5 with intelligent routing"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/gpt5/diagnostic", status="200").inc()
    
    # Check cache first
    cache_key = generate_cache_key(request)
    cached_response = get_cached_gpt5_response(cache_key)
    if cached_response:
        cached_response.cached = True
        return cached_response
    
    try:
        # Always use GPT-5 for diagnostics
        response = await call_gpt5_api(request)
        
        # Cache the response
        cache_gpt5_response(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"GPT-5 diagnostic generation error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/api/gpt5/diagnostic", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gpt5/cost-metrics", response_model=CostMetrics)
async def get_gpt5_cost_metrics():
    """Get current GPT-5 cost metrics"""
    monthly_cost = get_monthly_gpt5_cost()
    daily_cost = 0.0
    
    # Get today's cost
    today = datetime.now().strftime("%Y-%m-%d")
    today_costs = redis_client.hgetall(f"gpt5_costs:{today}")
    for cost_value in today_costs.values():
        daily_cost += float(cost_value)
    
    # Get today's request count
    requests_today = redis_client.get(f"gpt5_requests:{today}") or "0"
    
    return CostMetrics(
        monthly_cost=monthly_cost,
        daily_cost=daily_cost,
        requests_today=int(requests_today),
        average_cost_per_request=daily_cost / max(int(requests_today), 1),
        cost_alert=monthly_cost >= COST_ALERT_THRESHOLD
    )

# Combined Diagnostic Endpoint

@app.post("/api/diagnostic/complete")
async def complete_diagnostic(
    image_file: UploadFile = File(...),
    user_message: str = Form(...),
    user_tier: str = Form("free"),
    emergency_level: str = Form("normal"),
    start_time: float = Depends(time_request)
):
    """Complete diagnostic using YOLOv8 and GPT-5 with intelligent routing"""
    REQUEST_COUNT.labels(method="POST", endpoint="/api/diagnostic/complete", status="200").inc()
    
    if not service_health["yolo"]:
        raise HTTPException(status_code=503, detail="YOLOv8 service unavailable")
    
    try:
        # Run YOLOv8 detection
        yolo_files = {"file": (image_file.filename, await image_file.read(), image_file.content_type)}
        yolo_response = await http_client.post(
            f"{YOLO_SERVICE_URL}/detect",
            files=yolo_files
        )
        
        yolo_result = yolo_response.json() if yolo_response.status_code == 200 else None
        yolo_detections = yolo_result.get("detections", []) if yolo_result else []
        
        # Create GPT-5 diagnostic request
        gpt5_request = GPT5DiagnosticRequest(
            user_message=user_message,
            yolo_detections=yolo_detections,
            user_tier=user_tier,
            emergency_level=emergency_level
        )
        
        # Check cache first
        cache_key = generate_cache_key(gpt5_request)
        cached_response = get_cached_gpt5_response(cache_key)
        
        if cached_response:
            cached_response.cached = True
            ai_analysis = cached_response
        else:
            # Always use GPT-5 for diagnostics
            ai_analysis = await call_gpt5_api(gpt5_request)
            
            # Cache the response
            cache_gpt5_response(cache_key, ai_analysis)
        
        processing_time = time.time() - start_time
        
        return ServiceResponse(
            success=True,
            data={
                "visual_analysis": yolo_result,
                "ai_analysis": ai_analysis.dict(),
                "combined_diagnostic": True,
                "model_used": ai_analysis.model_used,
                "cost": ai_analysis.cost,
                "cached": ai_analysis.cached
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
