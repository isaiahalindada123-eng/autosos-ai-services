#!/usr/bin/env python3
"""
Ollama Service for AutoSOS
Cloud-deployed AI chat diagnostic service
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import structlog

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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:3b")
USE_MOCK_RESPONSES = os.getenv("USE_MOCK_RESPONSES", "true").lower() == "true"

# HTTP client for Ollama API calls
http_client = httpx.AsyncClient(timeout=60.0)

# Mock models for when Ollama is not available
MOCK_MODELS = {
    "models": [
        {
            "name": "llama3.2:3b",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 2000000000,
            "digest": "mock-digest-1",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "3B",
                "quantization_level": "Q4_0"
            }
        },
        {
            "name": "llama3.2:7b",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 4000000000,
            "digest": "mock-digest-2",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }
    ]
}

# Pydantic models
class GenerateRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    model: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Ollama Service")
    
    # Check if Ollama is available
    try:
        response = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Ollama service available with {len(models.get('models', []))} models")
        else:
            logger.warning("Ollama service not responding properly")
    except Exception as e:
        logger.warning(f"Ollama service not available: {e}")
    
    yield
    
    logger.info("Shutting down Ollama Service")
    await http_client.aclose()

# Initialize FastAPI app
app = FastAPI(
    title="AutoSOS Ollama Service",
    description="AI chat diagnostic service using Ollama",
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
    try:
        response = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json()
            return {
                "status": "healthy",
                "service": "ollama",
                "timestamp": time.time(),
                "models_available": len(models.get('models', [])),
                "ollama_url": OLLAMA_BASE_URL
            }
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Ollama API not responding"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": f"Ollama service unavailable: {str(e)}"}
        )

@app.get("/api/tags")
async def get_models():
    """Get available Ollama models"""
    if USE_MOCK_RESPONSES:
        logger.info("Returning mock models")
        return MOCK_MODELS
    
    try:
        response = await http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("Ollama API not responding, falling back to mock models")
            return MOCK_MODELS
    except Exception as e:
        logger.warning(f"Ollama service unavailable, using mock models: {str(e)}")
        return MOCK_MODELS

@app.post("/api/generate")
async def generate_response(request: GenerateRequest):
    """Generate AI response using Ollama"""
    try:
        # Prepare request data
        request_data = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": request.options or {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            }
        }
        
        logger.info("Generating response", model=request.model, prompt_length=len(request.prompt))
        
        # Call Ollama API
        response = await http_client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Response generated successfully", 
                       model=request.model, 
                       response_length=len(result.get('response', '')))
            return result
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama service timeout")
    except Exception as e:
        logger.error("Failed to generate response", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/pull")
async def pull_model(model_name: str):
    """Pull/download a new Ollama model"""
    try:
        logger.info("Pulling model", model=model_name)
        
        response = await http_client.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model_name}
        )
        
        if response.status_code == 200:
            logger.info("Model pulled successfully", model=model_name)
            return {"success": True, "model": model_name}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
            
    except Exception as e:
        logger.error("Failed to pull model", error=str(e), model=model_name)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/diagnostic")
async def generate_diagnostic_response(request: GenerateRequest):
    """Generate diagnostic response with motorcycle-specific context"""
    try:
        # Enhanced prompt for motorcycle diagnostics
        diagnostic_prompt = f"""
You are an expert motorcycle mechanic and diagnostic specialist. Based on the following information, provide a comprehensive diagnostic analysis:

User Description: {request.prompt}

Please provide:
1. **Issue Identification**: What problems do you see?
2. **Severity Assessment**: Rate the severity (Low/Medium/High/Critical)
3. **Immediate Actions**: What should be done right now?
4. **Long-term Solutions**: What repairs or maintenance are needed?
5. **Safety Warnings**: Any safety concerns?
6. **Follow-up Questions**: What additional information would help?

Be specific, practical, and prioritize safety. Use your expertise to provide actionable advice.
"""
        
        # Create diagnostic request
        diagnostic_request = GenerateRequest(
            model=request.model,
            prompt=diagnostic_prompt,
            stream=request.stream,
            options=request.options
        )
        
        # Generate response
        result = await generate_response(diagnostic_request)
        
        # Add diagnostic metadata
        result["diagnostic_analysis"] = True
        result["original_prompt"] = request.prompt
        result["enhanced_prompt"] = diagnostic_prompt
        
        return result
        
    except Exception as e:
        logger.error("Failed to generate diagnostic response", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
