#!/usr/bin/env python3
"""
Embedding model service for the Advanced MCP Server.

This module provides an API for generating embeddings using sentence-transformers.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("embedding_service")

# Get model name from environment variable or use default
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_PATH = f"sentence-transformers/{MODEL_NAME}"
CACHE_DIR = os.environ.get("CACHE_DIR", "/app/model_cache")

# Create the FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="A service for generating embeddings using sentence-transformers",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str]
    batch_size: Optional[int] = 32

# Response model
class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    processing_time: float

# Load the model
@app.on_event("startup")
async def startup_event():
    """Initialize the embedding model on startup."""
    global model
    try:
        logger.info(f"Loading model: {MODEL_PATH}")
        start_time = time.time()
        model = SentenceTransformer(MODEL_PATH, cache_folder=CACHE_DIR)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Simple check to make sure the model is working
        embedding = model.encode(["Hello world"])
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "dimensions": len(embedding[0])
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Model info endpoint
@app.get("/info")
async def model_info():
    """Return information about the embedding model."""
    return {
        "model": MODEL_NAME,
        "dimensions": model.get_sentence_embedding_dimension(),
        "framework": "sentence-transformers",
        "max_seq_length": model.get_max_seq_length(),
        "normalization": model.get_sentence_embedding_dimension_info()
    }

# Embedding endpoint
@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the given texts."""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        start_time = time.time()
        embeddings = model.encode(
            request.texts,
            batch_size=request.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        processing_time = time.time() - start_time
        
        # Convert numpy arrays to lists
        embeddings_as_lists = embeddings.tolist()
        
        return {
            "embeddings": embeddings_as_lists,
            "model": MODEL_NAME,
            "dimensions": len(embeddings_as_lists[0]),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests to the API."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.2f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, log_level="info")
