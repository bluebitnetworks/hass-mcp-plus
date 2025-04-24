#!/usr/bin/env python3
"""
Advanced MCP Server for Home Assistant

This module implements a Model Context Protocol (MCP) server that integrates with
Home Assistant to provide advanced AI capabilities for smart home control, automation,
and dashboard generation.
"""

import os
import sys
import json
import logging
import asyncio
import datetime
import ssl
from typing import Dict, List, Optional, Union, Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import local modules
from config import AppConfig, load_config
from llm import LLMManager
from rag import RAGSystem
from hass_client import HomeAssistantClient
from plugins import PluginManager
from utils import logger, setup_logging
from dashboard_generator import DashboardGenerator
from automation_generator import AutomationGenerator
from security import authenticate, AuthData, get_current_user
from schema import (
    MCPRequest, MCPResponse, QueryRequest, ContextData, 
    DashboardRequest, AutomationRequest, EntityInfo
)

# Setup logging
setup_logging()
logger = logging.getLogger("mcp_server")

# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Home Assistant Advanced MCP Integration",
    description="A Model Context Protocol server for Home Assistant with advanced AI capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm_manager = None
rag_system = None
hass_client = None
plugin_manager = None
dashboard_generator = None
automation_generator = None

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    global llm_manager, rag_system, hass_client, plugin_manager, dashboard_generator, automation_generator
    
    logger.info("Starting Home Assistant Advanced MCP Server...")
    
    # Initialize Home Assistant client
    hass_client = HomeAssistantClient(
        url=config.hass_url,
        token=config.hass_token,
    )
    await hass_client.connect()
    
    # Initialize LLM Manager
    llm_manager = LLMManager(
        provider=config.llm_provider,
        model=config.llm_model,
        api_key=config.llm_api_key,
        url=config.llm_url,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )
    
    # Initialize RAG system if enabled
    if config.enable_rag:
        rag_system = RAGSystem(
            vector_store=config.rag_vector_store,
            embedding_model=config.rag_embedding_model,
            chunk_size=config.rag_chunk_size,
            index_path=config.rag_index_path,
            hass_client=hass_client,
        )
        await rag_system.initialize()
    
    # Initialize plugin manager
    plugin_manager = PluginManager(config=config)
    await plugin_manager.load_plugins()
    
    # Initialize generators
    if config.enable_dashboard_gen:
        dashboard_generator = DashboardGenerator(
            llm_manager=llm_manager,
            hass_client=hass_client,
        )
    
    if config.enable_automation_gen:
        automation_generator = AutomationGenerator(
            llm_manager=llm_manager,
            hass_client=hass_client,
        )
    
    logger.info("Home Assistant Advanced MCP Server started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Home Assistant Advanced MCP Server...")
    
    if hass_client:
        await hass_client.disconnect()
    
    if rag_system:
        await rag_system.close()
    
    if plugin_manager:
        await plugin_manager.unload_plugins()
    
    logger.info("Server shutdown complete")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

@app.get("/info")
async def server_info():
    """Server information endpoint."""
    info = {
        "name": "Home Assistant Advanced MCP Server",
        "version": "1.0.0",
        "features": {
            "rag": config.enable_rag,
            "dashboard_generation": config.enable_dashboard_gen,
            "automation_generation": config.enable_automation_gen,
            "voice_processing": config.enable_voice_processing,
            "image_processing": config.enable_image_processing,
        },
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
    }
    return info

@app.post("/api/query", response_model=MCPResponse)
async def query(request: QueryRequest, auth: AuthData = Depends(authenticate)):
    """Process a natural language query about the smart home."""
    try:
        # Get current state from Home Assistant
        states = await hass_client.get_states()
        
        # Build context for the LLM
        context = ContextData(
            states=states,
            timestamp=datetime.datetime.now().isoformat(),
            user=auth.user,
        )
        
        # Enhance context with RAG if enabled
        if rag_system and config.enable_rag:
            context = await rag_system.enhance_context(request.query, context)
        
        # Apply plugins to enhance context further
        if plugin_manager:
            context = await plugin_manager.process_query(request.query, context)
        
        # Process query with LLM
        mcp_request = MCPRequest(
            query=request.query,
            context=context,
        )
        
        response = await llm_manager.process_query(mcp_request)
        
        # Log the interaction
        logger.info(f"Query: '{request.query}' processed successfully")
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/dashboard/generate", response_model=Dict[str, Any])
async def generate_dashboard(request: DashboardRequest, auth: AuthData = Depends(authenticate)):
    """Generate a dashboard from natural language description."""
    if not config.enable_dashboard_gen or not dashboard_generator:
        raise HTTPException(status_code=400, detail="Dashboard generation is not enabled")
    
    try:
        # Get current state from Home Assistant
        states = await hass_client.get_states()
        
        # Generate dashboard config
        dashboard_config = await dashboard_generator.generate(
            description=request.description,
            states=states,
            room=request.room,
            entities=request.entities,
        )
        
        return {
            "dashboard": dashboard_config,
            "preview_url": f"/api/dashboard/preview/{dashboard_config['id']}",
            "install_url": f"/api/dashboard/install/{dashboard_config['id']}",
        }
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")

@app.post("/api/automation/generate", response_model=Dict[str, Any])
async def generate_automation(request: AutomationRequest, auth: AuthData = Depends(authenticate)):
    """Generate an automation from natural language description."""
    if not config.enable_automation_gen or not automation_generator:
        raise HTTPException(status_code=400, detail="Automation generation is not enabled")
    
    try:
        # Get current state from Home Assistant
        states = await hass_client.get_states()
        
        # Generate automation config
        automation_config = await automation_generator.generate(
            description=request.description,
            states=states,
            entities=request.entities,
        )
        
        return {
            "automation": automation_config,
            "explanation": automation_generator.explain(automation_config),
            "install_url": f"/api/automation/install/{automation_config['id']}",
        }
    
    except Exception as e:
        logger.error(f"Error generating automation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating automation: {str(e)}")

@app.post("/api/automation/install/{automation_id}")
async def install_automation(automation_id: str, auth: AuthData = Depends(authenticate)):
    """Install a generated automation in Home Assistant."""
    try:
        success = await automation_generator.install(automation_id)
        if success:
            return {"status": "success", "message": "Automation installed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to install automation")
    
    except Exception as e:
        logger.error(f"Error installing automation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error installing automation: {str(e)}")

@app.post("/api/dashboard/install/{dashboard_id}")
async def install_dashboard(dashboard_id: str, auth: AuthData = Depends(authenticate)):
    """Install a generated dashboard in Home Assistant."""
    try:
        success = await dashboard_generator.install(dashboard_id)
        if success:
            return {"status": "success", "message": "Dashboard installed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to install dashboard")
    
    except Exception as e:
        logger.error(f"Error installing dashboard: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error installing dashboard: {str(e)}")

@app.post("/api/entities/suggest", response_model=List[EntityInfo])
async def suggest_entities(request: Dict[str, Any], auth: AuthData = Depends(authenticate)):
    """Suggest entities based on natural language description."""
    try:
        # Get all entities from Home Assistant
        states = await hass_client.get_states()
        
        # Use LLM to suggest relevant entities
        prompt = f"""
        Given the following request: "{request.get('description', '')}"
        And the following list of entities in the user's home:
        
        {json.dumps([{'entity_id': s.entity_id, 'state': s.state, 'attributes': s.attributes} for s in states])}
        
        Return a JSON array of entity IDs that are most relevant to the request.
        Format: [{"entity_id": "light.living_room", "relevance": "high", "reason": "This is the main light in the living room"}]
        """
        
        response = await llm_manager.generate_text(prompt)
        
        try:
            suggested_entities = json.loads(response)
            return suggested_entities
        except json.JSONDecodeError:
            logger.error(f"LLM returned invalid JSON: {response}")
            return []
    
    except Exception as e:
        logger.error(f"Error suggesting entities: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error suggesting entities: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    
    try:
        # Authenticate the WebSocket connection
        auth_message = await websocket.receive_json()
        token = auth_message.get("token")
        
        if not token:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close(code=1008)
            return
        
        try:
            auth_data = await authenticate(token)
        except HTTPException:
            await websocket.send_json({"error": "Invalid authentication"})
            await websocket.close(code=1008)
            return
        
        # Successfully authenticated
        await websocket.send_json({"status": "connected"})
        
        # Handle messages
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "query":
                # Process a query
                query_request = QueryRequest(query=data.get("query", ""))
                response = await query(query_request, auth_data)
                await websocket.send_json(response.dict())
            
            elif data.get("type") == "generate_dashboard":
                # Generate a dashboard
                dashboard_request = DashboardRequest(
                    description=data.get("description", ""),
                    room=data.get("room"),
                    entities=data.get("entities", []),
                )
                response = await generate_dashboard(dashboard_request, auth_data)
                await websocket.send_json(response)
            
            elif data.get("type") == "generate_automation":
                # Generate an automation
                automation_request = AutomationRequest(
                    description=data.get("description", ""),
                    entities=data.get("entities", []),
                )
                response = await generate_automation(automation_request, auth_data)
                await websocket.send_json(response)
            
            else:
                await websocket.send_json({"error": "Unknown message type"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass

if __name__ == "__main__":
    # Run the server
    ssl_context = None
    if config.enable_ssl:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(config.ssl_cert_path, config.ssl_key_path)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.port,
        ssl_certfile=config.ssl_cert_path if config.enable_ssl else None,
        ssl_keyfile=config.ssl_key_path if config.enable_ssl else None,
        log_level="info",
        reload=config.debug,
    )
