#!/usr/bin/env python3
"""
Data schema for the Advanced MCP Server.

This module defines the data models used for communication and storage.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class EntityInfo(BaseModel):
    """Information about a Home Assistant entity."""
    entity_id: str
    relevance: Optional[str] = None
    reason: Optional[str] = None

class ServiceCallAction(BaseModel):
    """Service call action."""
    type: str = "service_call"
    domain: str
    service: str
    entity_id: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    area_id: Optional[str] = None
    area_ids: Optional[List[str]] = None
    device_id: Optional[str] = None
    device_ids: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None

class WebhookAction(BaseModel):
    """Webhook action."""
    type: str = "webhook"
    webhook_id: str
    data: Optional[Dict[str, Any]] = None

class FireEventAction(BaseModel):
    """Fire event action."""
    type: str = "fire_event"
    event_type: str
    event_data: Optional[Dict[str, Any]] = None

class Action(BaseModel):
    """Action model."""
    type: str
    domain: Optional[str] = None
    service: Optional[str] = None
    entity_id: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    area_id: Optional[str] = None
    area_ids: Optional[List[str]] = None
    device_id: Optional[str] = None
    device_ids: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None
    webhook_id: Optional[str] = None
    event_type: Optional[str] = None
    event_data: Optional[Dict[str, Any]] = None

class ContextData(BaseModel):
    """Context data for MCP requests."""
    states: Optional[List[Any]] = None
    timestamp: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    rag_results: Optional[List[Dict[str, Any]]] = None
    plugin_data: Optional[Dict[str, Any]] = None
    history: Optional[Dict[str, Any]] = None
    
    def copy(self):
        """Create a copy of the context data."""
        return ContextData(
            states=self.states,
            timestamp=self.timestamp,
            user=self.user,
            rag_results=self.rag_results,
            plugin_data=self.plugin_data,
            history=self.history,
        )

class MCPRequest(BaseModel):
    """MCP request model."""
    query: str
    context: ContextData

class MCPResponse(BaseModel):
    """MCP response model."""
    response: str
    actions: Optional[List[Action]] = Field(default_factory=list)
    raw_llm_response: Optional[str] = None

class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    context_level: Optional[str] = "default"  # minimal, default, full
    include_history: Optional[bool] = False
    history_days: Optional[int] = 7

class DashboardRequest(BaseModel):
    """Dashboard generation request."""
    description: str
    room: Optional[str] = None
    entities: Optional[List[str]] = None
    style: Optional[str] = None
    layout: Optional[str] = None

class AutomationRequest(BaseModel):
    """Automation generation request."""
    description: str
    entities: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    advanced: Optional[bool] = False

class ScriptRequest(BaseModel):
    """Script generation request."""
    description: str
    entities: Optional[List[str]] = None
    parameters: Optional[List[str]] = None
    advanced: Optional[bool] = False

class EntityState(BaseModel):
    """Entity state model."""
    entity_id: str
    state: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    last_changed: Optional[str] = None
    last_updated: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    id: str
    title: str
    views: List[Dict[str, Any]]
    resource_url: Optional[str] = None
    show_in_sidebar: bool = True
    require_admin: bool = False
    mode: str = "storage"
    icon: Optional[str] = None
    filename: Optional[str] = None

class AutomationConfig(BaseModel):
    """Automation configuration."""
    id: str
    alias: str
    description: str
    trigger: List[Dict[str, Any]]
    condition: Optional[List[Dict[str, Any]]] = None
    action: List[Dict[str, Any]]
    mode: str = "single"
    enabled: bool = True

class ScriptConfig(BaseModel):
    """Script configuration."""
    id: str
    alias: str
    description: str
    sequence: List[Dict[str, Any]]
    mode: str = "single"
    fields: Optional[Dict[str, Dict[str, Any]]] = None
    enabled: bool = True
