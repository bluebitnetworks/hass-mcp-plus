#!/usr/bin/env python3
"""
Configuration handling for the Advanced MCP Server.
"""

import os
import json
import logging
import yaml
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("mcp_server.config")

class LLMConfig(BaseModel):
    """Configuration for the LLM."""
    provider: str = Field(default="local", description="LLM provider (local, ollama, openai, anthropic)")
    model: str = Field(default="llama3", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    url: Optional[str] = Field(default=None, description="URL for the model API")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")

class RAGConfig(BaseModel):
    """Configuration for the RAG system."""
    enabled: bool = Field(default=True, description="Whether RAG is enabled")
    vector_store: str = Field(default="chroma", description="Vector store type")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    chunk_size: int = Field(default=512, description="Chunk size for text splitting")
    index_path: str = Field(default="/data/rag_index", description="Path to the vector index")
    update_interval: int = Field(default=3600, description="Interval in seconds for index updates")

class SecurityConfig(BaseModel):
    """Security configuration."""
    require_confirmation: bool = Field(default=True, description="Require confirmation for actions")
    allowed_domains: List[str] = Field(default=["*"], description="Allowed domains for CORS")
    api_token: Optional[str] = Field(default=None, description="API token for authentication")
    token_expiry: int = Field(default=86400, description="Token expiry in seconds")
    restricted_entities: List[str] = Field(default=[], description="Restricted entity IDs")

class FeatureConfig(BaseModel):
    """Feature configuration."""
    dashboard_generation: bool = Field(default=True, description="Enable dashboard generation")
    automation_creation: bool = Field(default=True, description="Enable automation creation")
    entity_suggestions: bool = Field(default=True, description="Enable entity suggestions")
    voice_control: bool = Field(default=False, description="Enable voice control")
    image_processing: bool = Field(default=False, description="Enable image processing")

class HomeAssistantConfig(BaseModel):
    """Home Assistant connection configuration."""
    url: str = Field(..., description="Home Assistant URL")
    token: str = Field(..., description="Home Assistant access token")
    ws_retry_delay: int = Field(default=5, description="WebSocket retry delay in seconds")
    max_retries: int = Field(default=10, description="Maximum number of retries")

class AppConfig(BaseModel):
    """Main application configuration."""
    # Server configuration
    port: int = Field(default=8787, description="Port to run the server on")
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Home Assistant configuration
    hass_url: str = Field(..., description="Home Assistant URL")
    hass_token: str = Field(..., description="Home Assistant access token")
    
    # SSL configuration
    enable_ssl: bool = Field(default=False, description="Enable SSL")
    ssl_cert_path: Optional[str] = Field(default=None, description="Path to SSL certificate")
    ssl_key_path: Optional[str] = Field(default=None, description="Path to SSL key")
    
    # LLM configuration
    llm_provider: str = Field(default="local", description="LLM provider")
    llm_model: str = Field(default="llama3", description="Model name")
    llm_api_key: Optional[str] = Field(default=None, description="API key for the provider")
    llm_url: Optional[str] = Field(default=None, description="URL for the model API")
    llm_temperature: float = Field(default=0.7, description="Temperature for generation")
    llm_max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    
    # RAG configuration
    enable_rag: bool = Field(default=True, description="Enable RAG")
    rag_vector_store: str = Field(default="chroma", description="Vector store type")
    rag_embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    rag_chunk_size: int = Field(default=512, description="Chunk size for text splitting")
    rag_index_path: str = Field(default="/data/rag_index", description="Path to the vector index")
    rag_update_interval: int = Field(default=3600, description="Interval in seconds for index updates")
    
    # Feature flags
    enable_dashboard_gen: bool = Field(default=True, description="Enable dashboard generation")
    enable_automation_gen: bool = Field(default=True, description="Enable automation creation")
    enable_entity_suggestions: bool = Field(default=True, description="Enable entity suggestions")
    enable_voice_processing: bool = Field(default=False, description="Enable voice control")
    enable_image_processing: bool = Field(default=False, description="Enable image processing")
    
    # Security configuration
    require_confirmation: bool = Field(default=True, description="Require confirmation for actions")
    allowed_domains: List[str] = Field(default=["*"], description="Allowed domains for CORS")
    api_token: Optional[str] = Field(default=None, description="API token for authentication")
    token_expiry: int = Field(default=86400, description="Token expiry in seconds")
    restricted_entities: List[str] = Field(default=[], description="Restricted entity IDs")
    
    # Plugin configuration
    plugin_dir: str = Field(default="/plugins", description="Directory for plugins")
    enabled_plugins: List[str] = Field(default=[], description="Enabled plugin names")
    
    @validator('ssl_cert_path', 'ssl_key_path')
    def validate_ssl_paths(cls, v, values):
        """Validate SSL paths if SSL is enabled."""
        if values.get('enable_ssl') and not v:
            raise ValueError("SSL certificate and key paths are required when SSL is enabled")
        return v

def load_config() -> AppConfig:
    """Load configuration from environment variables and config file."""
    # Default config path
    config_path = os.environ.get("CONFIG_PATH", "/config/config.yaml")
    
    config_data = {}
    
    # Load from config file if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config_data = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
    
    # Override with environment variables
    env_config = {
        "port": int(os.environ.get("PORT", config_data.get("port", 8787))),
        "host": os.environ.get("HOST", config_data.get("host", "0.0.0.0")),
        "debug": os.environ.get("DEBUG", str(config_data.get("debug", False))).lower() in ("true", "1", "yes"),
        "log_level": os.environ.get("LOG_LEVEL", config_data.get("log_level", "INFO")),
        
        # Home Assistant configuration
        "hass_url": os.environ.get("HASS_URL", config_data.get("hass_url")),
        "hass_token": os.environ.get("HASS_TOKEN", config_data.get("hass_token")),
        
        # SSL configuration
        "enable_ssl": os.environ.get("ENABLE_SSL", str(config_data.get("enable_ssl", False))).lower() in ("true", "1", "yes"),
        "ssl_cert_path": os.environ.get("SSL_CERT_PATH", config_data.get("ssl_cert_path")),
        "ssl_key_path": os.environ.get("SSL_KEY_PATH", config_data.get("ssl_key_path")),
        
        # LLM configuration
        "llm_provider": os.environ.get("LLM_PROVIDER", config_data.get("llm_provider", "local")),
        "llm_model": os.environ.get("LLM_MODEL", config_data.get("llm_model", "llama3")),
        "llm_api_key": os.environ.get("LLM_API_KEY", config_data.get("llm_api_key")),
        "llm_url": os.environ.get("LLM_URL", config_data.get("llm_url")),
        "llm_temperature": float(os.environ.get("LLM_TEMPERATURE", config_data.get("llm_temperature", 0.7))),
        "llm_max_tokens": int(os.environ.get("LLM_MAX_TOKENS", config_data.get("llm_max_tokens", 1024))),
        
        # RAG configuration
        "enable_rag": os.environ.get("ENABLE_RAG", str(config_data.get("enable_rag", True))).lower() in ("true", "1", "yes"),
        "rag_vector_store": os.environ.get("RAG_VECTOR_STORE", config_data.get("rag_vector_store", "chroma")),
        "rag_embedding_model": os.environ.get("RAG_EMBEDDING_MODEL", config_data.get("rag_embedding_model", "all-MiniLM-L6-v2")),
        "rag_chunk_size": int(os.environ.get("RAG_CHUNK_SIZE", config_data.get("rag_chunk_size", 512))),
        "rag_index_path": os.environ.get("RAG_INDEX_PATH", config_data.get("rag_index_path", "/data/rag_index")),
        "rag_update_interval": int(os.environ.get("RAG_UPDATE_INTERVAL", config_data.get("rag_update_interval", 3600))),
        
        # Feature flags
        "enable_dashboard_gen": os.environ.get("ENABLE_DASHBOARD_GEN", str(config_data.get("enable_dashboard_gen", True))).lower() in ("true", "1", "yes"),
        "enable_automation_gen": os.environ.get("ENABLE_AUTOMATION_GEN", str(config_data.get("enable_automation_gen", True))).lower() in ("true", "1", "yes"),
        "enable_entity_suggestions": os.environ.get("ENABLE_ENTITY_SUGGESTIONS", str(config_data.get("enable_entity_suggestions", True))).lower() in ("true", "1", "yes"),
        "enable_voice_processing": os.environ.get("ENABLE_VOICE_PROCESSING", str(config_data.get("enable_voice_processing", False))).lower() in ("true", "1", "yes"),
        "enable_image_processing": os.environ.get("ENABLE_IMAGE_PROCESSING", str(config_data.get("enable_image_processing", False))).lower() in ("true", "1", "yes"),
        
        # Security configuration
        "require_confirmation": os.environ.get("REQUIRE_CONFIRMATION", str(config_data.get("require_confirmation", True))).lower() in ("true", "1", "yes"),
        "allowed_domains": os.environ.get("ALLOWED_DOMAINS", config_data.get("allowed_domains", ["*"])),
        "api_token": os.environ.get("API_TOKEN", config_data.get("api_token")),
        "token_expiry": int(os.environ.get("TOKEN_EXPIRY", config_data.get("token_expiry", 86400))),
        "restricted_entities": os.environ.get("RESTRICTED_ENTITIES", config_data.get("restricted_entities", [])),
        
        # Plugin configuration
        "plugin_dir": os.environ.get("PLUGIN_DIR", config_data.get("plugin_dir", "/plugins")),
        "enabled_plugins": os.environ.get("ENABLED_PLUGINS", config_data.get("enabled_plugins", [])),
    }
    
    # Convert string lists to actual lists if they came from environment variables
    for key in ["allowed_domains", "restricted_entities", "enabled_plugins"]:
        if isinstance(env_config[key], str):
            env_config[key] = [item.strip() for item in env_config[key].split(",") if item.strip()]
    
    # Validate required fields
    if not env_config["hass_url"]:
        raise ValueError("Home Assistant URL (HASS_URL) must be specified")
    
    if not env_config["hass_token"]:
        raise ValueError("Home Assistant access token (HASS_TOKEN) must be specified")
    
    # Create and validate the config
    return AppConfig(**env_config)
