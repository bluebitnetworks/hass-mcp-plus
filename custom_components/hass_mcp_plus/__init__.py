"""The Home Assistant Advanced MCP Integration."""

import logging
import os
import sys
import voluptuous as vol
from typing import Any, Dict, List, Optional, Union

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_SSL,
    CONF_TOKEN,
    Platform,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_LLM_API_KEY,
    CONF_LLM_URL,
    CONF_ENABLE_RAG,
    CONF_ENABLE_DASHBOARD_GEN,
    CONF_ENABLE_AUTOMATION_GEN,
    CONF_CHUNK_SIZE,
    DEFAULT_PORT,
    DEFAULT_HOST,
    DEFAULT_SSL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_ENABLE_RAG,
    DEFAULT_ENABLE_DASHBOARD_GEN,
    DEFAULT_ENABLE_AUTOMATION_GEN,
    DEFAULT_CHUNK_SIZE,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SENSOR]

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.Schema(
            {
                vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
                vol.Optional(CONF_SSL, default=DEFAULT_SSL): cv.boolean,
                vol.Required(CONF_TOKEN): cv.string,
                vol.Optional(CONF_LLM_PROVIDER, default=DEFAULT_LLM_PROVIDER): cv.string,
                vol.Optional(CONF_LLM_MODEL, default=DEFAULT_LLM_MODEL): cv.string,
                vol.Optional(CONF_LLM_API_KEY): cv.string,
                vol.Optional(CONF_LLM_URL): cv.string,
                vol.Optional(CONF_ENABLE_RAG, default=DEFAULT_ENABLE_RAG): cv.boolean,
                vol.Optional(CONF_ENABLE_DASHBOARD_GEN, default=DEFAULT_ENABLE_DASHBOARD_GEN): cv.boolean,
                vol.Optional(CONF_ENABLE_AUTOMATION_GEN, default=DEFAULT_ENABLE_AUTOMATION_GEN): cv.boolean,
                vol.Optional(CONF_CHUNK_SIZE, default=DEFAULT_CHUNK_SIZE): cv.positive_int,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)


async def async_setup(hass: HomeAssistant, config: Dict[str, Any]) -> bool:
    """Set up the Advanced MCP Integration component."""
    if DOMAIN not in config:
        return True

    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Advanced MCP Integration from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Get the configuration from the entry
    config = entry.data
    
    # Initialize the MCP server
    try:
        session = async_get_clientsession(hass)
        
        # Add the server binary directory to the path
        server_dir = os.path.join(os.path.dirname(__file__), "server")
        if server_dir not in sys.path:
            sys.path.append(server_dir)
        
        # Import server modules
        from server.config import AppConfig, load_config
        from server.main import start_server
        
        # Create the server configuration
        server_config = {
            "port": config.get(CONF_PORT, DEFAULT_PORT),
            "host": config.get(CONF_HOST, DEFAULT_HOST),
            "debug": False,
            "log_level": "INFO",
            
            # Home Assistant configuration
            "hass_url": hass.config.api.base_url,
            "hass_token": config.get(CONF_TOKEN),
            
            # SSL configuration
            "enable_ssl": config.get(CONF_SSL, DEFAULT_SSL),
            "ssl_cert_path": None,
            "ssl_key_path": None,
            
            # LLM configuration
            "llm_provider": config.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER),
            "llm_model": config.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
            "llm_api_key": config.get(CONF_LLM_API_KEY),
            "llm_url": config.get(CONF_LLM_URL),
            
            # Feature flags
            "enable_rag": config.get(CONF_ENABLE_RAG, DEFAULT_ENABLE_RAG),
            "enable_dashboard_gen": config.get(CONF_ENABLE_DASHBOARD_GEN, DEFAULT_ENABLE_DASHBOARD_GEN),
            "enable_automation_gen": config.get(CONF_ENABLE_AUTOMATION_GEN, DEFAULT_ENABLE_AUTOMATION_GEN),
            
            # RAG configuration
            "rag_chunk_size": config.get(CONF_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        }
        
        # Create the server config
        app_config = AppConfig(**server_config)
        
        # Start the server
        server = await start_server(app_config)
        
        # Store the server instance
        hass.data[DOMAIN][entry.entry_id] = {
            "server": server,
            "config": app_config,
        }

        # Forward the entry setup to the sensor platform
        for platform in PLATFORMS:
            hass.async_create_task(
                hass.config_entries.async_forward_entry_setup(entry, platform)
            )
            
        return True
        
    except Exception as err:
        _LOGGER.exception("Failed to start MCP server: %s", err)
        raise ConfigEntryNotReady(f"Failed to start MCP server: {err}") from err


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload the platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    # Stop the server
    if entry.entry_id in hass.data[DOMAIN]:
        server_data = hass.data[DOMAIN][entry.entry_id]
        if "server" in server_data:
            await server_data["server"].shutdown()
        
        # Remove the entry data
        hass.data[DOMAIN].pop(entry.entry_id)
        
    return unload_ok
