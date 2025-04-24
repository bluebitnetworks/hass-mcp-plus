"""Sensor platform for the Home Assistant Advanced MCP Integration."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_SSL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import (
    DOMAIN,
    SENSOR_NAME,
    ICON_SERVER,
    ATTR_SERVER_STATUS,
    ATTR_SERVER_URL,
    ATTR_LLM_PROVIDER,
    ATTR_LLM_MODEL,
    ATTR_FEATURES,
    ATTR_LAST_QUERY,
    ATTR_LAST_RESPONSE,
    ATTR_REQUEST_COUNT,
    ATTR_UPTIME,
    CONF_LLM_PROVIDER,
    CONF_LLM_MODEL,
    CONF_ENABLE_RAG,
    CONF_ENABLE_DASHBOARD_GEN,
    CONF_ENABLE_AUTOMATION_GEN,
    API_SERVER_INFO,
    API_HEALTH,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Advanced MCP sensor."""
    
    # Get the server info from entry data
    server_data = hass.data[DOMAIN][entry.entry_id]
    config = entry.data
    
    # Determine server URL
    host = config.get(CONF_HOST)
    port = config.get(CONF_PORT)
    use_ssl = config.get(CONF_SSL, False)
    scheme = "https" if use_ssl else "http"
    server_url = f"{scheme}://{host}:{port}"
    
    # Create sensor coordinator for regular data updates
    coordinator = MCPServerCoordinator(
        hass=hass,
        server_url=server_url,
        logger=_LOGGER,
        name="MCP Server Status",
        server_data=server_data,
    )
    
    # Fetch initial data
    await coordinator.async_config_entry_first_refresh()
    
    # Add entity to Home Assistant
    async_add_entities([MCPServerSensor(coordinator, entry)], True)


class MCPServerCoordinator(DataUpdateCoordinator):
    """Coordinator for fetching MCP server data."""

    def __init__(
        self,
        hass: HomeAssistant,
        server_url: str,
        logger: logging.Logger,
        name: str,
        server_data: Dict[str, Any],
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            logger,
            name=name,
            update_interval=None,  # Don't automatically update
        )
        self.server_url = server_url
        self.server_data = server_data
        self.start_time = datetime.now()
        self.request_count = 0
        self.last_query = None
        self.last_response = None
        
    @property
    def uptime(self) -> str:
        """Get the server uptime."""
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{uptime.days}d {hours}h {minutes}m {seconds}s"
    
    async def _get_server_status(self) -> str:
        """Get the server status."""
        try:
            # Try to reach the health endpoint
            response = await self.hass.async_add_executor_job(
                lambda: self.server_data["server"].get_health()
            )
            return response.get("status", "unknown")
        except Exception:
            return "offline"
    
    async def _get_server_info(self) -> Dict[str, Any]:
        """Get the server info."""
        try:
            # Try to reach the info endpoint
            response = await self.hass.async_add_executor_job(
                lambda: self.server_data["server"].get_info()
            )
            return response
        except Exception:
            return {
                "name": "Home Assistant Advanced MCP Server",
                "version": "unknown",
                "features": {
                    "rag": False,
                    "dashboard_generation": False,
                    "automation_generation": False,
                },
                "llm_provider": "unknown",
                "llm_model": "unknown",
            }
    
    async def _async_update_data(self) -> Dict[str, Any]:
        """Fetch data from the MCP server."""
        status = await self._get_server_status()
        info = await self._get_server_info()
        
        return {
            ATTR_SERVER_STATUS: status,
            ATTR_SERVER_URL: self.server_url,
            ATTR_LLM_PROVIDER: info.get("llm_provider", "unknown"),
            ATTR_LLM_MODEL: info.get("llm_model", "unknown"),
            ATTR_FEATURES: info.get("features", {}),
            ATTR_LAST_QUERY: self.last_query,
            ATTR_LAST_RESPONSE: self.last_response,
            ATTR_REQUEST_COUNT: self.request_count,
            ATTR_UPTIME: self.uptime,
        }
    
    def update_query_data(self, query: str, response: str) -> None:
        """Update the query and response data."""
        self.last_query = query
        self.last_response = response
        self.request_count += 1
        
        # Immediately update the coordinator data
        self.async_set_updated_data(self._async_update_data())


class MCPServerSensor(CoordinatorEntity, SensorEntity):
    """Sensor representing the Advanced MCP server."""

    def __init__(
        self,
        coordinator: MCPServerCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        
        self._entry = entry
        self._attr_name = SENSOR_NAME
        self._attr_icon = ICON_SERVER
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_unique_id = f"{entry.entry_id}_status"
        
        # Set device info
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="Home Assistant Advanced MCP Integration",
            manufacturer="Custom Integration",
            model="MCP Server",
            sw_version="1.0.0",
        )
    
    @property
    def available(self) -> bool:
        """Return if the sensor is available."""
        if self.coordinator.data is None:
            return False
        return self.coordinator.data.get(ATTR_SERVER_STATUS) == "healthy"
    
    @property
    def state(self) -> str:
        """Return the state of the sensor."""
        if self.coordinator.data is None:
            return "unknown"
        return self.coordinator.data.get(ATTR_SERVER_STATUS, "unknown")
    
    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return entity specific state attributes."""
        if self.coordinator.data is None:
            return {}
        
        return {
            ATTR_SERVER_URL: self.coordinator.data.get(ATTR_SERVER_URL),
            ATTR_LLM_PROVIDER: self.coordinator.data.get(ATTR_LLM_PROVIDER),
            ATTR_LLM_MODEL: self.coordinator.data.get(ATTR_LLM_MODEL),
            ATTR_FEATURES: self.coordinator.data.get(ATTR_FEATURES),
            ATTR_LAST_QUERY: self.coordinator.data.get(ATTR_LAST_QUERY),
            ATTR_LAST_RESPONSE: self.coordinator.data.get(ATTR_LAST_RESPONSE),
            ATTR_REQUEST_COUNT: self.coordinator.data.get(ATTR_REQUEST_COUNT),
            ATTR_UPTIME: self.coordinator.data.get(ATTR_UPTIME),
        }
