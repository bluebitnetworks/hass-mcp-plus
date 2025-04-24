#!/usr/bin/env python3
"""
Home Assistant client for the Advanced MCP Server.

This module handles communication with the Home Assistant API.
"""

import json
import logging
import asyncio
import aiohttp
import async_timeout
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime

from dataclasses import dataclass, field
from websockets.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("mcp_server.hass_client")

@dataclass
class HassEntity:
    """
    Class representing a Home Assistant entity state.
    """
    entity_id: str
    state: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    last_changed: Optional[str] = None
    last_updated: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class HomeAssistantClient:
    """
    Client for communicating with Home Assistant.
    """
    
    def __init__(
        self,
        url: str,
        token: str,
        ws_retry_delay: int = 5,
        max_retries: int = 10,
    ):
        """
        Initialize the Home Assistant client.
        
        Args:
            url: Home Assistant URL
            token: Long-lived access token
            ws_retry_delay: WebSocket retry delay in seconds
            max_retries: Maximum number of retries
        """
        self.url = url.rstrip("/")
        self.token = token
        self.ws_retry_delay = ws_retry_delay
        self.max_retries = max_retries
        
        # WebSocket connection
        self.ws = None
        self.ws_task = None
        self.ws_id = 1
        self.ws_handlers = {}
        self.ws_connected = False
        
        # State cache
        self.states = {}
        self.last_updated = None
        
        logger.info(f"Initialized Home Assistant client for {self.url}")
    
    async def connect(self):
        """Connect to Home Assistant and start the WebSocket task."""
        # Start the WebSocket task
        self.ws_task = asyncio.create_task(self._ws_loop())
        
        # Wait for the WebSocket to connect
        for _ in range(10):
            if self.ws_connected:
                break
            await asyncio.sleep(1)
        
        if not self.ws_connected:
            logger.warning("WebSocket didn't connect in time, continuing anyway")
        
        # Initial state sync
        await self.sync_states()
        
        logger.info("Home Assistant client connected")
    
    async def disconnect(self):
        """Disconnect from Home Assistant."""
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        if self.ws and not self.ws.closed:
            await self.ws.close()
        
        logger.info("Home Assistant client disconnected")
    
    async def _ws_loop(self):
        """WebSocket connection loop."""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                async with ws_connect(
                    f"{self.url.replace('http', 'ws')}/api/websocket"
                ) as ws:
                    self.ws = ws
                    logger.info("WebSocket connected")
                    
                    # Handle authentication
                    auth_msg = json.loads(await ws.recv())
                    
                    if auth_msg["type"] == "auth_required":
                        await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
                        auth_result = json.loads(await ws.recv())
                        
                        if auth_result["type"] == "auth_ok":
                            logger.info("WebSocket authenticated")
                            self.ws_connected = True
                            retry_count = 0  # Reset retry count on successful connection
                            
                            # Subscribe to state changes
                            await self._subscribe_to_events()
                            
                            # Main message loop
                            while True:
                                try:
                                    with async_timeout.timeout(60):
                                        msg = json.loads(await ws.recv())
                                        await self._handle_message(msg)
                                except asyncio.TimeoutError:
                                    # Send ping to keep the connection alive
                                    await self._send_ping()
                        else:
                            logger.error(f"WebSocket authentication failed: {auth_result}")
                            break
                    else:
                        logger.error(f"Unexpected message during authentication: {auth_msg}")
                        break
            
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}", exc_info=True)
            
            # Reconnection logic
            retry_count += 1
            logger.info(f"Reconnecting to WebSocket in {self.ws_retry_delay} seconds (retry {retry_count}/{self.max_retries})")
            self.ws_connected = False
            await asyncio.sleep(self.ws_retry_delay)
        
        logger.error(f"Max retries ({self.max_retries}) reached, giving up on WebSocket connection")
        self.ws_connected = False
    
    async def _subscribe_to_events(self):
        """Subscribe to Home Assistant events."""
        # Subscribe to state changes
        msg_id = await self._send_message({
            "type": "subscribe_events",
            "event_type": "state_changed"
        })
        
        # Wait for confirmation
        def state_sub_handler(msg):
            logger.info("Subscribed to state changes")
            return True
        
        await self._register_handler(msg_id, state_sub_handler)
    
    async def _send_message(self, message: Dict[str, Any]) -> int:
        """
        Send a message to the WebSocket.
        
        Args:
            message: Message to send
            
        Returns:
            Message ID
        """
        if not self.ws or self.ws.closed:
            raise Exception("WebSocket not connected")
        
        msg_id = self.ws_id
        self.ws_id += 1
        
        message["id"] = msg_id
        await self.ws.send(json.dumps(message))
        
        return msg_id
    
    async def _register_handler(self, msg_id: int, handler: Callable) -> Any:
        """
        Register a handler for a response and wait for it.
        
        Args:
            msg_id: Message ID
            handler: Handler function
            
        Returns:
            Result from the handler
        """
        future = asyncio.Future()
        
        def wrapper(msg):
            try:
                result = handler(msg)
                future.set_result(result)
                return True
            except Exception as e:
                future.set_exception(e)
                return True
        
        self.ws_handlers[msg_id] = wrapper
        
        try:
            with async_timeout.timeout(10):
                return await future
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response to message {msg_id}")
            self.ws_handlers.pop(msg_id, None)
            raise
    
    async def _handle_message(self, msg: Dict[str, Any]):
        """
        Handle a message from the WebSocket.
        
        Args:
            msg: Message to handle
        """
        if "id" in msg and msg["id"] in self.ws_handlers:
            handler = self.ws_handlers[msg["id"]]
            if handler(msg):
                self.ws_handlers.pop(msg["id"], None)
        
        elif msg.get("type") == "event" and msg.get("event", {}).get("event_type") == "state_changed":
            # Handle state change event
            event_data = msg["event"]["data"]
            entity_id = event_data["entity_id"]
            new_state = event_data["new_state"]
            
            if new_state:
                self.states[entity_id] = HassEntity(
                    entity_id=entity_id,
                    state=new_state["state"],
                    attributes=new_state["attributes"],
                    last_changed=new_state["last_changed"],
                    last_updated=new_state["last_updated"],
                    context=new_state["context"]
                )
                logger.debug(f"State updated: {entity_id} = {new_state['state']}")
            elif entity_id in self.states:
                self.states.pop(entity_id)
                logger.debug(f"Entity removed: {entity_id}")
    
    async def _send_ping(self):
        """Send a ping to keep the connection alive."""
        if self.ws and not self.ws.closed:
            await self._send_message({"type": "ping"})
    
    async def sync_states(self):
        """Sync states from Home Assistant."""
        try:
            # Get all states
            states = await self.get_states()
            
            # Update the state cache
            self.states = {state.entity_id: state for state in states}
            self.last_updated = datetime.now()
            
            logger.info(f"Synced {len(states)} states from Home Assistant")
        except Exception as e:
            logger.error(f"Error syncing states: {str(e)}", exc_info=True)
    
    async def get_states(self) -> List[HassEntity]:
        """
        Get all states from Home Assistant.
        
        Returns:
            List of entities
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.get(f"{self.url}/api/states", headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    states_json = await response.json()
                    
                    return [
                        HassEntity(
                            entity_id=state["entity_id"],
                            state=state["state"],
                            attributes=state.get("attributes", {}),
                            last_changed=state.get("last_changed"),
                            last_updated=state.get("last_updated"),
                            context=state.get("context")
                        )
                        for state in states_json
                    ]
            except Exception as e:
                logger.error(f"Error getting states: {str(e)}", exc_info=True)
                raise
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get Home Assistant configuration.
        
        Returns:
            Configuration dictionary
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.get(f"{self.url}/api/config", headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    return await response.json()
            except Exception as e:
                logger.error(f"Error getting config: {str(e)}", exc_info=True)
                raise
    
    async def get_services(self) -> Dict[str, Dict[str, Any]]:
        """
        Get available Home Assistant services.
        
        Returns:
            Dictionary of services by domain
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.get(f"{self.url}/api/services", headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    services_json = await response.json()
                    
                    # Restructure the services for easier access
                    services = {}
                    for service in services_json:
                        domain = service["domain"]
                        if domain not in services:
                            services[domain] = {}
                        
                        service_name = service["service"]
                        services[domain][service_name] = {
                            "description": service.get("description", ""),
                            "fields": service.get("fields", {})
                        }
                    
                    return services
            except Exception as e:
                logger.error(f"Error getting services: {str(e)}", exc_info=True)
                raise
    
    async def get_history(
        self,
        timestamp: datetime,
        filter_entities: Optional[List[str]] = None,
        end_time: Optional[datetime] = None,
    ) -> List[List[HassEntity]]:
        """
        Get history from Home Assistant.
        
        Args:
            timestamp: Start time
            filter_entities: List of entity IDs to filter (None for all)
            end_time: End time (None for now)
            
        Returns:
            List of entity histories
        """
        params = {
            "timestamp": timestamp.isoformat(),
        }
        
        if filter_entities:
            params["filter_entity_id"] = ",".join(filter_entities)
        
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.get(f"{self.url}/api/history/period", headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    history_json = await response.json()
                    
                    result = []
                    for entity_history in history_json:
                        entity_states = []
                        for state in entity_history:
                            entity_states.append(
                                HassEntity(
                                    entity_id=state["entity_id"],
                                    state=state["state"],
                                    attributes=state.get("attributes", {}),
                                    last_changed=state.get("last_changed"),
                                    last_updated=state.get("last_updated"),
                                    context=state.get("context")
                                )
                            )
                        result.append(entity_states)
                    
                    return result
            except Exception as e:
                logger.error(f"Error getting history: {str(e)}", exc_info=True)
                raise
    
    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Call a Home Assistant service.
        
        Args:
            domain: Service domain
            service: Service name
            service_data: Service data
            target: Service target
            
        Returns:
            True if successful
        """
        data = {
            "domain": domain,
            "service": service,
        }
        
        if service_data:
            data["service_data"] = service_data
        
        if target:
            data["target"] = target
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.post(f"{self.url}/api/services/{domain}/{service}", headers=headers, json=service_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    return True
            except Exception as e:
                logger.error(f"Error calling service: {str(e)}", exc_info=True)
                raise
    
    async def create_dashboard(self, dashboard_config: Dict[str, Any]) -> bool:
        """
        Create a dashboard in Home Assistant.
        
        Args:
            dashboard_config: Dashboard configuration
            
        Returns:
            True if successful
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.post(f"{self.url}/api/lovelace/dashboards", headers=headers, json=dashboard_config) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    return True
            except Exception as e:
                logger.error(f"Error creating dashboard: {str(e)}", exc_info=True)
                raise
    
    async def create_automation(self, automation_config: Dict[str, Any]) -> bool:
        """
        Create an automation in Home Assistant.
        
        Args:
            automation_config: Automation configuration
            
        Returns:
            True if successful
        """
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.post(f"{self.url}/api/config/automation/config/{automation_config['id']}", headers=headers, json=automation_config) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Home Assistant API: {response.status}, {error_text}")
                    
                    return True
            except Exception as e:
                logger.error(f"Error creating automation: {str(e)}", exc_info=True)
                raise
