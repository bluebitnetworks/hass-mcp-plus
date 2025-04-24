#!/usr/bin/env python3
"""
Dashboard Generator for the Advanced MCP Server.

This module generates Home Assistant dashboards from natural language descriptions.
"""

import os
import json
import logging
import uuid
import re
from typing import Dict, List, Optional, Union, Any

from llm import LLMManager
from hass_client import HomeAssistantClient, HassEntity
from schema import DashboardConfig

logger = logging.getLogger("mcp_server.dashboard_generator")

class DashboardGenerator:
    """
    Generator for Home Assistant dashboards.
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        hass_client: HomeAssistantClient,
        cache_dir: str = "/data/dashboards",
    ):
        """
        Initialize the Dashboard Generator.
        
        Args:
            llm_manager: LLM manager
            hass_client: Home Assistant client
            cache_dir: Directory for caching generated dashboards
        """
        self.llm_manager = llm_manager
        self.hass_client = hass_client
        self.cache_dir = cache_dir
        
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Initialized Dashboard Generator")
    
    async def generate(
        self,
        description: str,
        states: List[HassEntity],
        room: Optional[str] = None,
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a dashboard from a natural language description.
        
        Args:
            description: Dashboard description
            states: List of Home Assistant entity states
            room: Optional room name to filter entities
            entities: Optional list of entity IDs to include
            
        Returns:
            Dashboard configuration
        """
        try:
            logger.info(f"Generating dashboard for description: {description}")
            
            # Filter entities by room if specified
            if room:
                filtered_states = [
                    state for state in states
                    if (self._entity_in_room(state, room) or
                        (entities and state.entity_id in entities))
                ]
            elif entities:
                filtered_states = [
                    state for state in states
                    if state.entity_id in entities
                ]
            else:
                filtered_states = states
            
            # Create a prompt for the LLM
            prompt = self._create_dashboard_prompt(description, filtered_states)
            
            # Generate the dashboard configuration
            response = await self.llm_manager.generate_text(prompt)
            
            # Extract the JSON configuration
            dashboard_config = self._extract_json(response)
            
            if not dashboard_config:
                raise ValueError("Failed to generate valid dashboard configuration")
            
            # Add an ID if not present
            if "id" not in dashboard_config:
                dashboard_config["id"] = f"mcp_generated_{uuid.uuid4().hex[:8]}"
            
            # Save the dashboard to cache
            self._save_dashboard(dashboard_config)
            
            return dashboard_config
        
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}", exc_info=True)
            raise
    
    async def install(self, dashboard_id: str) -> bool:
        """
        Install a generated dashboard in Home Assistant.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            True if successful
        """
        try:
            # Load the dashboard from cache
            dashboard_config = self._load_dashboard(dashboard_id)
            
            if not dashboard_config:
                raise ValueError(f"Dashboard not found: {dashboard_id}")
            
            # Install the dashboard
            success = await self.hass_client.create_dashboard(dashboard_config)
            
            if success:
                logger.info(f"Dashboard installed: {dashboard_id}")
            else:
                logger.error(f"Failed to install dashboard: {dashboard_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error installing dashboard: {str(e)}", exc_info=True)
            raise
    
    def _create_dashboard_prompt(self, description: str, states: List[HassEntity]) -> str:
        """
        Create a prompt for generating a dashboard.
        
        Args:
            description: Dashboard description
            states: List of Home Assistant entity states
            
        Returns:
            Prompt string
        """
        # Convert states to a format that's easier for the LLM to use
        states_json = []
        for state in states:
            states_json.append({
                "entity_id": state.entity_id,
                "state": state.state,
                "attributes": state.attributes
            })
        
        # Create the prompt
        prompt = f"""
        You are a Home Assistant dashboard designer. Create a dashboard based on the following description:
        
        DESCRIPTION:
        {description}
        
        AVAILABLE ENTITIES:
        {json.dumps(states_json, indent=2)}
        
        Create a Lovelace dashboard configuration that matches the description.
        The configuration should be valid for Home Assistant and should include only the entities that are available.
        Use appropriate card types for different entity types (e.g., light, sensor, switch, climate, etc.).
        You can use conditional cards, stack cards, grid cards, and other advanced cards as needed.
        Focus on creating an attractive and functional layout.
        
        Your response should be a single valid JSON object for a Home Assistant dashboard configuration.
        Use the following format:
        
        ```json
        {{
            "id": "generated_unique_id",
            "title": "Dashboard Title",
            "icon": "mdi:appropriate-icon",
            "show_in_sidebar": true,
            "views": [
                {{
                    "title": "View Title",
                    "icon": "mdi:appropriate-icon",
                    "cards": [
                        // Cards go here, using appropriate types for each entity
                    ]
                }}
            ]
        }}
        ```
        
        Response with ONLY the JSON object, no additional text.
        """
        
        return prompt
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract a JSON object from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON object
        """
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON without code blocks
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If all else fails, try to parse the entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    
    def _entity_in_room(self, entity: HassEntity, room: str) -> bool:
        """
        Check if an entity is in a room.
        
        Args:
            entity: Entity state
            room: Room name or area
            
        Returns:
            True if the entity is in the room
        """
        # Check if entity has an area attribute
        if "area" in entity.attributes and entity.attributes["area"].lower() == room.lower():
            return True
        
        # Check if entity has a device_class attribute with the room name
        if "device_class" in entity.attributes and room.lower() in entity.attributes["device_class"].lower():
            return True
        
        # Check if entity_id contains the room name
        if room.lower().replace(" ", "_") in entity.entity_id.lower():
            return True
        
        # Check if friendly_name contains the room name
        if "friendly_name" in entity.attributes and room.lower() in entity.attributes["friendly_name"].lower():
            return True
        
        return False
    
    def _save_dashboard(self, dashboard_config: Dict[str, Any]):
        """
        Save a dashboard to cache.
        
        Args:
            dashboard_config: Dashboard configuration
        """
        dashboard_id = dashboard_config["id"]
        file_path = os.path.join(self.cache_dir, f"{dashboard_id}.json")
        
        with open(file_path, "w") as f:
            json.dump(dashboard_config, f, indent=2)
    
    def _load_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a dashboard from cache.
        
        Args:
            dashboard_id: Dashboard ID
            
        Returns:
            Dashboard configuration or None if not found
        """
        file_path = os.path.join(self.cache_dir, f"{dashboard_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r") as f:
            return json.load(f)
