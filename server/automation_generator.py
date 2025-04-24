#!/usr/bin/env python3
"""
Automation Generator for the Advanced MCP Server.

This module generates Home Assistant automations from natural language descriptions.
"""

import os
import json
import logging
import uuid
import re
from typing import Dict, List, Optional, Union, Any

from llm import LLMManager
from hass_client import HomeAssistantClient, HassEntity
from schema import AutomationConfig

logger = logging.getLogger("mcp_server.automation_generator")

class AutomationGenerator:
    """
    Generator for Home Assistant automations.
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        hass_client: HomeAssistantClient,
        cache_dir: str = "/data/automations",
    ):
        """
        Initialize the Automation Generator.
        
        Args:
            llm_manager: LLM manager
            hass_client: Home Assistant client
            cache_dir: Directory for caching generated automations
        """
        self.llm_manager = llm_manager
        self.hass_client = hass_client
        self.cache_dir = cache_dir
        
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("Initialized Automation Generator")
    
    async def generate(
        self,
        description: str,
        states: List[HassEntity],
        entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an automation from a natural language description.
        
        Args:
            description: Automation description
            states: List of Home Assistant entity states
            entities: Optional list of entity IDs to include
            
        Returns:
            Automation configuration
        """
        try:
            logger.info(f"Generating automation for description: {description}")
            
            # Filter entities if specified
            if entities:
                filtered_states = [
                    state for state in states
                    if state.entity_id in entities
                ]
            else:
                filtered_states = states
            
            # Get available services
            services = await self.hass_client.get_services()
            
            # Create a prompt for the LLM
            prompt = self._create_automation_prompt(description, filtered_states, services)
            
            # Generate the automation configuration
            response = await self.llm_manager.generate_text(prompt)
            
            # Extract the JSON configuration
            automation_config = self._extract_json(response)
            
            if not automation_config:
                raise ValueError("Failed to generate valid automation configuration")
            
            # Add an ID if not present
            if "id" not in automation_config:
                automation_config["id"] = f"mcp_generated_{uuid.uuid4().hex[:8]}"
            
            # Save the automation to cache
            self._save_automation(automation_config)
            
            return automation_config
        
        except Exception as e:
            logger.error(f"Error generating automation: {str(e)}", exc_info=True)
            raise
    
    async def install(self, automation_id: str) -> bool:
        """
        Install a generated automation in Home Assistant.
        
        Args:
            automation_id: Automation ID
            
        Returns:
            True if successful
        """
        try:
            # Load the automation from cache
            automation_config = self._load_automation(automation_id)
            
            if not automation_config:
                raise ValueError(f"Automation not found: {automation_id}")
            
            # Install the automation
            success = await self.hass_client.create_automation(automation_config)
            
            if success:
                logger.info(f"Automation installed: {automation_id}")
            else:
                logger.error(f"Failed to install automation: {automation_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error installing automation: {str(e)}", exc_info=True)
            raise
    
    def explain(self, automation_config: Dict[str, Any]) -> str:
        """
        Explain an automation in plain language.
        
        Args:
            automation_config: Automation configuration
            
        Returns:
            Plain language explanation
        """
        try:
            # Basic explanation template
            explanation = f"Automation: {automation_config.get('alias', 'Unnamed')}\n\n"
            
            # Add description if available
            if "description" in automation_config:
                explanation += f"{automation_config['description']}\n\n"
            else:
                explanation += "This automation will:\n\n"
            
            # Explain triggers
            explanation += "Triggers:\n"
            for trigger in automation_config.get("trigger", []):
                trigger_type = trigger.get("platform", "unknown")
                
                if trigger_type == "state":
                    entity_id = trigger.get("entity_id", "unknown")
                    from_state = trigger.get("from", "any")
                    to_state = trigger.get("to", "any")
                    explanation += f"- When {entity_id} changes"
                    if from_state != "any" and to_state != "any":
                        explanation += f" from {from_state} to {to_state}"
                    elif from_state != "any":
                        explanation += f" from {from_state}"
                    elif to_state != "any":
                        explanation += f" to {to_state}"
                    explanation += "\n"
                
                elif trigger_type == "numeric_state":
                    entity_id = trigger.get("entity_id", "unknown")
                    below = trigger.get("below")
                    above = trigger.get("above")
                    if below and above:
                        explanation += f"- When {entity_id} is between {above} and {below}\n"
                    elif below:
                        explanation += f"- When {entity_id} is below {below}\n"
                    elif above:
                        explanation += f"- When {entity_id} is above {above}\n"
                    else:
                        explanation += f"- When {entity_id} changes numerically\n"
                
                elif trigger_type == "time":
                    at_time = trigger.get("at", "unknown")
                    explanation += f"- At {at_time}\n"
                
                elif trigger_type == "time_pattern":
                    pattern = ""
                    if "hours" in trigger:
                        pattern += f"hour={trigger['hours']} "
                    if "minutes" in trigger:
                        pattern += f"minute={trigger['minutes']} "
                    if "seconds" in trigger:
                        pattern += f"second={trigger['seconds']} "
                    explanation += f"- At time pattern: {pattern}\n"
                
                elif trigger_type == "sun":
                    event = trigger.get("event", "unknown")
                    offset = trigger.get("offset", "")
                    explanation += f"- At sun{event}"
                    if offset:
                        explanation += f" with offset {offset}"
                    explanation += "\n"
                
                else:
                    explanation += f"- {trigger_type} trigger\n"
            
            # Explain conditions if present
            if "condition" in automation_config and automation_config["condition"]:
                explanation += "\nConditions:\n"
                for condition in automation_config["condition"]:
                    condition_type = condition.get("condition", "unknown")
                    
                    if condition_type == "state":
                        entity_id = condition.get("entity_id", "unknown")
                        state = condition.get("state", "unknown")
                        explanation += f"- {entity_id} is {state}\n"
                    
                    elif condition_type == "numeric_state":
                        entity_id = condition.get("entity_id", "unknown")
                        below = condition.get("below")
                        above = condition.get("above")
                        if below and above:
                            explanation += f"- {entity_id} is between {above} and {below}\n"
                        elif below:
                            explanation += f"- {entity_id} is below {below}\n"
                        elif above:
                            explanation += f"- {entity_id} is above {above}\n"
                    
                    elif condition_type == "time":
                        after = condition.get("after", "")
                        before = condition.get("before", "")
                        if after and before:
                            explanation += f"- Time is between {after} and {before}\n"
                        elif after:
                            explanation += f"- Time is after {after}\n"
                        elif before:
                            explanation += f"- Time is before {before}\n"
                    
                    elif condition_type == "sun":
                        after = condition.get("after", "")
                        before = condition.get("before", "")
                        if after and before:
                            explanation += f"- Sun is between {after} and {before}\n"
                        elif after:
                            explanation += f"- Sun is after {after}\n"
                        elif before:
                            explanation += f"- Sun is before {before}\n"
                    
                    else:
                        explanation += f"- {condition_type} condition\n"
            
            # Explain actions
            explanation += "\nActions:\n"
            for action in automation_config.get("action", []):
                action_type = action.get("service", "unknown").split(".")
                
                if len(action_type) == 2:
                    domain, service = action_type
                    entity_id = action.get("entity_id", "")
                    target = action.get("target", {})
                    
                    if entity_id:
                        explanation += f"- Call {domain}.{service} on {entity_id}"
                    elif "entity_id" in target:
                        explanation += f"- Call {domain}.{service} on {target['entity_id']}"
                    else:
                        explanation += f"- Call {domain}.{service}"
                    
                    if "data" in action and action["data"]:
                        explanation += " with:"
                        for key, value in action["data"].items():
                            explanation += f"\n  - {key}: {value}"
                    explanation += "\n"
                
                elif "delay" in action:
                    explanation += f"- Wait for {action['delay']}\n"
                
                elif "wait_template" in action:
                    explanation += f"- Wait until condition is met\n"
                
                else:
                    explanation += f"- Unknown action\n"
            
            # Add mode information
            mode = automation_config.get("mode", "single")
            if mode == "single":
                explanation += "\nThis automation runs in 'single' mode, meaning it won't start again if already running."
            elif mode == "restart":
                explanation += "\nThis automation runs in 'restart' mode, meaning it will restart if triggered while running."
            elif mode == "queued":
                explanation += "\nThis automation runs in 'queued' mode, meaning it will queue multiple triggers while running."
            elif mode == "parallel":
                explanation += "\nThis automation runs in 'parallel' mode, meaning it can run multiple instances in parallel."
            
            return explanation
        
        except Exception as e:
            logger.error(f"Error explaining automation: {str(e)}", exc_info=True)
            return "Unable to generate explanation for this automation."
    
    def _create_automation_prompt(
        self,
        description: str,
        states: List[HassEntity],
        services: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Create a prompt for generating an automation.
        
        Args:
            description: Automation description
            states: List of Home Assistant entity states
            services: Available Home Assistant services
            
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
        
        # Create a simplified services representation
        services_json = {}
        for domain, domain_services in services.items():
            services_json[domain] = list(domain_services.keys())
        
        # Create the prompt
        prompt = f"""
        You are a Home Assistant automation expert. Create an automation based on the following description:
        
        DESCRIPTION:
        {description}
        
        AVAILABLE ENTITIES:
        {json.dumps(states_json, indent=2)}
        
        AVAILABLE SERVICES:
        {json.dumps(services_json, indent=2)}
        
        Create a Home Assistant automation configuration that matches the description.
        The configuration should be valid for Home Assistant and should include only the entities that are available.
        Use appropriate trigger types, conditions, and actions based on the description.
        Make sure the automation is efficient and follows best practices.
        
        Your response should be a single valid JSON object for a Home Assistant automation configuration.
        Use the following format:
        
        ```json
        {{
            "id": "generated_unique_id",
            "alias": "Automation Name",
            "description": "Detailed description of what this automation does",
            "trigger": [
                // Triggers go here
            ],
            "condition": [
                // Optional conditions go here
            ],
            "action": [
                // Actions go here
            ],
            "mode": "single"
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
    
    def _save_automation(self, automation_config: Dict[str, Any]):
        """
        Save an automation to cache.
        
        Args:
            automation_config: Automation configuration
        """
        automation_id = automation_config["id"]
        file_path = os.path.join(self.cache_dir, f"{automation_id}.json")
        
        with open(file_path, "w") as f:
            json.dump(automation_config, f, indent=2)
    
    def _load_automation(self, automation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an automation from cache.
        
        Args:
            automation_id: Automation ID
            
        Returns:
            Automation configuration or None if not found
        """
        file_path = os.path.join(self.cache_dir, f"{automation_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r") as f:
            return json.load(f)
