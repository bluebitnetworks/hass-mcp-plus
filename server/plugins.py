#!/usr/bin/env python3
"""
Plugin System for the Advanced MCP Server.

This module implements a flexible plugin system that allows extending server 
functionality with custom plugins.
"""

import os
import sys
import importlib.util
import inspect
import logging
import asyncio
from typing import Dict, List, Optional, Union, Any, Type, Callable

from schema import ContextData

logger = logging.getLogger("mcp_server.plugins")

class Plugin:
    """
    Base class for all plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration
        """
        self.name = self.__class__.__name__
        self.config = config or {}
        self.enabled = True
        logger.debug(f"Initialized plugin: {self.name}")
    
    async def initialize(self) -> bool:
        """
        Initialize the plugin. This is called after the plugin is loaded.
        
        Override this method to perform plugin initialization.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin. This is called before the plugin is unloaded.
        
        Override this method to perform plugin cleanup.
        
        Returns:
            True if shutdown was successful, False otherwise
        """
        return True
    
    async def process_query(self, query: str, context: ContextData) -> ContextData:
        """
        Process a query and update the context.
        
        Override this method to perform plugin-specific processing.
        
        Args:
            query: User query
            context: Current context
            
        Returns:
            Updated context
        """
        return context
    
    async def register_hooks(self) -> Dict[str, Callable]:
        """
        Register hooks for plugin integration points.
        
        Override this method to register plugin hooks.
        
        Returns:
            Dictionary of hook names to hook functions
        """
        return {}
    
    def __str__(self) -> str:
        """Return string representation of the plugin."""
        return f"{self.name} (enabled: {self.enabled})"


class PluginManager:
    """
    Manager for loading, unloading, and managing plugins.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Plugin Manager.
        
        Args:
            config: Configuration for the plugin manager
        """
        self.config = config or {}
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_dir = self.config.get("plugin_dir", "/plugins")
        self.enabled_plugins = self.config.get("enabled_plugins", [])
        
        logger.info(f"Initialized Plugin Manager with plugin directory: {self.plugin_dir}")
    
    async def load_plugins(self) -> bool:
        """
        Load all plugins from the plugin directory.
        
        Returns:
            True if all plugins were loaded successfully, False otherwise
        """
        try:
            # Load built-in plugins first
            await self._load_builtin_plugins()
            
            # Load plugins from the plugin directory
            if os.path.exists(self.plugin_dir):
                for filename in os.listdir(self.plugin_dir):
                    if filename.endswith(".py") and not filename.startswith("_"):
                        plugin_name = filename[:-3]
                        
                        # Skip if not in enabled plugins list
                        if self.enabled_plugins and plugin_name not in self.enabled_plugins:
                            logger.debug(f"Skipping plugin {plugin_name} (not in enabled plugins list)")
                            continue
                        
                        await self._load_plugin_from_file(os.path.join(self.plugin_dir, filename))
            else:
                logger.warning(f"Plugin directory not found: {self.plugin_dir}")
            
            # Register all plugin hooks
            await self._register_all_hooks()
            
            logger.info(f"Loaded {len(self.plugins)} plugins")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugins: {str(e)}", exc_info=True)
            return False
    
    async def unload_plugins(self) -> bool:
        """
        Unload all plugins.
        
        Returns:
            True if all plugins were unloaded successfully, False otherwise
        """
        try:
            # Call shutdown on all plugins
            for name, plugin in list(self.plugins.items()):
                try:
                    logger.debug(f"Shutting down plugin: {name}")
                    await plugin.shutdown()
                    del self.plugins[name]
                except Exception as e:
                    logger.error(f"Error shutting down plugin {name}: {str(e)}", exc_info=True)
            
            # Clear hooks
            self.hooks.clear()
            
            logger.info("All plugins unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugins: {str(e)}", exc_info=True)
            return False
    
    async def process_query(self, query: str, context: ContextData) -> ContextData:
        """
        Process a query through all plugins.
        
        Args:
            query: User query
            context: Current context
            
        Returns:
            Updated context
        """
        try:
            current_context = context
            
            # Process through all plugins
            for name, plugin in self.plugins.items():
                if not plugin.enabled:
                    continue
                
                try:
                    logger.debug(f"Processing query with plugin: {name}")
                    current_context = await plugin.process_query(query, current_context)
                except Exception as e:
                    logger.error(f"Error processing query with plugin {name}: {str(e)}", exc_info=True)
            
            return current_context
            
        except Exception as e:
            logger.error(f"Error processing query through plugins: {str(e)}", exc_info=True)
            return context
    
    async def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Call all registered hooks for a given hook name.
        
        Args:
            hook_name: Hook name to call
            *args: Arguments to pass to the hook
            **kwargs: Keyword arguments to pass to the hook
            
        Returns:
            List of results from all hooks
        """
        results = []
        
        if hook_name not in self.hooks:
            return results
        
        for hook in self.hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(*args, **kwargs)
                else:
                    result = hook(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error calling hook {hook_name}: {str(e)}", exc_info=True)
        
        return results
    
    async def register_plugin(self, plugin: Plugin) -> bool:
        """
        Register a plugin with the Plugin Manager.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if the plugin was registered successfully, False otherwise
        """
        try:
            if plugin.name in self.plugins:
                logger.warning(f"Plugin {plugin.name} is already registered")
                return False
            
            # Initialize the plugin
            success = await plugin.initialize()
            if not success:
                logger.error(f"Failed to initialize plugin: {plugin.name}")
                return False
            
            # Register the plugin
            self.plugins[plugin.name] = plugin
            
            # Register hooks
            hooks = await plugin.register_hooks()
            for hook_name, hook_func in hooks.items():
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                self.hooks[hook_name].append(hook_func)
            
            logger.info(f"Registered plugin: {plugin.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin: {str(e)}", exc_info=True)
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin from the Plugin Manager.
        
        Args:
            plugin_name: Name of the plugin to unregister
            
        Returns:
            True if the plugin was unregistered successfully, False otherwise
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} is not registered")
                return False
            
            # Get the plugin
            plugin = self.plugins[plugin_name]
            
            # Shutdown the plugin
            success = await plugin.shutdown()
            if not success:
                logger.error(f"Failed to shutdown plugin: {plugin_name}")
                return False
            
            # Unregister hooks
            hooks = await plugin.register_hooks()
            for hook_name, hook_func in hooks.items():
                if hook_name in self.hooks and hook_func in self.hooks[hook_name]:
                    self.hooks[hook_name].remove(hook_func)
                    if not self.hooks[hook_name]:
                        del self.hooks[hook_name]
            
            # Unregister the plugin
            del self.plugins[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering plugin: {str(e)}", exc_info=True)
            return False
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_name: Name of the plugin to enable
            
        Returns:
            True if the plugin was enabled successfully, False otherwise
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} is not registered")
                return False
            
            # Enable the plugin
            self.plugins[plugin_name].enabled = True
            
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling plugin: {str(e)}", exc_info=True)
            return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a plugin.
        
        Args:
            plugin_name: Name of the plugin to disable
            
        Returns:
            True if the plugin was disabled successfully, False otherwise
        """
        try:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} is not registered")
                return False
            
            # Disable the plugin
            self.plugins[plugin_name].enabled = False
            
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling plugin: {str(e)}", exc_info=True)
            return False
    
    async def _load_builtin_plugins(self) -> None:
        """Load built-in plugins."""
        # Register built-in plugins here
        await self.register_plugin(WeatherPlugin(self.config.get("weather_plugin", {})))
        await self.register_plugin(StateHistoryPlugin(self.config.get("state_history_plugin", {})))
    
    async def _load_plugin_from_file(self, filepath: str) -> None:
        """
        Load a plugin from a file.
        
        Args:
            filepath: Path to the plugin file
        """
        try:
            # Get the plugin name from the filename
            plugin_name = os.path.basename(filepath)[:-3]
            
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, filepath)
            if not spec or not spec.loader:
                logger.error(f"Could not load plugin spec from file: {filepath}")
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin and 
                    obj.__module__ == plugin_name):
                    plugin_classes.append(obj)
            
            # Register all plugin classes
            for plugin_class in plugin_classes:
                plugin_config = self.config.get(plugin_class.__name__, {})
                plugin = plugin_class(plugin_config)
                await self.register_plugin(plugin)
            
        except Exception as e:
            logger.error(f"Error loading plugin from file {filepath}: {str(e)}", exc_info=True)
    
    async def _register_all_hooks(self) -> None:
        """Register all hooks from all plugins."""
        for name, plugin in self.plugins.items():
            try:
                hooks = await plugin.register_hooks()
                for hook_name, hook_func in hooks.items():
                    if hook_name not in self.hooks:
                        self.hooks[hook_name] = []
                    self.hooks[hook_name].append(hook_func)
            except Exception as e:
                logger.error(f"Error registering hooks for plugin {name}: {str(e)}", exc_info=True)


# Built-in plugins

class WeatherPlugin(Plugin):
    """
    Built-in plugin for weather data.
    
    This plugin adds weather data to the context.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Weather plugin."""
        super().__init__(config)
        self.name = "WeatherPlugin"
    
    async def process_query(self, query: str, context: ContextData) -> ContextData:
        """Add weather data to the context."""
        # Check if the query contains weather-related words
        weather_keywords = [
            "weather", "temperature", "rain", "sunny", "forecast", 
            "humidity", "wind", "cold", "hot", "cloudy", "storm"
        ]
        
        if not any(keyword in query.lower() for keyword in weather_keywords):
            return context
        
        # If we have a Home Assistant client, try to get weather data
        from hass_client import HomeAssistantClient
        hass_client = getattr(context, "hass_client", None)
        
        if isinstance(hass_client, HomeAssistantClient):
            try:
                # Find weather entities
                states = await hass_client.get_states()
                weather_entities = [
                    state for state in states
                    if state.entity_id.startswith("weather.")
                ]
                
                if weather_entities:
                    # Add weather data to the context
                    if not hasattr(context, "plugin_data"):
                        context.plugin_data = {}
                    
                    context.plugin_data["weather"] = {
                        entity.entity_id: {
                            "state": entity.state,
                            "attributes": entity.attributes
                        }
                        for entity in weather_entities
                    }
            except Exception as e:
                logger.error(f"Error getting weather data: {str(e)}", exc_info=True)
        
        return context


class StateHistoryPlugin(Plugin):
    """
    Built-in plugin for state history.
    
    This plugin adds state history to the context.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the State History plugin."""
        super().__init__(config)
        self.name = "StateHistoryPlugin"
        self.history_days = self.config.get("history_days", 1)
        self.max_entries = self.config.get("max_entries", 10)
    
    async def process_query(self, query: str, context: ContextData) -> ContextData:
        """Add state history to the context."""
        # Check if the query contains history-related words
        history_keywords = [
            "history", "past", "before", "previously", "last time", 
            "yesterday", "earlier", "recent", "log", "record"
        ]
        
        if not any(keyword in query.lower() for keyword in history_keywords):
            return context
        
        # If we have a Home Assistant client, try to get state history
        from hass_client import HomeAssistantClient
        from datetime import datetime, timedelta
        
        hass_client = getattr(context, "hass_client", None)
        
        if isinstance(hass_client, HomeAssistantClient):
            try:
                # Get history for mentioned entities
                entities = []
                
                # Extract entity IDs from the context
                for state in getattr(context, "states", []):
                    if state.entity_id in query:
                        entities.append(state.entity_id)
                
                if entities:
                    # Get history for the entities
                    timestamp = datetime.now() - timedelta(days=self.history_days)
                    history = await hass_client.get_history(
                        timestamp=timestamp,
                        filter_entities=entities
                    )
                    
                    # Add history data to the context
                    if not hasattr(context, "plugin_data"):
                        context.plugin_data = {}
                    
                    context.plugin_data["history"] = {}
                    
                    for entity_history in history:
                        if entity_history:
                            entity_id = entity_history[0].entity_id
                            # Limit the number of entries
                            limited_history = entity_history[:self.max_entries]
                            context.plugin_data["history"][entity_id] = [
                                {
                                    "state": state.state,
                                    "last_updated": state.last_updated
                                }
                                for state in limited_history
                            ]
            except Exception as e:
                logger.error(f"Error getting state history: {str(e)}", exc_info=True)
        
        return context
