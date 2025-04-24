# HASS-MCP+ Plugins

This directory contains plugins for the Home Assistant Advanced MCP Integration. Plugins extend the functionality of the MCP server with custom capabilities.

## What Are Plugins?

Plugins are Python modules that implement the `Plugin` interface defined in `server/plugins.py`. They can:

- Add new context to LLM queries
- Process user queries for special commands
- Integrate with external systems
- Add custom tools or capabilities to the MCP server

## Creating a Custom Plugin

1. Create a new Python file in this directory (e.g., `my_custom_plugin.py`)
2. Import the Plugin base class from the server
3. Create a class that inherits from `Plugin`
4. Implement the required methods
5. The plugin will be automatically loaded on server startup

### Example Plugin

```python
from server.plugins import Plugin

class WeatherForecastPlugin(Plugin):
    """Plugin for providing detailed weather forecasts."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = "WeatherForecastPlugin"
        # Initialize any resources needed
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        # Setup code goes here
        return True
    
    async def shutdown(self) -> bool:
        """Shutdown the plugin."""
        # Cleanup code goes here
        return True
    
    async def process_query(self, query: str, context) -> object:
        """Process a query and update the context."""
        # Check if this is a weather-related query
        if "weather" in query.lower() or "forecast" in query.lower():
            # Add weather data to the context
            if not hasattr(context, "plugin_data"):
                context.plugin_data = {}
            
            context.plugin_data["weather"] = {
                # Your weather data here
            }
        
        return context
    
    async def register_hooks(self) -> dict:
        """Register hooks for plugin integration points."""
        return {
            "before_llm_call": self.enhance_prompt,
            "after_llm_call": self.process_response
        }
    
    async def enhance_prompt(self, prompt, context):
        """Example hook that enhances the prompt."""
        # Modify the prompt if needed
        return prompt
    
    async def process_response(self, response, context):
        """Example hook that processes the response."""
        # Modify the response if needed
        return response




        Configuration
Plugins can be configured in the .env file or through the web UI:
# Enable specific plugins
ENABLED_PLUGINS=WeatherPlugin,StateHistoryPlugin

# Plugin-specific configuration
WEATHER_PLUGIN_API_KEY=your_weather_api_key
Available Hooks
Plugins can register hooks for various integration points:

before_llm_call: Called before the query is sent to the LLM
after_llm_call: Called after the response is received from the LLM
before_action: Called before an action is executed
after_action: Called after an action is executed
startup: Called when the server starts
shutdown: Called when the server shuts down

Best Practices

Performance: Keep plugins lightweight and efficient
Error Handling: Implement proper error handling to prevent crashes
Documentation: Document your plugin's functionality and configuration
Testing: Write tests for your plugin using pytest
Security: Validate all inputs and outputs

Built-in Plugins
The following plugins are built into the MCP server:

WeatherPlugin: Adds weather data to the context
StateHistoryPlugin: Adds historical state data to the context

You can use these as examples for creating your own plugins.