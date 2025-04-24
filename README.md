# Home Assistant Advanced MCP Integration (HASS-MCP+)

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Compose](https://img.shields.io/badge/docker--compose-%3E%3D1.27.0-blue.svg)](https://docs.docker.com/compose/)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-%3E%3D2025.4.1-blue.svg)](https://www.home-assistant.io/)

A comprehensive Model Context Protocol (MCP) server integration for Home Assistant that enables advanced AI interactions with your smart home system. This integration goes beyond basic LLM interactions to provide a complete solution for AI-powered home automation.

## 🌟 Features

- **Easy Installation**: Install via HACS or Docker Compose with minimal configuration
- **Multi-LLM Support**: Connect to local or remote LLMs (Ollama, OpenAI, Anthropic, etc.)
- **RAG Integration**: Built-in Retrieval Augmented Generation for context-aware responses
- **Natural Language Home Control**: Control your smart home with conversational language
- **AI-Generated Dashboards**: Automatically create dashboard UIs from natural language descriptions
- **Automation Assistant**: Generate automations, scripts, and scenes from natural language
- **Multi-Modal Input**: Process text, audio, and even image inputs for context
- **Semantic Entity Understanding**: Understands relationships between devices and rooms
- **Historical Context**: Considers historical data when making decisions
- **Voice Assistant Integration**: Works with voice assistants for hands-free control
- **Security-Focused**: Fine-grained permission controls and secure communication
- **Extensible Plugin System**: Add custom capabilities through plugins

## 📋 Requirements

- Home Assistant OS or Core (version 2025.4.1 or newer)
- Docker (if using the Docker installation method)
- Python 3.11 or newer (if using the HACS installation method)
- At least 2GB of RAM for basic functionality (4GB+ recommended for local LLM integration)

## 🚀 Installation

### Option 1: Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/bluebitnetworks/hass-mcp-plus.git
   cd hass-mcp-plus
   ```

2. Create a `.env` file with your configuration:
   ```
   HASS_URL=http://homeassistant:8123
   HASS_TOKEN=your_long_lived_access_token
   LLM_PROVIDER=ollama  # Options: ollama, openai, anthropic, local
   LLM_URL=http://ollama:11434  # Only for ollama or local
   LLM_MODEL=llama3  # Model name
   TZ=America/New_York
   ```

3. Start the container:

   ```bash
   docker-compose up -d
   ```

4. Add the integration to Home Assistant via the UI:
   - Go to Settings → Devices & Services → Add Integration
   - Search for "MCP Server" and select it
   - Enter the URL: `http://hass-mcp-plus:8787` (or your container IP if not on the same network)

### Option 2: HACS Installation

1. Make sure [HACS](https://hacs.xyz/) is installed in your Home Assistant instance
2. Add this repository as a custom repository in HACS:
   - Go to HACS → Integrations → ⋮ → Custom repositories
   - Add `https://github.com/bluebitnetworks/hass-mcp-plus` with category "Integration"
3. Install the "Home Assistant Advanced MCP Integration" from HACS
4. Restart Home Assistant
5. Add the integration via the UI:
   - Go to Settings → Devices & Services → Add Integration
   - Search for "Advanced MCP Server" and add it
   - Follow the configuration steps

## ⚙️ Configuration

### Basic Configuration (UI)

After installation, configure the integration through the Home Assistant UI. You'll need to provide:

- Connection type (local or remote)
- LLM provider and API keys (if using external LLMs)
- Features to enable (dashboard generation, automation creation, etc.)
- Context level (how much data to include in LLM context)

### Advanced Configuration (YAML)

For advanced configuration, add the following to your `configuration.yaml`:

```yaml
mcp_server_plus:
  url: http://localhost:8787
  llm:
    provider: ollama  # ollama, openai, anthropic, local
    model: llama3
    temperature: 0.7
    context_length: 4096
  features:
    dashboard_generation: true
    automation_creation: true
    entity_suggestions: true
    voice_control: true
  security:
    require_confirmation: true
    allowed_domains: ['*']
    restricted_entities: []
  rag:
    enabled: true
    vector_store: chroma
    embedding_model: all-minilm-l6-v2
    chunk_size: 512
    index_path: /config/mcp_rag_index
  history:
    days_to_include: 7
    max_entries_per_entity: 100
```

## 🔧 Usage

### Dashboard Generation

To generate a new dashboard:

1. Go to the MCP Assistant in Home Assistant
2. Type a request like:
   - "Create a dashboard for my living room with lights, TV, and temperature controls"
   - "Make me a dashboard for monitoring my home's energy usage"
   - "I need a weather dashboard with forecast and current conditions"

The assistant will generate a dashboard UI and offer to install it for you.

### Creating Automations

To create new automations:

1. Go to the MCP Assistant in Home Assistant
2. Type a request like:
   - "Turn on my living room lights when motion is detected and it's after sunset"
   - "Notify me if my front door is left open for more than 5 minutes"
   - "Set up an automation to preheat my house in the morning"

The assistant will generate the automation, explain how it works, and offer to install it for you.

### Natural Language Control

Control your home directly with natural language:

- "Make the living room warmer"
- "Turn off all lights except the kitchen"
- "Show me which windows are open"
- "Is anyone home right now?"

### Voice Integration

Connect to voice assistants for hands-free control:

1. Configure the voice platform of your choice (Google Assistant, Alexa, etc.)
2. Add the MCP webhook to your voice assistant configuration
3. Use trigger phrases like "Ask Smart Home to..." or "Tell MCP to..."

## 🧩 Plugin System

The integration includes a powerful plugin system that allows you to add custom functionality:

1. Create a new plugin file in `plugins/` directory
2. Implement the Plugin interface
3. The plugin will be automatically loaded on server start

Example plugin:

```python
from mcp_server_plus.plugin import Plugin

class WeatherForecastPlugin(Plugin):
    """Plugin for providing detailed weather forecasts."""
    
    def __init__(self, config):
        self.name = "weather_forecast"
        self.config = config
        
    async def process(self, query, context):
        # Process the query and return enhanced weather information
        # ...
        return enhanced_context
```

## 🔌 Integration with Other Systems

### LangChain Integration

Example Python code for integrating with LangChain:

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import requests

# Define tool for interacting with HASS-MCP+
def query_home_assistant(query):
    response = requests.post(
        "http://localhost:8787/api/query",
        json={"query": query},
        headers={"Authorization": "Bearer YOUR_TOKEN"}
    )
    return response.json()["response"]

tools = [
    Tool(
        name="HomeAssistant",
        func=query_home_assistant,
        description="Use this tool to interact with your smart home devices and services."
    )
]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

result = agent.run("What's the temperature in my living room and should I turn on the AC?")
```

### RAG System Connection

Connect your own RAG system:

```python
import requests
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to your vector DB
# ... your vector DB connection code here ...

# Register with HASS-MCP+
requests.post(
    "http://localhost:8787/api/plugins/register",
    json={
        "name": "custom_rag",
        "type": "rag",
        "endpoint": "http://your-rag-service:8000/query",
        "api_key": "your_api_key"
    },
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

## 🛠️ Development & Customization

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bluebitnetworks/hass-mcp-plus.git
   cd hass-mcp-plus
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration

5. Run the server:
   ```bash
   python server/main.py
   ```

### Customizing the Web UI

The web UI is built with Vue.js and can be customized:

1. Navigate to the webui directory:
   ```bash
   cd webui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Make your changes and build:
   ```bash
   npm run build
   ```

## 🔍 Troubleshooting

### Common Issues

#### Connection Problems

If you're having trouble connecting to the MCP server:

1. Verify Home Assistant is running
2. Check `HASS_URL` accessibility
3. Validate token permissions
4. Ensure WebSocket connection for real-time updates

#### High Memory Usage

If you're experiencing high memory usage:

1. Reduce the context size in configuration
2. Use a lighter LLM model
3. Disable features you don't use
4. Increase memory allocation if possible

#### Authorization Failures

If you see authorization errors:

1. Check that your Home Assistant long-lived token is correct
2. Verify the MCP server has proper permissions
3. Ensure your network allows the required connections

### Logs

View logs for troubleshooting:

```bash
# Docker logs
docker logs hass-mcp-plus

# Home Assistant logs
grep "mcp_server_plus" /config/home-assistant.log
```

## 📈 Performance Considerations

- **Hardware Requirements**: For optimal performance, we recommend:
  - 4GB+ RAM
  - Quad-core CPU or better
  - SSD storage for vector database

- **Network Impact**: The integration exchanges data with LLMs which can consume bandwidth:
  - Local LLMs: Minimal external bandwidth, higher local resource usage
  - Remote LLMs: Higher bandwidth usage, lower local resource impact

- **Storage Requirements**:
  - RAG index: 100MB - 1GB depending on your home data volume
  - Logs and history: 10-100MB depending on usage

## 📝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Home Assistant community](https://community.home-assistant.io/)
- [home-llm project](https://github.com/acon96/home-llm)
- [Home Assistant Datasets](https://github.com/allenporter/home-assistant-datasets)
- [homeassistant-mcp](https://github.com/tevonsb/homeassistant-mcp)
