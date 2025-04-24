# Installation Guide

This guide will walk you through the installation and configuration of the Home Assistant Advanced MCP Integration. There are two primary installation methods: HACS (Home Assistant Community Store) and Docker.

## HACS Installation

The HACS installation method is best for users who want to integrate directly with their existing Home Assistant installation without setting up additional containers.

### Prerequisites

- Home Assistant version 2024.3.0 or newer
- HACS (Home Assistant Community Store) installed
- A long-lived access token from Home Assistant

### Installation Steps

1. Open HACS in your Home Assistant instance
2. Go to Integrations
3. Click the "+" button in the bottom right corner
4. Click "Custom repositories"
5. Add the repository URL: `https://github.com/yourusername/hass-mcp-plus`
6. Select "Integration" as the category
7. Click "Add"
8. Find and install "Home Assistant Advanced MCP Integration" from the list
9. Restart Home Assistant

### Configuration

After installation:

1. Go to Settings → Devices & Services
2. Click "+ Add Integration" in the bottom right corner
3. Search for "Home Assistant Advanced MCP Integration"
4. Follow the configuration wizard:
   - Enter your Home Assistant long-lived access token
   - Select your preferred LLM provider
   - Configure additional settings as needed

## Docker Installation (Recommended)

The Docker installation method is recommended for more advanced users who want the most features and performance.

### Prerequisites

- Docker and Docker Compose installed
- Home Assistant instance accessible to the Docker container
- A long-lived access token from Home Assistant

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hass-mcp-plus.git
   cd hass-mcp-plus
   ```

2. Create a `.env` file with the following content:
   ```
   HASS_URL=http://your-home-assistant-url:8123
   HASS_TOKEN=your_long_lived_access_token
   LLM_PROVIDER=ollama  # Options: ollama, openai, anthropic, local
   LLM_URL=http://ollama:11434  # Only for ollama or local
   LLM_MODEL=llama3  # Model name
   TZ=America/New_York  # Your timezone
   ```

3. Start the containers:
   ```bash
   docker-compose up -d
   ```

4. Add the integration to Home Assistant:
   - Go to Settings → Devices & Services
   - Click "+ Add Integration"
   - Search for "MCP Server"
   - Enter the URL: `http://hass-mcp-plus:8787` (or your container IP if not on the same network)

## Manual Setup with Local LLM

For users who want to use a local LLM without Docker:

1. Follow the HACS installation steps above
2. Download a supported GGUF model file (e.g., from Hugging Face)
3. Place the model file in a location accessible to Home Assistant
4. During configuration, select "local" as the LLM provider
5. Specify the path to your model file

## Advanced Configuration

For advanced configuration options, check the following files:

- `custom_components/hass_mcp_plus/const.py` - For changing default settings
- `server/config.py` - For additional server options
- Docker environment variables for each container

## Verify Installation

After installation, you should see a new "MCP Server" sensor in your Home Assistant instance that shows the server status. You can also access the web UI at:

- `http://your-server-ip:8080` (if using Docker)
- `http://your-home-assistant-ip:8787` (if using HACS)

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure your Home Assistant URL and token are correct
2. **Model Loading Error**: Check that your LLM model is correctly specified
3. **Memory Issues**: If running on a low-memory device, try a smaller model

For more detailed troubleshooting, check the logs:

```bash
# Docker logs
docker logs hass-mcp-plus

# Home Assistant logs
grep "mcp_server_plus" /config/home-assistant.log
```

## Updating

### HACS Update

1. Open HACS in your Home Assistant instance
2. Go to Integrations
3. Find "Home Assistant Advanced MCP Integration"
4. Click the update button

### Docker Update

```bash
cd hass-mcp-plus
git pull
docker-compose down
docker-compose up -d
```
