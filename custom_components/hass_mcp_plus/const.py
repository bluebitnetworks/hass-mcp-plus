"""Constants for the Home Assistant Advanced MCP Integration."""

# Integration domain
DOMAIN = "hass_mcp_plus"

# Configuration constants
CONF_LLM_PROVIDER = "llm_provider"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_URL = "llm_url"
CONF_ENABLE_RAG = "enable_rag"
CONF_ENABLE_DASHBOARD_GEN = "enable_dashboard_gen"
CONF_ENABLE_AUTOMATION_GEN = "enable_automation_gen"
CONF_CHUNK_SIZE = "chunk_size"

# Default values
DEFAULT_PORT = 8787
DEFAULT_HOST = "0.0.0.0"
DEFAULT_SSL = False
DEFAULT_LLM_PROVIDER = "local"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_ENABLE_RAG = True
DEFAULT_ENABLE_DASHBOARD_GEN = True
DEFAULT_ENABLE_AUTOMATION_GEN = True
DEFAULT_CHUNK_SIZE = 512

# Supported LLM providers
SUPPORTED_LLM_PROVIDERS = [
    "local", 
    "ollama",
    "openai",
    "anthropic"
]

# Sensor attributes
ATTR_SERVER_STATUS = "server_status"
ATTR_SERVER_URL = "server_url"
ATTR_LLM_PROVIDER = "llm_provider"
ATTR_LLM_MODEL = "llm_model"
ATTR_FEATURES = "features"
ATTR_LAST_QUERY = "last_query"
ATTR_LAST_RESPONSE = "last_response"
ATTR_REQUEST_COUNT = "request_count"
ATTR_UPTIME = "uptime"

# Integration sensor name
SENSOR_NAME = "MCP Server"

# Icons
ICON_SERVER = "mdi:server"
ICON_MODEL = "mdi:brain"
ICON_DASHBOARD = "mdi:view-dashboard"
ICON_AUTOMATION = "mdi:robot"
ICON_RAG = "mdi:database-search"

# API endpoints
API_SERVER_INFO = "/info"
API_HEALTH = "/health"
API_QUERY = "/api/query"
API_DASHBOARD_GENERATE = "/api/dashboard/generate"
API_AUTOMATION_GENERATE = "/api/automation/generate"

# Error messages
ERROR_CONNECTION = "Connection error"
ERROR_AUTHENTICATION = "Authentication error"
ERROR_SERVER_START = "Failed to start MCP server"
ERROR_INVALID_PROVIDER = "Invalid LLM provider"
ERROR_MISSING_API_KEY = "Missing API key"
ERROR_MISSING_MODEL = "Missing model name"
ERROR_MISSING_URL = "Missing URL"
