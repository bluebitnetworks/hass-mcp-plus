"""Config flow for Home Assistant Advanced MCP Integration."""

import logging
import voluptuous as vol
from typing import Any, Dict, Optional

from homeassistant import config_entries
from homeassistant.const import (
    CONF_HOST,
    CONF_PORT,
    CONF_SSL,
    CONF_TOKEN,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
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
    SUPPORTED_LLM_PROVIDERS,
    ERROR_CONNECTION,
    ERROR_AUTHENTICATION,
    ERROR_INVALID_PROVIDER,
    ERROR_MISSING_API_KEY,
    ERROR_MISSING_MODEL,
    ERROR_MISSING_URL,
)

_LOGGER = logging.getLogger(__name__)


async def validate_input(hass: HomeAssistant, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    errors = {}

    # Validate LLM provider
    llm_provider = data.get(CONF_LLM_PROVIDER)
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        errors[CONF_LLM_PROVIDER] = ERROR_INVALID_PROVIDER

    # Validate provider-specific requirements
    if llm_provider in ["openai", "anthropic"]:
        if not data.get(CONF_LLM_API_KEY):
            errors[CONF_LLM_API_KEY] = ERROR_MISSING_API_KEY

    if llm_provider in ["ollama", "local"] and not data.get(CONF_LLM_URL):
        errors[CONF_LLM_URL] = ERROR_MISSING_URL

    if not data.get(CONF_LLM_MODEL):
        errors[CONF_LLM_MODEL] = ERROR_MISSING_MODEL

    # Check if we can connect to Home Assistant with the token
    session = async_get_clientsession(hass)
    try:
        async with session.get(
            f"{hass.config.api.base_url}/api/",
            headers={"Authorization": f"Bearer {data[CONF_TOKEN]}"},
        ) as response:
            if response.status != 200:
                errors["base"] = ERROR_AUTHENTICATION
    except Exception:
        errors["base"] = ERROR_CONNECTION

    if errors:
        return {"errors": errors}

    return {"title": "Advanced MCP Server"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Assistant Advanced MCP Integration."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: Dict[str, str] = {}

        if user_input is not None:
            result = await validate_input(self.hass, user_input)
            if "errors" in result:
                errors = result["errors"]
            else:
                return self.async_create_entry(title=result["title"], data=user_input)

        # Provide default values
        schema = vol.Schema(
            {
                vol.Required(CONF_TOKEN): str,
                vol.Optional(CONF_HOST, default=DEFAULT_HOST): str,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): int,
                vol.Optional(CONF_SSL, default=DEFAULT_SSL): bool,
                vol.Optional(CONF_LLM_PROVIDER, default=DEFAULT_LLM_PROVIDER): vol.In(
                    SUPPORTED_LLM_PROVIDERS
                ),
                vol.Required(CONF_LLM_MODEL, default=DEFAULT_LLM_MODEL): str,
                vol.Optional(CONF_LLM_API_KEY): str,
                vol.Optional(CONF_LLM_URL): str,
                vol.Optional(CONF_ENABLE_RAG, default=DEFAULT_ENABLE_RAG): bool,
                vol.Optional(
                    CONF_ENABLE_DASHBOARD_GEN, default=DEFAULT_ENABLE_DASHBOARD_GEN
                ): bool,
                vol.Optional(
                    CONF_ENABLE_AUTOMATION_GEN, default=DEFAULT_ENABLE_AUTOMATION_GEN
                ): bool,
                vol.Optional(CONF_CHUNK_SIZE, default=DEFAULT_CHUNK_SIZE): int,
            }
        )

        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle a option flow for Advanced MCP Integration."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        self.options = dict(config_entry.options)

    async def async_step_init(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle options flow."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        current = self.config_entry.data
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_LLM_PROVIDER, 
                    default=current.get(CONF_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
                ): vol.In(SUPPORTED_LLM_PROVIDERS),
                vol.Optional(
                    CONF_LLM_MODEL, 
                    default=current.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL)
                ): str,
                vol.Optional(
                    CONF_LLM_API_KEY, 
                    default=current.get(CONF_LLM_API_KEY, "")
                ): str,
                vol.Optional(
                    CONF_LLM_URL, 
                    default=current.get(CONF_LLM_URL, "")
                ): str,
                vol.Optional(
                    CONF_ENABLE_RAG, 
                    default=current.get(CONF_ENABLE_RAG, DEFAULT_ENABLE_RAG)
                ): bool,
                vol.Optional(
                    CONF_ENABLE_DASHBOARD_GEN,
                    default=current.get(CONF_ENABLE_DASHBOARD_GEN, DEFAULT_ENABLE_DASHBOARD_GEN)
                ): bool,
                vol.Optional(
                    CONF_ENABLE_AUTOMATION_GEN,
                    default=current.get(CONF_ENABLE_AUTOMATION_GEN, DEFAULT_ENABLE_AUTOMATION_GEN)
                ): bool,
                vol.Optional(
                    CONF_CHUNK_SIZE, 
                    default=current.get(CONF_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)
                ): int,
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )
