#!/usr/bin/env python3
"""
Utility functions for the Advanced MCP Server.
"""

import os
import re
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Set up global logger
logger = logging.getLogger("mcp_server")

def setup_logging(level: str = None) -> None:
    """
    Set up global logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    
    numeric_level = getattr(logging, level, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Set level for our loggers
    logging.getLogger("mcp_server").setLevel(numeric_level)
    
    # Set level for external libraries
    logging.getLogger("uvicorn").setLevel(numeric_level)
    logging.getLogger("fastapi").setLevel(numeric_level)
    
    # Lower level for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    
    logger.info(f"Logging configured with level: {level}")

def generate_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())

def sanitize_entity_id(entity_id: str) -> str:
    """
    Sanitize an entity ID to ensure it matches Home Assistant requirements.
    
    Args:
        entity_id: Entity ID to sanitize
        
    Returns:
        Sanitized entity ID
    """
    # Keep only letters, numbers, and allowed special characters
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', entity_id.lower())
    
    # Ensure it's in the correct format (domain.entity)
    if '.' not in sanitized:
        sanitized = f"sensor.{sanitized}"
    
    return sanitized

def sanitize_string(text: str) -> str:
    """
    Sanitize a string for safe use.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove control characters and other potentially dangerous characters
    return re.sub(r'[\x00-\x1F\x7F<>:"\'`;&|]', '', text)

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Datetime object or None if parsing fails
    """
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date string: {date_str}")
    return None

def serialize_datetime(dt: datetime) -> str:
    """
    Serialize a datetime object to ISO format.
    
    Args:
        dt: Datetime object to serialize
        
    Returns:
        ISO formatted date string
    """
    return dt.isoformat()

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return {}

def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def json_serializer(obj: Any) -> Any:
    """
    JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized object
    """
    if isinstance(obj, datetime):
        return serialize_datetime(obj)
    
    # Add more serializers for custom types as needed
    
    raise TypeError(f"Type {type(obj)} not serializable")

def format_response(data: Any) -> Dict[str, Any]:
    """
    Format a response for the API.
    
    Args:
        data: Data to format
        
    Returns:
        Formatted response
    """
    return {
        "data": data,
        "timestamp": serialize_datetime(datetime.now()),
        "status": "success"
    }

def format_error(message: str, code: str = "UNKNOWN_ERROR", details: Any = None) -> Dict[str, Any]:
    """
    Format an error response for the API.
    
    Args:
        message: Error message
        code: Error code
        details: Error details
        
    Returns:
        Formatted error response
    """
    return {
        "error": {
            "message": message,
            "code": code,
            "details": details
        },
        "timestamp": serialize_datetime(datetime.now()),
        "status": "error"
    }

def parse_bool(value: Union[str, bool, int]) -> bool:
    """
    Parse a boolean value from various input types.
    
    Args:
        value: Value to parse
        
    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, int):
        return value != 0
    
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "t", "1", "on")
    
    return bool(value)

def get_entity_domain(entity_id: str) -> str:
    """
    Get the domain from an entity ID.
    
    Args:
        entity_id: Entity ID
        
    Returns:
        Entity domain
    """
    parts = entity_id.split(".", 1)
    if len(parts) == 2:
        return parts[0]
    return ""

def truncate_string(text: str, max_length: int = 100) -> str:
    """
    Truncate a string to a maximum length and add ellipsis if truncated.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text, useful for parsing LLM responses.
    
    Args:
        text: Text to parse
        
    Returns:
        Extracted JSON or None if not found
    """
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
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
    
    return None

def extract_service_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract service calls from text, useful for parsing LLM responses.
    
    Args:
        text: Text to parse
        
    Returns:
        List of extracted service calls
    """
    service_calls = []
    
    # Look for homeassistant code blocks
    matches = re.findall(r'```(?:homeassistant)?\s*(.*?)\s*```', text, re.DOTALL)
    
    for match in matches:
        try:
            # Parse the JSON
            data = json.loads(match)
            
            # Check if it's a valid service call
            if "service" in data:
                service_calls.append(data)
            elif isinstance(data, list):
                # Handle list of service calls
                for item in data:
                    if isinstance(item, dict) and "service" in item:
                        service_calls.append(item)
        except json.JSONDecodeError:
            continue
    
    return service_calls

def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to check
        
    Returns:
        True if valid, False otherwise
    """
    url_pattern = re.compile(
        r'^(?:http|https)://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))

def to_camel_case(snake_str: str) -> str:
    """
    Convert a snake_case string to camelCase.
    
    Args:
        snake_str: Snake case string
        
    Returns:
        Camel case string
    """
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake_case(camel_str: str) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Args:
        camel_str: Camel case string
        
    Returns:
        Snake case string
    """
    # Insert underscore before uppercase letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
