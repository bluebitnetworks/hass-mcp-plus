#!/usr/bin/env python3
"""
Security module for the Advanced MCP Server.

This module handles authentication, authorization, and security-related
functionality for the MCP server.
"""

import os
import time
import logging
import hashlib
import hmac
import base64
import secrets
import json
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger("mcp_server.security")

# Models for authentication data
class AuthData(BaseModel):
    """Authentication data."""
    user: Dict[str, Any]
    token: str
    token_type: str
    expires_at: Optional[datetime] = None

class UserData(BaseModel):
    """User data."""
    id: str
    username: str
    name: Optional[str] = None
    email: Optional[str] = None
    role: str
    permissions: List[str]

# Security configuration defaults
DEFAULT_TOKEN_EXPIRY = 86400  # 24 hours
DEFAULT_JWT_SECRET = secrets.token_hex(32)
DEFAULT_API_KEY_HEADER = "X-API-Key"
DEFAULT_HASH_ALGORITHM = "HS256"

# Security settings
jwt_secret = os.environ.get("JWT_SECRET", DEFAULT_JWT_SECRET)
token_expiry = int(os.environ.get("TOKEN_EXPIRY", DEFAULT_TOKEN_EXPIRY))
api_key_header = os.environ.get("API_KEY_HEADER", DEFAULT_API_KEY_HEADER)
hash_algorithm = os.environ.get("HASH_ALGORITHM", DEFAULT_HASH_ALGORITHM)

# Security bearer for JWT tokens
security = HTTPBearer(auto_error=False)

# In-memory token storage (replace with database in production)
tokens: Dict[str, AuthData] = {}
api_keys: Dict[str, UserData] = {}

def configure_security(config: Dict[str, Any]) -> None:
    """
    Configure security settings.
    
    Args:
        config: Security configuration
    """
    global jwt_secret, token_expiry, api_key_header, hash_algorithm
    
    jwt_secret = config.get("jwt_secret", jwt_secret)
    token_expiry = config.get("token_expiry", token_expiry)
    api_key_header = config.get("api_key_header", api_key_header)
    hash_algorithm = config.get("hash_algorithm", hash_algorithm)
    
    # Initialize API keys from config
    api_key = config.get("api_token")
    if api_key:
        add_api_key(api_key, UserData(
            id="admin",
            username="admin",
            name="Administrator",
            role="admin",
            permissions=["*"]
        ))
    
    logger.info("Security module configured")

def add_api_key(key: str, user: UserData) -> None:
    """
    Add an API key to the system.
    
    Args:
        key: API key
        user: User data
    """
    api_keys[key] = user
    logger.info(f"Added API key for user: {user.username}")

def remove_api_key(key: str) -> bool:
    """
    Remove an API key from the system.
    
    Args:
        key: API key
        
    Returns:
        True if the key was removed, False otherwise
    """
    if key in api_keys:
        del api_keys[key]
        logger.info(f"Removed API key")
        return True
    return False

def generate_token(user: UserData) -> str:
    """
    Generate a JWT token for a user.
    
    Args:
        user: User data
        
    Returns:
        JWT token
    """
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=token_expiry)
    
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions,
        "iat": now.timestamp(),
        "exp": expires_at.timestamp()
    }
    
    if user.name:
        payload["name"] = user.name
    
    if user.email:
        payload["email"] = user.email
    
    token = jwt.encode(payload, jwt_secret, algorithm=hash_algorithm)
    
    # Store token data
    auth_data = AuthData(
        user=user.dict(),
        token=token,
        token_type="bearer",
        expires_at=expires_at
    )
    tokens[token] = auth_data
    
    logger.info(f"Generated token for user: {user.username}")
    return token

def validate_token(token: str) -> Optional[AuthData]:
    """
    Validate a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        Authentication data if valid, None otherwise
    """
    try:
        # Check if token is in memory
        if token in tokens:
            auth_data = tokens[token]
            
            # Check if token is expired
            if auth_data.expires_at and auth_data.expires_at < datetime.utcnow():
                logger.warning("Token is expired")
                del tokens[token]
                return None
            
            return auth_data
        
        # Decode token
        payload = jwt.decode(token, jwt_secret, algorithms=[hash_algorithm])
        
        # Check expiration
        if "exp" in payload and payload["exp"] < datetime.utcnow().timestamp():
            logger.warning("Token is expired")
            return None
        
        # Create user data from payload
        user = UserData(
            id=payload["sub"],
            username=payload["username"],
            name=payload.get("name"),
            email=payload.get("email"),
            role=payload.get("role", "user"),
            permissions=payload.get("permissions", [])
        )
        
        # Create auth data
        auth_data = AuthData(
            user=user.dict(),
            token=token,
            token_type="bearer",
            expires_at=datetime.fromtimestamp(payload["exp"]) if "exp" in payload else None
        )
        
        # Store token
        tokens[token] = auth_data
        
        return auth_data
    
    except jwt.PyJWTError as e:
        logger.warning(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}", exc_info=True)
        return None

def invalidate_token(token: str) -> bool:
    """
    Invalidate a JWT token.
    
    Args:
        token: JWT token
        
    Returns:
        True if the token was invalidated, False otherwise
    """
    if token in tokens:
        del tokens[token]
        logger.info(f"Invalidated token")
        return True
    return False

def clean_expired_tokens() -> int:
    """
    Clean expired tokens from memory.
    
    Returns:
        Number of tokens removed
    """
    now = datetime.utcnow()
    expired_tokens = [
        token for token, auth_data in tokens.items()
        if auth_data.expires_at and auth_data.expires_at < now
    ]
    
    for token in expired_tokens:
        del tokens[token]
    
    if expired_tokens:
        logger.info(f"Cleaned {len(expired_tokens)} expired tokens")
    
    return len(expired_tokens)

def verify_api_key(key: str) -> Optional[UserData]:
    """
    Verify an API key.
    
    Args:
        key: API key
        
    Returns:
        User data if valid, None otherwise
    """
    return api_keys.get(key)

def get_api_key(request: Request) -> Optional[str]:
    """
    Get API key from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        API key if found, None otherwise
    """
    return request.headers.get(api_key_header)

async def authenticate(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthData:
    """
    Authenticate a request.
    
    This is a FastAPI dependency that can be used to authenticate requests.
    
    Args:
        request: FastAPI request
        credentials: HTTP authorization credentials
        
    Returns:
        Authentication data
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try API key first
    api_key = get_api_key(request)
    if api_key:
        user = verify_api_key(api_key)
        if user:
            # Create token for API key
            token = generate_token(user)
            auth_data = tokens[token]
            return auth_data
    
    # Try JWT token
    if credentials:
        token = credentials.credentials
        auth_data = validate_token(token)
        if auth_data:
            return auth_data
    
    # Try token from query parameter
    token = request.query_params.get("token")
    if token:
        auth_data = validate_token(token)
        if auth_data:
            return auth_data
    
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"}
    )

def get_current_user(auth_data: AuthData = Depends(authenticate)) -> UserData:
    """
    Get the current authenticated user.
    
    This is a FastAPI dependency that can be used to get the current user.
    
    Args:
        auth_data: Authentication data
        
    Returns:
        User data
    """
    return UserData(**auth_data.user)

def check_permission(permission: str) -> Callable:
    """
    Check if the current user has a specific permission.
    
    This is a FastAPI dependency factory that can be used to check permissions.
    
    Args:
        permission: Permission to check
        
    Returns:
        FastAPI dependency
    """
    def _check_permission(user: UserData = Depends(get_current_user)) -> UserData:
        if "*" in user.permissions or permission in user.permissions:
            return user
        raise HTTPException(
            status_code=403,
            detail=f"Permission denied: {permission}"
        )
    return _check_permission

def check_role(role: str) -> Callable:
    """
    Check if the current user has a specific role.
    
    This is a FastAPI dependency factory that can be used to check roles.
    
    Args:
        role: Role to check
        
    Returns:
        FastAPI dependency
    """
    def _check_role(user: UserData = Depends(get_current_user)) -> UserData:
        if user.role == role:
            return user
        raise HTTPException(
            status_code=403,
            detail=f"Role required: {role}"
        )
    return _check_role

def create_webhook_signature(payload: Union[Dict, str], secret: str) -> str:
    """
    Create a webhook signature for a payload.
    
    Args:
        payload: Payload to sign
        secret: Secret for signing
        
    Returns:
        Signature
    """
    if isinstance(payload, dict):
        payload = json.dumps(payload)
    
    # Create signature
    signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).digest()
    
    # Encode as base64
    return base64.b64encode(signature).decode()

def verify_webhook_signature(
    payload: Union[Dict, str],
    signature: str,
    secret: str
) -> bool:
    """
    Verify a webhook signature.
    
    Args:
        payload: Payload to verify
        signature: Signature to verify
        secret: Secret for verification
        
    Returns:
        True if the signature is valid, False otherwise
    """
    expected = create_webhook_signature(payload, secret)
    return hmac.compare_digest(signature, expected)

def generate_safe_id(prefix: str = "") -> str:
    """
    Generate a safe ID.
    
    Args:
        prefix: Prefix for the ID
        
    Returns:
        Safe ID
    """
    # Generate a UUID-like string that is safe for IDs
    random_id = secrets.token_hex(16)
    if prefix:
        return f"{prefix}_{random_id}"
    return random_id

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password.
    
    Args:
        password: Password to hash
        salt: Salt for hashing (optional)
        
    Returns:
        Tuple of (hash, salt)
    """
    if not salt:
        salt = secrets.token_hex(16)
    
    # Create hash
    hash_obj = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt.encode(),
        100000
    )
    
    # Encode as hex
    password_hash = hash_obj.hex()
    
    return password_hash, salt

def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """
    Verify a password.
    
    Args:
        password: Password to verify
        password_hash: Hash to verify against
        salt: Salt for hashing
        
    Returns:
        True if the password is valid, False otherwise
    """
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, password_hash)
