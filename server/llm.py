#!/usr/bin/env python3
"""
LLM Manager for the Advanced MCP Server.

This module handles interactions with various LLM providers 
including local models, Ollama, OpenAI, and Anthropic.
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union, Any

from schema import MCPRequest, MCPResponse

logger = logging.getLogger("mcp_server.llm")

class LLMManager:
    """
    Manager for handling interactions with Language Models.
    Supports multiple providers: local, ollama, openai, and anthropic.
    """
    
    def __init__(
        self,
        provider: str = "local",
        model: str = "llama3",
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize the LLM Manager.
        
        Args:
            provider: LLM provider (local, ollama, openai, anthropic)
            model: Model name
            api_key: API key for the provider
            url: URL for the model API
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up the API URL based on provider
        if url:
            self.url = url
        else:
            if self.provider == "local":
                self.url = "http://localhost:8000/v1"
            elif self.provider == "ollama":
                self.url = "http://localhost:11434/api"
            elif self.provider == "openai":
                self.url = "https://api.openai.com/v1"
            elif self.provider == "anthropic":
                self.url = "https://api.anthropic.com/v1"
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info(f"Initialized LLM Manager with provider: {self.provider}, model: {self.model}")
    
    async def generate_text(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "local":
                return await self._generate_local(prompt)
            elif self.provider == "ollama":
                return await self._generate_ollama(prompt)
            elif self.provider == "openai":
                return await self._generate_openai(prompt)
            elif self.provider == "anthropic":
                return await self._generate_anthropic(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            raise
    
    async def process_query(self, request: MCPRequest) -> MCPResponse:
        """
        Process an MCP query.
        
        Args:
            request: MCP request object
            
        Returns:
            MCP response object
        """
        # Convert the context to a string representation
        context_str = json.dumps(request.context.dict(), indent=2)
        
        # Create a system prompt that instructs the LLM how to handle the context
        system_prompt = f"""
        You are an AI assistant for a smart home system. You have access to the
        current state of the home through the provided context. Answer the user's
        query based on this context. If asked to perform actions, explain what 
        actions would be taken but do not actually perform them unless you're
        explicitly instructed to do so by the system.
        
        Guidelines:
        - Be concise and helpful
        - If you don't know, say so
        - If an action is requested, explain what would happen
        - Format your responses appropriately for the medium
        - Use markdown for formatting when helpful
        
        JSON format for responses with actions:
        {{
            "response": "Your text response to the user",
            "actions": [
                {{
                    "type": "service_call",
                    "domain": "light",
                    "service": "turn_on",
                    "entity_id": "light.living_room",
                    "data": {{
                        "brightness": 255,
                        "color_name": "red"
                    }}
                }}
            ]
        }}
        """
        
        # Combine system prompt, context, and user query
        full_prompt = f"""
        {system_prompt}
        
        CONTEXT:
        {context_str}
        
        USER QUERY:
        {request.query}
        
        Your response (include a JSON actions array if appropriate):
        """
        
        # Generate the response
        try:
            response_text = await self.generate_text(full_prompt)
            
            # Try to parse actions if they're included in JSON format
            actions = []
            
            try:
                # Try to find and parse JSON in the response
                json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group(1))
                    if "actions" in parsed_json:
                        actions = parsed_json["actions"]
                        # Replace the JSON block with just the text response
                        response_text = parsed_json.get("response", "")
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'{(?:[^{}]|{[^{}]*})*}', response_text)
                    if json_match:
                        try:
                            parsed_json = json.loads(json_match.group(0))
                            if "actions" in parsed_json:
                                actions = parsed_json["actions"]
                                # Remove the JSON from the response text
                                response_text = response_text.replace(json_match.group(0), "")
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Failed to parse actions from response: {str(e)}")
            
            # Clean up the response text
            response_text = response_text.strip()
            
            # Create and return the MCP response
            return MCPResponse(
                response=response_text,
                actions=actions,
                raw_llm_response=response_text,
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return MCPResponse(
                response=f"Sorry, I encountered an error processing your query: {str(e)}",
                actions=[],
                raw_llm_response="",
            )
    
    async def _generate_local(self, prompt: str) -> str:
        """
        Generate text using a local model API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stop": ["<|im_end|>"]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                async with session.post(f"{self.url}/completions", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from local LLM API: {response.status}, {error_text}")
                    
                    result = await response.json()
                    return result.get("choices", [{}])[0].get("text", "")
            except Exception as e:
                logger.error(f"Error calling local LLM API: {str(e)}", exc_info=True)
                raise
    
    async def _generate_ollama(self, prompt: str) -> str:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                async with session.post(f"{self.url}/generate", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Ollama API: {response.status}, {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")
            except Exception as e:
                logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
                raise
    
    async def _generate_openai(self, prompt: str) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("API key is required for OpenAI")
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            try:
                async with session.post(f"{self.url}/chat/completions", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from OpenAI API: {response.status}, {error_text}")
                    
                    result = await response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}", exc_info=True)
                raise
    
    async def _generate_anthropic(self, prompt: str) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.api_key:
            raise ValueError("API key is required for Anthropic")
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": self.temperature,
                "max_tokens_to_sample": self.max_tokens,
                "stop_sequences": ["\n\nHuman:"]
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            try:
                async with session.post(f"{self.url}/complete", json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from Anthropic API: {response.status}, {error_text}")
                    
                    result = await response.json()
                    return result.get("completion", "")
            except Exception as e:
                logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
                raise
