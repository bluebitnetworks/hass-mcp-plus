#!/usr/bin/env python3
"""
Retrieval Augmented Generation (RAG) system for the Advanced MCP Server.

This module implements a RAG system that enhances the context for LLM queries
with relevant information retrieved from a vector database.
"""

import os
import json
import logging
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

from schema import ContextData
from hass_client import HomeAssistantClient

logger = logging.getLogger("mcp_server.rag")

class RAGSystem:
    """
    Retrieval Augmented Generation system for enhancing context.
    """
    
    def __init__(
        self,
        vector_store: str = "chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        index_path: str = "/data/rag_index",
        hass_client: Optional[HomeAssistantClient] = None,
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Vector store type (currently only supports "chroma")
            embedding_model: Embedding model to use
            chunk_size: Chunk size for text splitting
            index_path: Path to the vector index
            hass_client: Home Assistant client
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.index_path = index_path
        self.hass_client = hass_client
        
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.initialized = False
        self.last_update_time = None
        
        logger.info(f"Initialized RAG system with vector store: {vector_store}, embedding model: {embedding_model}")
    
    async def initialize(self):
        """Initialize the RAG system."""
        if self.initialized:
            return
        
        try:
            # Create the index directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            # Set up the embedding function
            if self.embedding_model == "all-MiniLM-L6-v2":
                # Try to use the local embedding model service first
                try:
                    self.embedding_function = HuggingFaceEmbeddingFunction(
                        api_url="http://embedding-model:8001/embed"
                    )
                    logger.info("Using local embedding model service")
                except:
                    # Fall back to the HuggingFace embedding function
                    self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                        api_key=os.environ.get("HF_API_KEY"),
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    logger.info("Using HuggingFace embedding service")
            else:
                # Use the specified model with HuggingFace
                self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                    api_key=os.environ.get("HF_API_KEY"),
                    model_name=f"sentence-transformers/{self.embedding_model}"
                )
            
            # Set up the vector store
            if self.vector_store == "chroma":
                self.client = chromadb.PersistentClient(
                    path=self.index_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # Create or get the collection
                collection_name = "hass_mcp_rag"
                
                try:
                    self.collection = self.client.get_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Retrieved existing collection '{collection_name}' with {self.collection.count()} documents")
                except:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Created new collection '{collection_name}'")
                
                # Initialize the index if it's empty
                if self.collection.count() == 0:
                    await self.update_index()
                
                self.initialized = True
                logger.info("RAG system initialized successfully")
            else:
                raise ValueError(f"Unsupported vector store: {self.vector_store}")
        
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}", exc_info=True)
            raise
    
    async def close(self):
        """Close the RAG system."""
        if self.client:
            # Nothing to do for chroma client
            pass
        
        logger.info("RAG system closed")
    
    async def update_index(self):
        """Update the vector index with the latest data from Home Assistant."""
        if not self.hass_client:
            logger.warning("Home Assistant client not provided, cannot update index")
            return
        
        try:
            logger.info("Updating RAG index...")
            
            # Get data from Home Assistant
            states = await self.hass_client.get_states()
            configs = await self.hass_client.get_config()
            services = await self.hass_client.get_services()
            history = await self.hass_client.get_history(
                timestamp=datetime.now() - timedelta(days=7),
                filter_entities=None
            )
            
            # Convert to text documents
            documents = []
            
            # States document
            states_text = "Current States:\n"
            for state in states:
                states_text += f"Entity: {state.entity_id}, State: {state.state}\n"
                for attr_key, attr_value in state.attributes.items():
                    states_text += f"  {attr_key}: {attr_value}\n"
            documents.append(("states", "Current States", states_text))
            
            # Config document
            config_text = "Home Assistant Configuration:\n"
            config_text += json.dumps(configs, indent=2)
            documents.append(("config", "Home Assistant Configuration", config_text))
            
            # Services document
            services_text = "Available Services:\n"
            for domain, domain_services in services.items():
                services_text += f"Domain: {domain}\n"
                for service_name, service_data in domain_services.items():
                    services_text += f"  Service: {service_name}\n"
                    if "description" in service_data:
                        services_text += f"    Description: {service_data['description']}\n"
                    if "fields" in service_data:
                        services_text += f"    Fields:\n"
                        for field_name, field_data in service_data["fields"].items():
                            services_text += f"      {field_name}: {field_data.get('description', '')}\n"
            documents.append(("services", "Available Services", services_text))
            
            # History document
            history_text = "Recent History:\n"
            for entity_history in history:
                if entity_history:
                    entity_id = entity_history[0].entity_id
                    history_text += f"Entity: {entity_id}\n"
                    for state in entity_history:
                        history_text += f"  {state.last_updated}: {state.state}\n"
            documents.append(("history", "Recent History", history_text))
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=50,
                length_function=len,
            )
            
            # Clear the collection
            self.collection.delete(where={})
            
            # Add documents to the collection
            for doc_id, doc_title, doc_text in documents:
                chunks = text_splitter.split_text(doc_text)
                for i, chunk in enumerate(chunks):
                    self.collection.add(
                        documents=[chunk],
                        metadatas=[{"source": doc_id, "title": doc_title, "chunk": i}],
                        ids=[f"{doc_id}_{i}"]
                    )
            
            self.last_update_time = datetime.now()
            logger.info(f"RAG index updated successfully with {self.collection.count()} chunks")
        
        except Exception as e:
            logger.error(f"Error updating RAG index: {str(e)}", exc_info=True)
    
    async def enhance_context(self, query: str, context: ContextData) -> ContextData:
        """
        Enhance the context with relevant information retrieved from the vector database.
        
        Args:
            query: User query
            context: Current context
            
        Returns:
            Enhanced context
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Query the vector store
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Extract the relevant information
            retrieved_info = []
            for i in range(len(results["documents"][0])):
                document = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                retrieved_info.append({
                    "content": document,
                    "source": metadata["source"],
                    "title": metadata["title"],
                })
            
            # Add the retrieved information to the context
            enhanced_context = context.copy()
            enhanced_context.rag_results = retrieved_info
            
            return enhanced_context
        
        except Exception as e:
            logger.error(f"Error enhancing context: {str(e)}", exc_info=True)
            return context  # Return the original context if there's an error


class HuggingFaceEmbeddingFunction:
    """
    Custom embedding function for the local Hugging Face model service.
    """
    
    def __init__(self, api_url: str):
        """
        Initialize the embedding function.
        
        Args:
            api_url: URL of the embedding service
        """
        self.api_url = api_url
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Use aiohttp in a synchronous way since chromadb expects a sync function
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._async_embed(texts))
    
    async def _async_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts asynchronously.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "texts": texts
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Error from embedding service: {response.status}, {error_text}")
                    
                    result = await response.json()
                    return result.get("embeddings", [])
            except Exception as e:
                logger.error(f"Error calling embedding service: {str(e)}", exc_info=True)
                raise
