# graph_service.py
import httpx
from typing import List, Dict, Optional
from fastapi import HTTPException
from pydantic import AnyHttpUrl, ValidationError
from pydantic_settings import BaseSettings
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    BASE_URL: AnyHttpUrl = "http://localhost:3000"
    
    # Additional environment variables from .env
    google_api_key: Optional[str] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[str] = None
    user_agent: Optional[str] = None
    
    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'allow'
    }

class GraphService:
    def __init__(self):
        try:
            self.settings = Settings()
            self.base_url = str(self.settings.BASE_URL).rstrip('/')
            self.timeout = httpx.Timeout(30.0)
            logger.info(f"GraphService initialized with base URL: {self.base_url}")
            
        except ValidationError as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise ValueError(f"Invalid service configuration: {str(e)}") from e
    
    async def get_graph_context(self, node_ids: List[str], edge_ids: List[str]) -> Dict:
        """Get context from Next.js API"""
        try:
            # logger.info(f"Fetching graph context for {len(node_ids)} nodes and {len(edge_ids)} edges")
            
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.post(
                    f"{self.base_url}/api/graph/context",
                    json={
                        "node_ids": node_ids,
                        "edge_ids": edge_ids
                    }
                )
                response.raise_for_status()
                
                # logger.info("Successfully fetched graph context")
                return response.json()
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Graph service returned {e.response.status_code}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=502,
                detail=error_msg
            )
            
        except Exception as e:
            error_msg = f"Failed to get graph context: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
    
    async def reason(self, input: str, force_web: bool = False, external_context: Optional[Dict] = None) -> Dict:
        """Call reasoning endpoint with improved error handling"""
        try:
            # logger.info(f"Requesting reasoning for input: {input[:50]}...")
            
            payload = {
                "input": input,
                "force_web": force_web,
                "external_context": external_context or {}
            }
            
            logger.debug(f"Reasoning payload: {payload}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/reason",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                # logger.info(f"Reasoning result: strategy={result.get('strategy')}, confidence={result.get('confidence')}")

                # Validasi struktur response
                required_keys = ["strategy", "confidence", "context_sources", "reasoning_chain"]
                missing_keys = [key for key in required_keys if key not in result]
                
                if missing_keys:
                    logger.warning(f"Missing keys in reasoning response: {missing_keys}")
                    # Set default values untuk missing keys
                    defaults = {
                        "strategy": "hybrid",
                        "confidence": 0.8,
                        "context_sources": ["vector_db"],
                        "reasoning_chain": ["Fallback reasoning applied"]
                    }
                    for key in missing_keys:
                        result[key] = defaults[key]

                return {
                    "strategy": result.get("strategy", "hybrid"),
                    "confidence": result.get("confidence", 0.8),
                    "context_sources": result.get("context_sources", ["vector_db"]),
                    "reasoning_chain": result.get("reasoning_chain", ["Default reasoning chain"])
                }
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Reasoning service returned {e.response.status_code}: {str(e)}"
            logger.error(error_msg)
            
            # Return fallback response instead of raising exception
            return {
                "strategy": "rag_only",  # Fallback strategy
                "confidence": 0.6,
                "context_sources": ["vector_db"],
                "reasoning_chain": [f"Fallback due to API error: {error_msg}"]
            }
            
        except Exception as e:
            error_msg = f"Reasoning service error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return fallback response instead of raising exception
            return {
                "strategy": "rag_only",  # Fallback strategy
                "confidence": 0.5,
                "context_sources": ["vector_db"],
                "reasoning_chain": [f"Fallback due to service error: {error_msg}"]
            }

# Initialize service instance
try:
    graph_service = GraphService()
    logger.info("GraphService instance created successfully")
except Exception as e:
    logger.critical(f"Failed to initialize GraphService: {str(e)}")
    raise