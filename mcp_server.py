# mcp_server.py
import json
import asyncio
import sys
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
import os

# Import your existing system components
from main import system, init_system, ChatRequest, ProcessPDFRequest, SuggestionRequest, FollowupRequest, GenerateEdgesRequest

load_dotenv()
user_agent = os.getenv("USER_AGENT")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPError(Exception):
    """Base MCP error class"""
    def __init__(self, code: int, message: str, data: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"{code}: {message}")

class MCPErrorCode(Enum):
    """JSON-RPC error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000

@dataclass
class MCPRequest:
    """Standard MCP request structure"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

@dataclass
class MCPResponse:
    """Standard MCP response structure"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]

@dataclass
class MCPResource:
    """MCP resource definition"""
    uri: str
    name: str
    mimeType: str
    description: Optional[str] = None

@dataclass
class MCPPrompt:
    """MCP prompt definition"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None

class Transport(ABC):
    """Abstract transport layer"""
    
    @abstractmethod
    async def send(self, message: str) -> None:
        pass
    
    @abstractmethod
    async def receive(self) -> str:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass

class StdioTransport(Transport):
    """Standard input/output transport"""
    
    def __init__(self):
        self.running = True
    
    async def send(self, message: str) -> None:
        print(message, flush=True)
    
    async def receive(self) -> str:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                self.running = False
                return ""
            return line.strip()
        except EOFError:
            self.running = False
            return ""
    
    async def close(self) -> None:
        self.running = False

class HTTPTransport(Transport):
    """HTTP transport layer"""

    def __init__(self):
        self.app = FastAPI(title="MCP Server")
        self.server = None

        @self.app.post("/mcp")
        async def mcp_endpoint(request: Request):
            req_json = await request.json()
            response = await self.server._handle_request(req_json)
            return JSONResponse(content=asdict(response))
        
    async def send(self, message: str) -> None:
        pass

    async def receive(self) -> str:
        pass

    async def close(self) -> None:
        pass

class MCPServer:
    """Main MCP server implementation"""
    
    def __init__(self, transport: Transport):
        self.transport = transport
        self.methods: Dict[str, Callable] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.running = False
        
        # Initialize system components
        self.system = system
        
        # Register standard MCP methods
        self._register_methods()
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_methods(self):
        """Register standard MCP methods"""
        self.methods = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "server/info": self._handle_server_info,
            "server/health": self._handle_health_check,
        }
    
    def _register_tools(self):
        """Register PRAL cycle tools"""
        self.tools = {
            "pral_chat": MCPTool(
                name="pral_chat",
                description="Execute PRAL cycle for chat interactions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "User question"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "mode": {
                            "type": "string",
                            "enum": ["general", "single_node", "multi_nodes"],
                            "default": "general"
                        },
                        "node_id": {"type": "string", "description": "Single node ID"},
                        "node_ids": {"type": "array", "items": {"type": "string"}},
                        "force_web": {"type": "boolean", "default": False},
                        "context_node_ids": {"type": "array", "items": {"type": "string"}},
                        "context_edge_ids": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["question", "session_id"]
                }
            ),
            "process_pdf": MCPTool(
                name="process_pdf",
                description="Process PDF documents for RAG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_url": {"type": "string", "description": "PDF URL"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "node_id": {"type": "string", "description": "Associated node ID"},
                        "metadata": {"type": "object", "description": "Additional metadata"}
                    },
                    "required": ["pdf_url", "session_id"]
                }
            ),
            "get_suggestions": MCPTool(
                name="get_suggestions",
                description="Generate contextual suggestions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query"},
                        "context": {"type": "object", "description": "Context information"},
                        "suggestion_type": {
                            "type": "string",
                            "enum": ["input", "followup"],
                            "default": "input"
                        },
                        "chat_history": {"type": "array", "items": {"type": "object"}}
                    },
                    "required": ["query"]
                }
            ),
            "get_followup_suggestions": MCPTool(
                name="get_followup_suggestions",
                description="Generate follow-up suggestions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "lastMessage": {"type": "string", "description": "Last message"},
                        "conversationHistory": {"type": "array", "items": {"type": "object"}},
                        "context": {"type": "object", "description": "Context information"},
                        "suggestion_type": {"type": "string", "default": "followup"}
                    },
                    "required": ["lastMessage"]
                }
            ),
            "generate_edges_from_all_nodes": MCPTool(
                name="generate_edges_from_all_nodes",
                description="Generate edges between nodes based on their content within a session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "all_nodes_data": {
                            "type": "array",
                            "description": "List of node data dictionaries for edge generation.",
                            "items": {
                                "type" : "object",
                                "properties": {
                                     "id": {"type": "string"},
                                    "label": {"type": "string"},
                                    "title": {"type": ["string", "null"]},
                                    "att_goal": {"type": ["string", "null"]},
                                    "att_method": {"type": ["string", "null"]},
                                    "att_background": {"type": ["string", "null"]},
                                    "att_future": {"type": ["string", "null"]},
                                    "att_gaps": {"type": ["string", "null"]},
                                    "att_url": {"type": ["string", "null"]},
                                    "articleId": {"type": "string"}
                                },
                                "required": ["id", "label", "articleId"]
                            }
                        }
                    },
                    "required": ["all_nodes_data"]
                }
            )
        }
    
    def _register_resources(self):
        """Register available resources"""
        self.resources = {
            "chroma_metadata": MCPResource(
                uri="internal://chroma/metadata",
                name="ChromaDB Metadata",
                mimeType="application/json",
                description="ChromaDB collection metadata and statistics"
            ),
            "system_info": MCPResource(
                uri="internal://system/info",
                name="System Information",
                mimeType="application/json",
                description="System components and provider information"
            ),
            "learning_metrics": MCPResource(
                uri="internal://learning/metrics",
                name="Learning Metrics",
                mimeType="application/json",
                description="Learning system metrics and insights"
            )
        }
    
    def _register_prompts(self):
        """Register available prompts"""
        self.prompts = {
            "brainstorm": MCPPrompt(
                name="brainstorm",
                description="Generate brainstorming suggestions",
                arguments=[
                    {"name": "topic", "description": "Topic to brainstorm about", "required": True},
                    {"name": "context", "description": "Additional context", "required": False}
                ]
            ),
            "followup": MCPPrompt(
                name="followup",
                description="Generate follow-up questions",
                arguments=[
                    {"name": "previous_response", "description": "Previous AI response", "required": True},
                    {"name": "conversation_history", "description": "Conversation history", "required": False}
                ]
            )
        }
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False}
            },
            "serverInfo": {
                "name": "PRAL MCP Server",
                "version": "1.0.0",
                "description": "MCP server implementing PRAL cycle (Perceive, Reason, Act, Learn)"
            }
        }
    
    async def _handle_tools_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools list request"""
        return {
            "tools": [asdict(tool) for tool in self.tools.values()]
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise MCPError(
                MCPErrorCode.METHOD_NOT_FOUND.value,
                f"Tool '{tool_name}' not found"
            )
        
        try:
            if tool_name == "pral_chat":
                request = ChatRequest(**arguments)
                # Use your existing chat handler logic
                from main import handle_chat
                result = await handle_chat(request)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
            elif tool_name == "process_pdf":
                request = ProcessPDFRequest(**arguments)
                # Use your existing PDF processing logic
                from main import process_pdf_from_url
                result = await process_pdf_from_url(request)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
            elif tool_name == "get_suggestions":
                request = SuggestionRequest(**arguments)
                # Use your existing suggestion logic
                from main import get_suggestions
                result = await get_suggestions(request)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
            elif tool_name == "get_followup_suggestions":
                request = FollowupRequest(**arguments)
                # Use your existing followup logic
                from main import get_followup_suggestions
                result = await get_followup_suggestions(request)
                return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
            elif tool_name == "generate_edges_from_all_nodes":
                request = GenerateEdgesRequest(**arguments)
                from main import generate_edges_from_all_nodes
                generated_edges = await generate_edges_from_all_nodes(request.all_nodes_data)
                return {"edges": generated_edges}
            
            else:
                raise MCPError(
                    MCPErrorCode.METHOD_NOT_FOUND.value,
                    f"Tool '{tool_name}' handler not implemented"
                )
                
        except Exception as e:
            logger.error(f"Tool call error: {str(e)}")
            raise MCPError(
                MCPErrorCode.INTERNAL_ERROR.value,
                f"Tool execution failed: {str(e)}"
            )
    
    async def _handle_resources_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources list request"""
        return {
            "resources": [asdict(resource) for resource in self.resources.values()]
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource read request"""
        uri = params.get("uri")
        
        if not uri:
            raise MCPError(
                MCPErrorCode.INVALID_PARAMS.value,
                "URI parameter is required"
            )
        
        try:
            if uri == "internal://chroma/metadata":
                # Get ChromaDB metadata
                from main import debug_chroma_metadata
                result = await debug_chroma_metadata(limit=5)
                return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(result)}]}
            
            elif uri == "internal://system/info":
                # Get system information
                from main import system_info
                result = await system_info()
                return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(result)}]}
            
            elif uri == "internal://learning/metrics":
                # Get learning metrics
                metrics = self.system["learning"].get_learning_insights()
                return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(metrics)}]}
            
            else:
                raise MCPError(
                    MCPErrorCode.METHOD_NOT_FOUND.value,
                    f"Resource '{uri}' not found"
                )
                
        except Exception as e:
            logger.error(f"Resource read error: {str(e)}")
            raise MCPError(
                MCPErrorCode.INTERNAL_ERROR.value,
                f"Resource read failed: {str(e)}"
            )
    
    async def _handle_prompts_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts list request"""
        return {
            "prompts": [asdict(prompt) for prompt in self.prompts.values()]
        }
    
    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt get request"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name not in self.prompts:
            raise MCPError(
                MCPErrorCode.METHOD_NOT_FOUND.value,
                f"Prompt '{name}' not found"
            )
        
        try:
            if name == "brainstorm":
                topic = arguments.get("topic", "general research")
                context = arguments.get("context", "")
                prompt_text = f"""Generate brainstorming suggestions for the topic: {topic}
                
Context: {context}

Please provide 5 creative and thought-provoking suggestions that explore different aspects of this topic."""
                
                return {
                    "description": self.prompts[name].description,
                    "messages": [
                        {"role": "user", "content": {"type": "text", "text": prompt_text}}
                    ]
                }
            
            elif name == "followup":
                previous_response = arguments.get("previous_response", "")
                conversation_history = arguments.get("conversation_history", "")
                
                prompt_text = f"""Based on the previous AI response, generate follow-up questions:

Previous Response: {previous_response}

Conversation History: {conversation_history}

Please generate 3-5 follow-up questions that would help explore the topic deeper or clarify important points."""
                
                return {
                    "description": self.prompts[name].description,
                    "messages": [
                        {"role": "user", "content": {"type": "text", "text": prompt_text}}
                    ]
                }
            
            else:
                raise MCPError(
                    MCPErrorCode.METHOD_NOT_FOUND.value,
                    f"Prompt '{name}' handler not implemented"
                )
                
        except Exception as e:
            logger.error(f"Prompt get error: {str(e)}")
            raise MCPError(
                MCPErrorCode.INTERNAL_ERROR.value,
                f"Prompt generation failed: {str(e)}"
            )
    
    async def _handle_server_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle server info request"""
        return {
            "name": "PRAL MCP Server",
            "version": "1.0.0",
            "description": "MCP server implementing PRAL cycle (Perceive, Reason, Act, Learn)",
            "capabilities": {
                "tools": len(self.tools),
                "resources": len(self.resources),
                "prompts": len(self.prompts)
            },
            "status": "healthy",
            "uptime": datetime.now().isoformat()
        }
    
    async def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request"""
        try:
            # Check system components
            system_status = {
                "chroma_db": bool(self.system.get("chroma_db")),
                "mcp": bool(self.system.get("mcp")),
                "perception": bool(self.system.get("perception")),
                "reasoning": bool(self.system.get("reasoning")),
                "action": bool(self.system.get("action")),
                "learning": bool(self.system.get("learning"))
            }
            
            all_healthy = all(system_status.values())
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp": datetime.now().isoformat(),
                "components": system_status
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _create_error_response(self, error: MCPError, request_id: Optional[Union[str, int]] = None) -> MCPResponse:
        """Create standard error response"""
        return MCPResponse(
            error={
                "code": error.code,
                "message": error.message,
                "data": error.data
            },
            id=request_id
        )
    
    async def _handle_request(self, request_data: Dict[str, Any]) -> MCPResponse:
        """Handle individual request"""
        try:
            # Parse request
            request = MCPRequest(**request_data)
            
            # Find and execute method
            if request.method not in self.methods:
                raise MCPError(
                    MCPErrorCode.METHOD_NOT_FOUND.value,
                    f"Method '{request.method}' not found"
                )
            
            handler = self.methods[request.method]
            result = await handler(request.params or {})
            
            return MCPResponse(result=result, id=request.id)
            
        except MCPError as e:
            return self._create_error_response(e, request_data.get("id"))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self._create_error_response(
                MCPError(MCPErrorCode.INTERNAL_ERROR.value, f"Internal error: {str(e)}"),
                request_data.get("id")
            )
    
    async def _process_message(self, message: str) -> None:
        """Process incoming message"""
        try:
            # Parse JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                error_response = self._create_error_response(
                    MCPError(MCPErrorCode.PARSE_ERROR.value, f"Parse error: {str(e)}")
                )
                await self.transport.send(json.dumps(asdict(error_response)))
                return
            
            # Handle batch requests
            if isinstance(data, list):
                responses = []
                for item in data:
                    response = await self._handle_request(item)
                    responses.append(asdict(response))
                await self.transport.send(json.dumps(responses))
            else:
                # Handle single request
                response = await self._handle_request(data)
                await self.transport.send(json.dumps(asdict(response)))
                
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")
            error_response = self._create_error_response(
                MCPError(MCPErrorCode.INTERNAL_ERROR.value, f"Processing error: {str(e)}")
            )
            await self.transport.send(json.dumps(asdict(error_response)))
    
    async def start(self) -> None:
        """Start the MCP server"""
        self.running = True
        logger.info("MCP Server started")
        
        try:
            while self.running:
                message = await self.transport.receive()
                if not message:
                    break
                
                await self._process_message(message)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the MCP server"""
        self.running = False
        await self.transport.close()
        logger.info("MCP Server stopped")

def create_mcp_server(transport_type: str = "stdio") -> MCPServer:
    """Factory function to create MCP server with specified transport"""
    if transport_type == "stdio":
        transport = StdioTransport()
    elif transport_type == "http":
        transport = HTTPTransport()
    else:
        raise ValueError(f"Unsupported transport type: {transport_type}")
    
    server = MCPServer(transport)
    if transport_type == "http":
        transport.server = server
    return server

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="PRAL MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", 
                       help="Transport type (default: stdio)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if 'system' not in globals():
        system = init_system()

    server = create_mcp_server(args.transport)
    if args.transport == "http":
        # Panggil uvicorn.run() langsung, JANGAN di dalam fungsi async
        uvicorn.run(server.transport.app, host="0.0.0.0", port=8000)
    else:
        asyncio.run(server.start())