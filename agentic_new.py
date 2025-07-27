#agentic_new.py
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

# Document Processing
import logging 
from fastapi.logger import logger as fastapi_logger
from langchain_core.messages import HumanMessage

# --- MCP Protocol Implementation ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Integrate with FastAPI's logger
fastapi_logger.handlers = logger.handlers
fastapi_logger.setLevel(logger.level)

class Agent:
    """Simplified Agent implementation"""
    def __init__(self, name: str, model: Any, tools: List[Any], 
                 instructions: str, markdown: bool = False):
        self.name = name
        self.model = model
        self.tools = tools
        self.instructions = instructions
    
    def run(self, prompt: str) -> Any:
        # Implementasi sederhana
        return self.model.invoke(prompt)

class Gemini:
    """Wrapper for Google Gemini"""
    def __init__(self, id: str):
        self.model = ChatGoogleGenerativeAI(model=id)
    
    def invoke(self, prompt: str) -> Any:
        return self.model.invoke(prompt)

class DuckDuckGoTools:
    """Wrapper for DuckDuckGo Search"""
    def __init__(self):
        self.search = DuckDuckGoSearchRun()
    
    def run(self, query: str) -> str:
        return self.search.run(query)
    
# class ChromaDb:
#     def __init__(self, collection: str, path: str = "./chroma_data", embedder=None, persistent_client: bool = True):
#         self.embedder - embedder or GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@dataclass
class MCPMessage:
    """MCP Protocol Message Structure"""
    id: str
    method: str
    params: Dict[str, Any]
    timestamp: str = datetime.now().isoformat()

@dataclass
class MCPResource:
    """MCP Resource Definition"""
    uri: str
    name: str
    description: str
    mimeType: str
    metadata: Dict[str, Any]

class MCPProvider(ABC):
    """Abstract MCP Provider Base Class"""
    
    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        pass
    
    @abstractmethod
    async def read_resource(self, uri: str) -> str:
        pass
    
    @abstractmethod
    async def get_context(self, query: str) -> Dict[str, Any]:
        pass

class VectorDBMCPProvider(MCPProvider):
    """MCP Provider for Vector Database"""
    
    def __init__(self, vector_db, collection_name: str):
        # self.chroma_db = chroma_db
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.provider_id = "vector_db"
        self._last_error = None

    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        result = await self.vector_db.health_check(self.collection_name)
        print(f"[VectorDB] health_check() for '{self.collection_name}' returned: {result}")
        return result
    
    async def list_resources(self) -> List[MCPResource]:
        """List all documents in vector database"""
        try:
            # collection = self.chroma_db.client.get_collection(name=self.collection_name)
            # results = collection.get()
            docs = self.vector_db.similarity_search(query="", k=10)
            resources = []
            
            # for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            #     resource = MCPResource(
            #         uri=f"vectordb://{self.collection_name}/{doc_id}",
            #         name=metadata.get('file_name', f"Document_{i}"),
            #         description=f"Vector document from {metadata.get('source_type', 'unknown')}",
            #         mimeType="text/plain",
            #         metadata=metadata
            #     )
            #     resources.append(resource)

            for i, doc in enumerate(docs):
                meta = doc.get("metadata", {})
                resource = MCPResource(
                    uri=f"vectordb://{self.collection_name}/{doc.get('id', f'doc_{i}')}",
                    name=meta.get('file_name', f"Document_{i}"),
                    description=f"Vector document from {meta.get('source_type', 'unknown')}",
                    mimeType="text/plain",
                    metadata=meta
                )
                resources.append(resource)
            
            return resources
        except Exception as e:
            logger.error(f"list_resources error: {e}")
            return []
    
    async def read_resource(self, uri: str) -> str:
        """Read specific document from vector database"""
        doc_id = uri.split('/')[-1]
        try:
            # collection = self.chroma_db.client.get_collection(name=self.collection_name)
            # results = collection.get(ids=[doc_id])
            # return results['documents'][0] if results['documents'] else ""
            docs = self.vector_db.similarity_search(query="", k=1, filter_metadata={"id": doc_id})
            if docs:
                return docs[0].get("content", "")
            return ""
        except Exception as e:
            return ""
    
    async def get_context(self, query: str, external_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get context from Supabase VectorDB with proper structure, applying
        'article_id' filters directly via database similarity search.
        """
        base_response = {
            "provider": self.provider_id,
            "query": query,
            "documents": [],
            "metadatas": [],
            "distances": [],
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[VectorDBMCPProvider] Incoming query: '{query}'")
        logger.info(f"[VectorDBMCPProvider] Incoming external_filters: {external_filters}")

        try:
            # 1. Validate collection exists
            # Menggunakan async health_check dari SupabaseVectorWrapper
            if not await self.vector_db.health_check(self.collection_name): 
                logger.error(f"Collection {self.collection_name} not available via health check.")
                raise ValueError(f"Collection {self.collection_name} not available")

            # --- START MODIFICATION ---
            # Siapkan filter untuk diteruskan langsung ke SupabaseVectorDb.
            # Kita hanya tertarik pada 'article_id' untuk VectorDB.
            filter_for_similarity_search = {}
            if external_filters and "article_id" in external_filters:
                # Pastikan ini mengambil format {"in": [...]} yang sudah dibuat di main.py
                filter_for_similarity_search["article_id"] = external_filters["article_id"]
                logger.info(f"[VectorDBMCPProvider] Applying filter_for_similarity_search: {filter_for_similarity_search}")
            # --- END MODIFICATION ---

            # Jalankan similarity_search dengan filter langsung di level database
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.vector_db.similarity_search(
                    query=query,
                    k=10, # k dokumen teratas yang relevan SETELAH DIFILTER oleh database
                    filter_metadata=filter_for_similarity_search # <--- INI BAGIAN KUNCI!
                )
            )

            # --- START MODIFICATION ---
            # Hapus semua logika pemfilteran manual di Python
            # Karena filter sudah diterapkan di level database melalui RPC.
            # Ini berarti 'results' yang dikembalikan sudah seharusnya dokumen yang relevan.
            # filtered_documents = []
            # filtered_metadatas = []
            # filtered_distances = []
            # target_node_ids = set() 
            # if external_filters and "node_ids" in external_filters and external_filters["node_ids"]:
            #     target_node_ids.update(external_filters["node_ids"])

            # for r in results:
            #     if isinstance(r, dict):
            #         doc_content = r.get('content', '')
            #         doc_metadata = r.get('metadata', {})
            #         doc_similarity = r.get('similarity', None)
            #         doc_node_id = doc_metadata.get('node_id') 
            #         if not target_node_ids or (doc_node_id and str(doc_node_id) in target_node_ids):
            #             filtered_documents.append(doc_content)
            #             filtered_metadatas.append(doc_metadata)
            #             filtered_distances.append(doc_similarity)
            #     else:
            #         logger.warning(f"Unexpected result format from Supabase: {type(r)} - {r}")
            # --- END MODIFICATION ---
            
            # Ekstrak data dari hasil yang sudah difilter oleh database
            # Asumsi: `results` adalah list of dictionaries seperti yang dikembalikan fungsi RPC Anda
            documents = [r.get('content', '') for r in results]
            metadatas = [r.get('metadata', {}) for r in results]
            distances = [r.get('similarity', None) for r in results] # Menggunakan 'similarity' sebagai 'distance'
            
            # Jika ada kemungkinan 'id' dibutuhkan di metadata, tambahkan juga
            for i, meta in enumerate(metadatas):
                meta['id'] = results[i].get('id')

            logger.info(f"VectorDB search completed: documents={len(documents)}, metadatas={len(metadatas)}, distances={len(distances)}")
            
            return {
                **base_response,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
                "success": True
            }

        except Exception as e:
            logger.error(f"VectorDB error in get_context: {str(e)}", exc_info=True)
            return {
                **base_response,
                "error": str(e),
                "last_error": str(e),
                "success": False
            }
        
class GraphDBMCPProvider(MCPProvider):
    """MCP Provider for Graph Database"""
    
    def __init__(self, graph_service):
        self.graph_service = graph_service
        self.provider_id = "graphdb"
    
    async def get_context(self, query: str, external_filters: Optional[Dict] = None) -> Dict[str, Any]:
        node_ids = external_filters.get("node_ids", []) if external_filters else []
        edge_ids = external_filters.get("edge_ids", []) if external_filters else []
        
        try:
            graph_data = await self.graph_service.get_graph_context(node_ids, edge_ids)
            print(f"DEBUG: GraphDBMCPProvider.get_context received graph_data: {json.dumps(graph_data, indent=2)}")
            return {
                "provider": self.provider_id,
                "nodes": graph_data.get("nodes", []),
                "edges": graph_data.get("edges", []),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def list_resources(self) -> List[MCPResource]:
        return []  # GraphDB tidak menyediakan daftar resource statis

    async def read_resource(self, uri: str) -> str:
        return ""  # Tidak applicable untuk GraphDB
    
class GraphReasoningProvider(MCPProvider):
    """MCP Provider khusus untuk Graph Reasoning"""
    
    def __init__(self, graph_service):
        self.graph_service = graph_service
        self.provider_id = "graph_reasoning"
    
    async def get_context(self, query: str, external_filters: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            result = await self.graph_service.reason(
                input=query,
                force_web=external_filters.get("force_web", False),
                external_context={
                    "node_ids": external_filters.get("node_ids", []),
                    "edge_ids": external_filters.get("edge_ids", [])
                }
            )
            return {
                "provider": self.provider_id,
                "strategy": result["strategy"],
                "confidence": result["confidence"],
                "context_sources": result["context_sources"],
                "reasoning_chain": result["reasoning_chain"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "provider": self.provider_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def list_resources(self) -> List[MCPResource]:
        return []
    
    async def read_resource(self, uri: str) -> str:
        return ""

class WebSearchMCPProvider(MCPProvider):
    """MCP Provider for Web Search"""
    
    def __init__(self):
        self.provider_id = "web_search"
        try:
            self.search_tool = DuckDuckGoSearchRun()
            self.search_agent = Agent(
                name="Web Search MCP Agent",
                model=Gemini(id="gemini-2.0-flash"),
                tools=[self.search_tool],
                instructions="Search the web and provide structured results.",
                markdown=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize WebSearchMCPProvider: {e}", exc_info=True) # Tambah exc_info untuk detail traceback
            self.search_tool = None
            self.search_agent = None
    
    async def list_resources(self) -> List[MCPResource]:
        """List available web search capabilities"""
        # ... (tidak ada perubahan di sini)
        return [
            MCPResource(
                uri="websearch://duckduckgo",
                name="DuckDuckGo Search",
                description="Real-time web search using DuckDuckGo",
                mimeType="application/json",
                metadata={"provider": "duckduckgo", "realtime": True}
            )
        ]
    
    async def read_resource(self, uri: str) -> str:
        """Not applicable for web search"""
        return "Web search resources are query-dependent"
    
    async def get_context(self, query: str, external_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get web search context with enhanced error handling"""
        base_response = {
            "provider": self.provider_id,
            "query": query,
            "results": "",
            "timestamp": datetime.now().isoformat()
        }
        
        # --- PERBAIKAN: Validasi query sebelum menjalankan pencarian ---
        if not query or not query.strip():
            logger.warning(f"Web search skipped for provider '{self.provider_id}': Empty or whitespace-only query received.")
            return {
                **base_response,
                "results": "No valid query provided for web search",
                "error": "empty_query",
                "success": False
            }
        # --- AKHIR PERBAIKAN ---

        try:
            if not self.search_tool:
                raise ValueError("Search tool not initialized")
                    
            # logger.info(f"Executing web search for: '{query}'") # Tambah kutip untuk melihat query kosong
            
            # Execute search with timeout
            search_results = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.search_tool.run(query)
                ),
                timeout=15.0
            )
            
            # Validate search results
            if not search_results:
                logger.warning("Empty search results from DuckDuckGo.")
                search_results = "No search results found"
            elif not isinstance(search_results, str):
                # logger.warning(f"Unexpected search result type: {type(search_results)}. Converting to string.")
                search_results = str(search_results)
            
            # Process with agent if available
            if self.search_agent:
                try:
                    processed_results = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.search_agent.run(f"Summarize and structure this search result: {search_results[:1000]}")
                        ),
                        timeout=10.0
                    )
                    
                    if processed_results and hasattr(processed_results, 'content'):
                        search_results = processed_results.content
                    elif processed_results:
                        search_results = str(processed_results)
                        
                except asyncio.TimeoutError:
                    logger.warning("Agent processing timeout, using raw results.")
                except Exception as e:
                    logger.warning(f"Agent processing failed: {e}, using raw results.", exc_info=True) # Tambah exc_info
            
            # logger.info(f"Web search completed for query '{query}': {len(search_results)} characters.") # Tambah kutip
            
            return {
                **base_response,
                "results": search_results,
                "success": True
            }
                
        except asyncio.TimeoutError:
            logger.error("Web search timeout.")
            return {
                **base_response,
                "results": "Search timeout - please try again",
                "error": "timeout",
                "success": False
            }
        except Exception as e:
            logger.error(f"Web search error: {str(e)}", exc_info=True) # Pastikan exc_info=True
            return {
                **base_response,
                "results": f"Search error: {str(e)}",
                "error": str(e),
                "success": False
            }

class MCPOrchestrator:
    """MCP Protocol Orchestrator"""
    
    def __init__(self):
        self.providers: Dict[str, MCPProvider] = {}
        self.message_history: List[MCPMessage] = []
        self._context_cache={}
        self.cache_ttl = 300
    
    def register_provider(self, provider: MCPProvider):
        """Register MCP Provider"""
        self.providers[provider.provider_id] = provider
    
    async def list_all_resources(self) -> Dict[str, List[MCPResource]]:
        """List resources from all providers"""
        all_resources = {}
        for provider_id, provider in self.providers.items():
            all_resources[provider_id] = await provider.list_resources()
        return all_resources
    
    async def get_unified_context(self, query: str, providers: List[str] = None, external_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get unified context from multiple providers"""
        
        # logger.info(f"[MCP][UNIFIED_CTX] Processing query: {query}")
        # logger.info(f"[MCP][UNIFIED_CTX] Providers: {providers}")
        # logger.info(f"[MCP][UNIFIED_CTX] External filters: {external_filters}")
        
        if providers is None:
            providers = list(self.providers.keys())
        
        unified_context = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "providers": {}
        }
        
        for provider_id in providers:
            if provider_id in self.providers:
                try:
                    # logger.info(f"Getting context from provider: {provider_id}")
                    context = await self.providers[provider_id].get_context(query, external_filters)
                    
                    # print(f"🔧 Provider {provider_id} returned: {context.keys()}")
                    
                    # Debug: Log hasil dari provider
                    if provider_id == "vector_db":
                        docs_count = len(context.get('documents', []))
                        metas_count = len(context.get('metadatas', []))
                        # logger.info(f"Provider {provider_id}: documents={docs_count}, metadatas={metas_count}")
                        
                        # Consistency check
                        if docs_count != metas_count:
                            logger.warning(f"INCONSISTENCY: documents={docs_count} != metadatas={metas_count}")
                    
                    unified_context["providers"][provider_id] = context
                    # logger.info(f"Successfully got context from {provider_id}")
                    
                except Exception as e:
                    logger.error(f"Error getting context from provider {provider_id}: {e}")
                    unified_context["providers"][provider_id] = {
                        "provider": provider_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Final debug output
        # print(f"🔥 Final unified context structure:")
        # print(f"- Query: {unified_context['query']}")
        # print(f"- Providers: {list(unified_context['providers'].keys())}")
        
        for provider_id, provider_data in unified_context['providers'].items():
            if 'error' not in provider_data:
                docs_count = len(provider_data.get('documents', []))
                metas_count = len(provider_data.get('metadatas', []))
                print(f"- {provider_id}: docs={docs_count}, metas={metas_count}")
        
        return unified_context

# --- Enhanced Agentic AI Implementation ---

@dataclass
class PerceptionData:
    """Structured perception data"""
    user_input: str
    context_type: str
    confidence_score: float
    extracted_entities: List[str]
    intent: str
    timestamp: str = datetime.now().isoformat()

@dataclass
class ReasoningResult:
    """Structured reasoning result"""
    strategy: str
    confidence: float
    context_sources: List[str]
    reasoning_chain: List[str]
    timestamp: str = datetime.now().isoformat()

@dataclass
class ActionResult:
    """Structured action result"""
    action_type: str
    response: str
    sources_used: List[str]
    success: bool
    timestamp: str = datetime.now().isoformat()
    references: List[Dict[str, str]] = field(default_factory=list)
    usage_metadata: Optional[Dict[str, Any]] = field(default=None)

@dataclass
class LearningEvent:
    """Learning event data"""
    interaction_id: str
    user_feedback: Optional[str]
    performance_metrics: Dict[str, float]
    improvement_areas: List[str]
    timestamp: str = datetime.now().isoformat()

class EnhancedPerceptionModule:
    """Enhanced Perception with Multi-modal Input Analysis"""
    
    def __init__(self):
        self.entity_extractor = Agent(
            name="Entity Extractor",
            model=Gemini(id="gemini-2.0-flash"),
            instructions="Extract entities and analyze user intent from text.",
            markdown=False,
            tools=[],
        )
    
    async def perceive(self, user_input: str, context: Dict[str, Any]) -> PerceptionData:
        """Enhanced perception with entity extraction and intent analysis"""
        try:
            # Extract entities and intent
            analysis_prompt = f"""
            Analyze this user input and extract:
            1. Key entities (people, places, concepts)
            2. User intent (question, request, command)
            3. Context type (factual, creative, analytical)
            4. Confidence score (0-1)
            
            Input: {user_input}
            
            Return JSON format:
            {{
                "entities": ["entity1", "entity2"],
                "intent": "intent_type",
                "context_type": "type",
                "confidence_score": 0.0-1.0
            }}
            """
            
            result = self.entity_extractor.run(analysis_prompt)
            
            # Parse result (simplified - in production use proper JSON parsing)
            perception_data = PerceptionData(
                user_input=user_input,
                context_type="analytical",  # Default fallback
                confidence_score=0.8,      # Default fallback
                extracted_entities=["general"],  # Default fallback
                intent="question"           # Default fallback
            )
            
            return perception_data
            
        except Exception as e:
            return PerceptionData(
                user_input=user_input,
                context_type="general",
                confidence_score=0.5,
                extracted_entities=[],
                intent="unknown"
            )

class EnhancedReasoningModule:
    """Enhanced Reasoning with Strategy Selection"""
    
    def __init__(self, mcp_orchestrator: MCPOrchestrator):
        self.mcp_orchestrator = mcp_orchestrator
        self.reasoning_strategies = {
            "rag_only": {"providers": ["vector_db"], "weight": 0.8},
            "web_only": {"providers": ["web_search"], "weight": 0.6},
            "hybrid": {"providers": ["vector_db", "web_search"], "weight": 0.9},
            "technical_article_analysis": {"providers": ["vector_db", "graphdb"], "weight": 0.9},
            "theoretical_article_analysis": {"providers": ["vector_db", "graphdb"], "weight": 0.85},
            "goal_oriented_analysis": {"providers": ["vector_db", "graphdb"], "weight": 0.8},
            "web_enhanced": {"providers": ["vector_db", "web_search"], "weight": 0.9}
        }
    
    async def reason(self, perception_data: PerceptionData, force_web: bool = False, external_context: Optional[Dict] = None) -> ReasoningResult:
        try:
            # Gunakan MCP untuk reasoning
            reasoning_context = await self.mcp_orchestrator.get_unified_context(
                query=perception_data.user_input,
                providers=["graph_reasoning"],  # Khusus provider reasoning
                external_filters={
                    "force_web": force_web,
                    "node_ids": external_context.get("node_ids", []) if external_context else [],
                    "edge_ids": external_context.get("edge_ids", []) if external_context else []
                }
            )

            # Jika request tentang hubungan tapi node kurang dari 2
            if (external_context and len(external_context.get("node_ids", [])) < 2 
                and any(kw in perception_data.user_input.lower() 
                    for kw in ["hubungan", "bandingkan", "kaitannya"])):
                return ReasoningResult(
                    strategy="direct_response",
                    confidence=0.9,
                    context_sources=[],
                    reasoning_chain=["Fallback: only 1 node available for comparison"]
                )

            result = reasoning_context["providers"]["graph_reasoning"]
            return ReasoningResult(
                strategy=result["strategy"],
                confidence=result["confidence"],
                context_sources=result["context_sources"],
                reasoning_chain=result["reasoning_chain"]
            )
        

        except Exception as e:
            logger.error(f"Reasoning error: {str(e)}")
            return ReasoningResult(
                strategy="rag_only",
                confidence=0.5,
                context_sources=["vector_db"],
                reasoning_chain=[f"Fallback due to error: {str(e)}"]
            )
        

class EnhancedActionModule:
    def __init__(self):
        self.response_agent = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    def _build_context_text(self, unified_context: Dict[str, Any]) -> str:
        """Build context text with comprehensive error handling"""
        try:
            # 1. Validate root structure
            if not isinstance(unified_context, dict):
                logger.error(f"Invalid unified_context type: {type(unified_context)}")
                return "No context available"

            # logger.debug(f"Unified context structure: {unified_context.keys()}")
            
            # 2. Safely extract providers
            providers = unified_context.get("providers", {})
            if not isinstance(providers, dict):
                logger.error(f"Invalid providers type: {type(providers)}")
                providers = {}

            context_parts = []
            
            # 3. Process VectorDB Context (SUDAH BENAR) tapi saya comment karena mau coba yang lain
            # if 'vector_db' in providers:
            #     vector_context = providers['vector_db']
            #     if isinstance(vector_context, dict):
            #         documents = vector_context.get('documents', [])
            #         if isinstance(documents, list):
            #             logger.debug(f"Found {len(documents)} vector documents")
            #             context_parts.append("=== Knowledge Base ===")
            #             for i, doc in enumerate(documents[:13]):
            #                 if isinstance(doc, str) and doc.strip():
            #                     context_parts.append(f"Doc {i+1}: {doc[:300]}...")
            #         else:
            #             logger.warning("VectorDB documents is not a list")
            #     else:
            #         logger.warning("VectorDB context is not a dictionary")

            if 'vector_db' in providers:
                vector_context = providers['vector_db']
                if isinstance(vector_context, dict):
                    documents = vector_context.get('documents', [])
                    metadatas = vector_context.get('metadatas', [])
                    
                    if isinstance(documents, list) and documents:
                        logger.debug(f"Found {len(documents)} vector documents")
                        context_parts.append("=== Knowledge Base ===")
                        
                        # Group by source untuk menghindari "Doc X"
                        source_groups = {}
                        
                        for i, doc in enumerate(documents[:13]):
                            if isinstance(doc, str) and doc.strip():
                                # Coba ambil metadata untuk source info
                                source_title = "Dokumen"
                                if i < len(metadatas) and isinstance(metadatas[i], dict):
                                    source_title = metadatas[i].get('title', metadatas[i].get('source_url', f'Dokumen {i+1}'))
                                
                                # Group berdasarkan source
                                if source_title not in source_groups:
                                    source_groups[source_title] = []
                                source_groups[source_title].append(doc[:300])
                        
                        # Tambahkan ke context tanpa "Doc X"
                        for source_title, docs in source_groups.items():
                            context_parts.append(f"\n**Dari {source_title}:**")
                            for doc_content in docs:
                                context_parts.append(doc_content + "...")
                            context_parts.append("")  # Separator
                    else:
                        logger.warning("VectorDB documents is not a list or empty")
                else:
                    logger.warning("VectorDB context is not a dictionary")

            # 4. Process Web Search Context
            # if 'web_search' in providers:
            #     web_context = providers['web_search']
            #     if isinstance(web_context, dict):
            #         results = web_context.get('results')
            #         if results is not None and results == "": # Explicit None check
            #             logger.debug("Processing web results")
            #             context_parts.append("\n=== Web Results ===")
            #             try:
            #                 if isinstance(results, str):
            #                     context_parts.append(str(results)[:500])
            #                 else:
            #                     context_parts.append(str(results)[:500])
            #             except Exception as e:
            #                 logger.error(f"Web results formatting failed: {e}")
            #     else:
            #         logger.warning("Web context is not a dictionary")

            if 'web_search' in providers:
                web_context = providers['web_search']
                if isinstance(web_context, dict):
                    results = web_context.get('results') # Asumsi 'results' adalah string atau list
                    
                    # --- PERBAIKAN DI SINI ---
                    if results: # Cek jika ada data (tidak None atau string kosong)
                        context_parts.append("\n=== Web Search Results ===")
                        if isinstance(results, str):
                            # Jika hasil web adalah string teks besar (misal, dari scraping langsung)
                            context_parts.append(results[:1000]) # Batasi agar tidak terlalu panjang
                            if len(results) > 1000:
                                context_parts.append("... (further web results truncated)")
                        elif isinstance(results, list):
                            # Jika hasil web adalah list of dictionaries (lebih umum dari search API)
                            for i, res in enumerate(results[:5]): # Ambil 5 hasil teratas
                                if isinstance(res, dict):
                                    title = res.get('title', 'No Title')
                                    snippet = res.get('snippet', 'No snippet available')
                                    url = res.get('link', res.get('url', 'No URL')) # Coba 'link' atau 'url'
                                    context_parts.append(f"Result {i+1}: {title}\nSnippet: {snippet[:200]}...\nURL: {url}\n")
                        else:
                            logger.warning(f"Web results data is not string or list: {type(results)}")
                    else:
                        logger.debug("Web search results are empty or invalid.")
                else:
                    logger.warning("Web context is not a dictionary")
                # Tambahkan baris kosong agar lebih rapi antar bagian
                context_parts.append("")

            # 5. Process Graph Context
            # graph_data = unified_context.get('graph', {})
            # if isinstance(graph_data, dict):
            #     nodes = graph_data.get('nodes', []) or []  # Double safety
            #     edges = graph_data.get('edges', []) or []
            #     if nodes or edges:
            #         context_parts.append("\n=== Graph Context ===")
            #         context_parts.append(f"Nodes: {len(nodes)}")
            #         context_parts.append(f"Edges: {len(edges)}")

            # if 'graphdb' in providers:
            #     graph_context = providers['graphdb']
            #     if isinstance(graph_context, dict) and 'nodes' in graph_context:
            #         context_parts.append("\n=== Graph Knowledge ===")
            #         for node in graph_context['nodes'][:3]:  # Ambil 3 node teratas
            #             context_parts.append(
            #                 f"Node {node.get('id')}: {node.get('title', 'No title')} | "
            #                 f"Type: {node.get('type', 'unknown')}"
            #             )
            #         nodes = graph_context.get('nodes', [])
            #         if len(nodes) < 2:
            #             return "Hanya tersedia 1 artikel dalam konteks saat ini"

            # return "\n".join(context_parts) if context_parts else "No context available"

            # 5. Process Graph Context
            if 'graphdb' in providers:
                graph_context = providers['graphdb']
                if isinstance(graph_context, dict) and 'nodes' in graph_context:
                    nodes = graph_context.get('nodes', [])
                    edges = graph_context.get('edges', []) # Jangan lupakan edges!

                    if nodes or edges: # Tambahkan bagian ini untuk memastikan ada data graf
                        context_parts.append("\n=== Graph Knowledge ===")

                        if nodes:
                            context_parts.append("\n--- Relevant Graph Nodes (Articles) ---")
                            for i, node in enumerate(nodes[:5]): # Batasi jumlah node, misal 5
                                node_id = node.get('id')
                                node_title = node.get('title', 'No title') # PDF file name
                                node_content = node.get('content', '') # Main summary content
                                attributes = node.get('attributes', {})

                                context_parts.append(f"Node {i+1} ID: {node_id}")
                                context_parts.append(f"  Title: {node_title}")
                                context_parts.append(f"  Type: {node.get('type', 'unknown')}") # Still relevant for context

                                if node_content:
                                    context_parts.append(f"  Summary Excerpt: {node_content[:250]}...")

                                # Add all specific attributes
                                if attributes.get('goal'): context_parts.append(f"  Goal: {attributes['goal']}")
                                if attributes.get('method'): context_parts.append(f"  Method: {attributes['method']}")
                                if attributes.get('background'): context_parts.append(f"  Background: {attributes['background']}")
                                if attributes.get('future'): context_parts.append(f"  Future Work: {attributes['future']}")
                                if attributes.get('gaps'): context_parts.append(f"  Gaps: {attributes['gaps']}")
                                if attributes.get('url'): context_parts.append(f"  URL: {attributes['url']}")
                                context_parts.append("---") # Separator for nodes

                        if edges:
                            context_parts.append("\n--- Relevant Graph Edges (Connections) ---")
                            for i, edge in enumerate(edges[:5]): # Batasi jumlah edge, misal 5
                                from_label = edge.get('nodes_info', {}).get('from', {}).get('label', 'N/A')
                                to_label = edge.get('nodes_info', {}).get('to', {}).get('label', 'N/A')
                                context_parts.append(
                                    f"Edge {i+1} ID: {edge.get('id')}: "
                                    f"'{from_label}' {edge.get('relation', 'connected to')} '{to_label}' "
                                    f"(Label: {edge.get('label', 'No label')})"
                                )
                                context_parts.append("---") # Separator for edges

            return "\n".join(context_parts).strip() if context_parts else "No context available"    


        except Exception as e:
            logger.error(f"Critical error in _build_context_text: {e}", exc_info=True)
            return "Error generating context"
            
    # def _extract_sources(self, unified_context: Dict[str, Any]) -> List[Dict[str, str]]:
    #     """Extract sources from unified context with proper error handling"""
    #     sources = []
        
    #     # Debug: Verifikasi struktur
    #     # print(f"🔍 Keys in unified_context: {unified_context.keys()}")
    #     # print(f"🔍 Providers keys: {unified_context.get('providers', {}).keys()}")
        
    #     # Ambil data dari vector_db provider
    #     vector_data = unified_context.get('providers', {}).get('vector_db', {})
    #     # print(f"🔍 VectorDB data keys: {vector_data.keys()}")
        
    #     # Handle error case
    #     if 'error' in vector_data:
    #         print(f"❌ VectorDB error: {vector_data['error']}")
    #         return sources
        
    #     # Get metadatas and documents
    #     metadatas = vector_data.get('metadatas', [])
    #     documents = vector_data.get('documents', [])
        
    #     # print(f"📌 Metadata count: {len(metadatas)}")
    #     # print(f"📌 Documents count: {len(documents)}")

    #     #tambahan untuk tidak extract (saat tidak relevan)
    #     if not documents or all(not doc.strip() for doc in documents):
    #         print("X no valid documents found, skipping source extraction")
    #         return sources
    #     if not metadatas or all(not isinstance(meta, dict) or not meta.get('source_url') for meta in metadatas):
    #         print("X no valid metadata with source_url found, skipping source extraction")
    #         return sources

    #     # Tambahkan GraphDB sources
    #     if 'graphdb' in unified_context.get('providers', {}):
    #         graph_data = unified_context['providers']['graphdb']
    #         for node in graph_data.get('nodes', []):
    #             if 'id' in node:
    #                 sources.append({
    #                     'url': f"graph://node/{node['id']}",
    #                     'text': node.get('title', 'Graph Node'),
    #                     'type': 'graph_node',
    #                     'ref_mark': f"[G-{node['id'][:4]}]"
    #                 })
        
    #     # Consistency check
    #     if len(metadatas) != len(documents):
    #         # print(f"⚠️  WARNING: Metadata count ({len(metadatas)}) != Documents count ({len(documents)})")
    #         # Take minimum to avoid index errors
    #         min_count = min(len(metadatas), len(documents))
    #         metadatas = metadatas[:min_count]
    #         documents = documents[:min_count]

    #     seen_urls = {}

    #     #tambahan
    #     valid_sources_count = 0
        
    #     # Extract sources
    #     for idx, (meta, doc) in enumerate(zip(metadatas, documents), start=1):
    #         if isinstance(meta, dict) and meta.get('source_url'):
    #             url = meta['source_url']

    #             # if url in seen_urls:
    #             #     ref_mark = seen_urls[url]
    #             # else:
    #             #     ref_mark = f"[{len(seen_urls) + 1}]"
    #             #     seen_urls[url] = ref_mark

    #             #tambahan juga
    #             if not doc or len(doc.strip()) < 10:
    #                 print(f"⏭️ Skipping document {idx} - content too short or empty")
    #                 continue

    #             if url not in seen_urls:
    #                 ref_mark = f"[{len(seen_urls) + 1}]"
    #                 seen_urls[url] = ref_mark

    #             sources.append({
    #                 'url': url,
    #                 'text': meta.get('title', f"Artikel {idx}"),
    #                 'preview': (doc[:100] + '...') if doc else '',
    #                 'type': 'document',
    #                 'ref_mark': ref_mark
    #             })
    #         else:
    #             print(f"⚠️  Invalid metadata at index {idx}: {meta}")
        
    #     #tambahn juga
    #     if not sources:
    #         print("X No valid sources extracted, returning empty list")
    #         return []
    #     # print(f"✅ Total references extracted: {len(sources)}")
    #     return sources

    #sudah benar tapi tidak ada graph
    def _extract_sources(self, unified_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract sources with smarter relevance detection"""
        sources = []
        
        query = unified_context.get('query', '').lower()
        vector_data = unified_context.get('providers', {}).get('vector_db', {})
        
        if 'error' in vector_data:
            print(f"❌ VectorDB error: {vector_data['error']}")
            return sources
        
        metadatas = vector_data.get('metadatas', [])
        documents = vector_data.get('documents', [])
        distances = vector_data.get('distances', [])

        if 'graphdb' in unified_context.get('providers', {}):
            graph_data = unified_context['providers']['graphdb']
            for node in graph_data.get('nodes', []):
                if 'id' in node:
                    sources.append({
                        'url': f"graph://node/{node['id']}",
                        'text': node.get('title', 'Graph Node'),
                        'type': 'graph_node',
                        'ref_mark': f"[G-{node['id'][:4]}]"
                    })
            
        # Basic validation
        if not documents or all(not doc.strip() for doc in documents):
            print("❌ No valid documents found, skipping source extraction")
            return sources
        
        # ENHANCED GENERAL QUERY DETECTION
        general_patterns = [
            'apa itu', 'what is', 'definisi', 'pengertian', 'arti dari',
            'jelaskan tentang', 'explain about', 'mengapa', 'kenapa'
        ]
        
        is_general_query = any(pattern in query for pattern in general_patterns)
        
        if is_general_query and query:
            print(f"🔍 General query detected: '{query}'")
            
            # Extract topic dari query
            query_topic = ""
            for pattern in general_patterns:
                if pattern in query:
                    query_topic = query.replace(pattern, "").strip()
                    # Remove common words
                    query_topic = query_topic.replace("pertanyaan umum:", "").strip()
                    break
            
            print(f"📝 Extracted topic: '{query_topic}'")
            
            # KOMBINASI: Similarity Score + Content Analysis
            if distances:
                max_similarity = max(distances)
                print(f"📈 Max similarity: {max_similarity:.3f}")
                
                # Raised threshold untuk general queries
                similarity_threshold = 0.75  # Lebih strict untuk general query
                
                if max_similarity < similarity_threshold:
                    print(f"❌ Low similarity for general query: {max_similarity:.3f} < {similarity_threshold}")
                    return []
            
            # CONTENT RELEVANCE CHECK
            if query_topic:
                topic_keywords = [word.strip().lower() for word in query_topic.split() if len(word.strip()) > 2]
                print(f"🔍 Topic keywords: {topic_keywords}")
                
                relevant_docs = 0
                total_checked = min(len(documents), 5)  # Check top 5 docs
                
                for i, doc in enumerate(documents[:total_checked]):
                    if isinstance(doc, str) and doc.strip():
                        doc_lower = doc.lower()
                        
                        # Count exact keyword matches
                        keyword_matches = sum(1 for keyword in topic_keywords if keyword in doc_lower)
                        match_ratio = keyword_matches / len(topic_keywords) if topic_keywords else 0
                        
                        print(f"📄 Doc {i+1}: {keyword_matches}/{len(topic_keywords)} keywords found ({match_ratio:.1%})")
                        
                        # Minimal 50% keywords harus ada di dokumen
                        if match_ratio >= 0.5:
                            relevant_docs += 1
                
                relevance_ratio = relevant_docs / total_checked if total_checked > 0 else 0
                content_threshold = 0.4  # 40% dokumen harus relevan
                
                print(f"📊 Content relevance: {relevant_docs}/{total_checked} docs ({relevance_ratio:.1%})")
                
                if relevance_ratio < content_threshold:
                    print(f"❌ Low content relevance: {relevance_ratio:.1%} < {content_threshold:.0%}")
                    return []  # TIDAK ada sources
                else:
                    print(f"✅ Sufficient content relevance: {relevance_ratio:.1%} >= {content_threshold:.0%}")
        
        # Rest of extraction logic...
        if not metadatas or all(not isinstance(meta, dict) or not meta.get('source_url') for meta in metadatas):
            print("❌ No valid metadata with source_url found, skipping source extraction")
            return sources
        
        seen_urls = {}
        valid_sources_count = 0
        
        for idx, (meta, doc) in enumerate(zip(metadatas, documents), start=1):
            if isinstance(meta, dict) and meta.get('source_url'):
                url = meta['source_url']
                
                if not doc or len(doc.strip()) < 10:
                    continue
                
                if url not in seen_urls:
                    ref_mark = f"[{len(seen_urls) + 1}]"
                    seen_urls[url] = ref_mark
                    valid_sources_count += 1
                    
                    sources.append({
                        'url': url,
                        'text': meta.get('title', f"Artikel {valid_sources_count}"),
                        'preview': (doc[:100] + '...') if doc else '',
                        'type': 'document',
                        'ref_mark': ref_mark
                    })
        
        if not sources:
            print("❌ No valid sources extracted, returning empty list")
            return []
        
        print(f"✅ Total references extracted: {len(sources)}")
        return sources



    async def act(self, perception_data: PerceptionData, reasoning_result: ReasoningResult, unified_context: Dict[str, Any]) -> ActionResult:
        try:
            # logger.info(f"Action requested - strategy: {reasoning_result.strategy}")
            # logger.info(f"Context sources: {reasoning_result.context_sources}")

            # Validate unified_context structure
            if not isinstance(unified_context, dict):
                logger.error("Invalid unified_context - not a dictionary")
                unified_context = {"providers": {}}
    
            # Ensure providers exists and is a dictionary
            if 'providers' not in unified_context or not isinstance(unified_context['providers'], dict):
                unified_context['providers'] = {}

            if not unified_context or not isinstance(unified_context, dict):
                unified_context = {"providers": {}}

            graph_data = unified_context.get('providers', {}).get('graphdb', {})

            available_nodes = len(graph_data.get('nodes', []))

            # if available_nodes < 2 and "hubungan" in perception_data.user_input.lower():
            # if available_nodes < 2 and any(keyword in perception_data.user_input.lower() 
            #                   for keyword in ["hubungan", "bandingkan", "perbandingan", "compare"]):

            #     return ActionResult(
            #         action_type=reasoning_result.strategy,
            #         response="Maaf saat ini hanya tersedia 1 artikel sehingga tidak dapat membandingkan hubungan antar artikel.",
            #         sources_used=reasoning_result.context_sources,
            #         success=True,
            #         references=self._extract_sources(unified_context)
            #     )
                

            context_text = self._build_context_text(unified_context)

            print("\n--- Context Text Sent to LLM ---")
            print(context_text)
            print("--------------------------------\n")

            # Pastikan context_text valid
            if not context_text:
                context_text = "No context available"
            
            prompt = f"""**IMPORTANT**: The response MUST be in the same language as the user's question.
            
            **Task**: Answer the question thoroughly and provide detailed explanations based on the provided context. If the context contains distinct points or sections, elaborate on each of them.
            
            **Context**:
            {context_text}
            
            **Question**:
            {perception_data.user_input}
            
            **Instructions**:
            - Respond in the same language as the question. NO EXCEPTIONS.
            - Provide a comprehensive and detailed answer.
            - Explain each relevant point or aspect from context thoroughly.
            - Be precise and factual
            - Cite sources when applicable
            
            **REMEMBER**: Your answer must be in the language of the question.
            """
            
            response = self.response_agent.invoke(prompt)

            print("\n--- Response From LLM ---")
            print(response)
            print("--------------------------------\n")

            # Pastikan response valid
            if not response or not hasattr(response, 'content'):
                return ActionResult(
                    action_type="error",
                    response="Failed to generate response",
                    sources_used=[],
                    success=False
            )

            # Extract usage metadata
            usage_metadata = None
            if hasattr(response, 'usage_metadata'):
                usage_metadata = {
                    'input_tokens': response.usage_metadata.get('input_tokens', 0),
                    'output_tokens': response.usage_metadata.get('output_tokens', 0),
                    'total_tokens': response.usage_metadata.get('total_tokens', 0),
                    'model_name': getattr(response, 'response_metadata', {}).get('model_name', 'unknown')
            }

            # Get sources/references
            sources = self._extract_sources(unified_context)
            
            # Format response with references
            # formatted_response = response.content
            # if sources:
            #     ref_marks = []
            #     for i, source in enumerate(sources, 1):
            #         if source['url'] and source['url'] != 'web_search':
            #             ref_marks.append(f"[{i}]")
            #             formatted_response += f"\n\n[{i}] {source['url']}"

            formatted_response = response.content
            
            # if sources:
            #     for ref in sources:
            #         if ref.get("ref_mark") not in formatted_response:
            #             formatted_response += f" {ref['ref_mark']}"

            if sources:
                existing_marks = set(re.findall(r"\[[^\[\]]+\]", formatted_response))
                added_ref_marks = set()

                for ref in sources:
                    ref_mark = ref.get("ref_mark")
                    if (
                        ref_mark 
                        and ref_mark not in existing_marks 
                        and ref_mark not in added_ref_marks
                    ):
                        formatted_response += f" {ref_mark}"
                        added_ref_marks.add(ref_mark)

            return ActionResult(
                action_type=reasoning_result.strategy,
                # response=response.content,
                response=formatted_response,
                sources_used=reasoning_result.context_sources,
                success=True,
                references=sources,
                usage_metadata = usage_metadata
            )
        except Exception as e:
            return ActionResult(
                action_type="error",
                response=f"Error generating response: {str(e)}",
                sources_used=[],
                success=False
            )
        
    async def generate_suggestions(self, query: str) -> List[str]:
        try:
            print(f"🔍 Processing query: '{query}'")
            
            # Gunakan prompt yang lebih efektif
            messages = [HumanMessage(content=f"""
                Kamu adalah asisten brainstorming akademik. Berikan 5 ide eksploratif atau pertanyaan berdasarkan topik: '{query}'
                - Format numerik (1. ... 2. ...)
                - Setiap rekomendasi maksimal 5 kata
                - Fokus pada penyempurnaan query, bukan jawaban
                - Contoh format output:
                1. ...
                2. ...
                3. ...
                4. ...
                5. ...
            """)]
            
            result = await self.response_agent.ainvoke(messages)
            
            # Ekstrak suggestions dari response
            suggestions = [
                line.strip() 
                for line in result.content.split('\n') 
                if line.strip() and line[0].isdigit()
            ][:5]
            
            # Fallback jika format tidak sesuai
            if not suggestions or len(suggestions[0]) < 5:
                suggestions = [
                    f"1. Topik lanjutan tentang {query}",
                    f"2. Pertanyaan kritis seputar {query}",
                    f"3. aplikasi {query} di dunia nyata",
                    f"4. perbandingan {query} dengan metode lain",
                    f"5. Gap riset dalam {query}"
                ]
            
            print(f"✨ Generated: {suggestions}")
            return suggestions
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return [
                f"1. penjelasan {query}",
                f"2. definisi {query}",
                f"3. penerapan {query}",
                f"4. tutorial {query}",
                f"5. penelitian tentang {query}"
            ]
    
    # Di dalam class ActionSystem
    async def generate_followup_suggestions(self, query: str) -> List[str]:
        try:
            print(f"🔍 Processing followup query: '{query[:100]}...'")
            
            # Prompt khusus untuk followup questions
            messages = [HumanMessage(content=f"""
                Berdasarkan informasi berikut, buat 5 pertanyaan lanjutan yang relevan:
                
                {query}
                
                Buat pertanyaan yang:
                - Menggali lebih dalam topik yang dibahas
                - Praktis dan actionable
                - Mengeksplorasi aspek berbeda
                - Mudah dipahami
                
                Format output:
                1. [pertanyaan 1]
                2. [pertanyaan 2]
                3. [pertanyaan 3]
                4. [pertanyaan 4]
                5. [pertanyaan 5]
                
                Setiap pertanyaan maksimal 10 kata.
            """)]
            
            result = await self.response_agent.ainvoke(messages)
            
            # Ekstrak suggestions dari response
            suggestions = [
                line.strip() 
                for line in result.content.split('\n') 
                if line.strip() and line[0].isdigit()
            ][:5]
            
            # Fallback jika format tidak sesuai
            if not suggestions or len(suggestions) < 3:
                # Ekstrak topik utama dari query untuk fallback yang lebih kontekstual
                key_topics = self._extract_key_topics(query)
                suggestions = [
                    f"1. Bagaimana cara mengimplementasikan {key_topics}?",
                    f"2. Apa tantangan utama dalam {key_topics}?",
                    f"3. Bisakah berikan contoh kasus {key_topics}?",
                    f"4. Bagaimana perkembangan terbaru {key_topics}?",
                    f"5. Apa alternatif lain untuk {key_topics}?"
                ]
            
            print(f"✨ Generated followup: {suggestions}")
            return suggestions
            
        except Exception as e:
            print(f"❌ Followup Error: {str(e)}")
            return [
                "1. Bisakah dijelaskan lebih detail?",
                "2. Bagaimana cara praktis menerapkannya?",
                "3. Apa contoh kasus penggunaannya?",
                "4. Bagaimana tren perkembangannya?",
                "5. Apa kelebihan dan kekurangannya?"
            ]

    def _extract_key_topics(self, text: str) -> str:
        """Extract key topics from the text for better fallback suggestions"""
        # Simple keyword extraction (bisa diperbaiki dengan NLP yang lebih canggih)
        words = text.lower().split()
        # Filter common words and get potential topics
        stop_words = {'adalah', 'dapat', 'akan', 'yang', 'dengan', 'untuk', 'dalam', 'pada', 'dari', 'ke', 'di', 'ini', 'itu'}
        topics = [word for word in words if len(word) > 3 and word not in stop_words]
        return topics[0] if topics else "konsep ini"

class LearningModule:
    """Learning Module with Feedback and Adaptation"""
    
    def __init__(self):
        self.learning_history: List[LearningEvent] = []
        self.performance_metrics = {
            "response_quality": [],
            "user_satisfaction": [],
            "context_relevance": [],
            "response_time": []
        }
    
    def record_interaction(self, interaction_id: str, perception_data: PerceptionData, 
    reasoning_result: ReasoningResult, action_result: ActionResult, context_ids: Optional[Dict] = None):

        """Record interaction for learning"""
        # Calculate performance metrics
        performance_metrics = {
            "perception_confidence": perception_data.confidence_score,
            "reasoning_confidence": reasoning_result.confidence,
            "action_success": 1.0 if action_result.success else 0.0,
            "response_length": len(action_result.response.split()),
            "context_nodes": len(context_ids.get("nodes") or []) if context_ids else 0,
            "context_edges": len(context_ids.get("edges") or []) if context_ids else 0
        }
        
        # Identify improvement areas
        improvement_areas = []
        if perception_data.confidence_score < 0.7:
            improvement_areas.append("perception_accuracy")
        if reasoning_result.confidence < 0.8:
            improvement_areas.append("reasoning_strategy")
        if not action_result.success:
            improvement_areas.append("action_execution")
        
        learning_event = LearningEvent(
            interaction_id=interaction_id,
            user_feedback=None,  # Will be updated when feedback is received
            performance_metrics=performance_metrics,
            improvement_areas=improvement_areas
        )
        
        self.learning_history.append(learning_event)
        
        # Update running metrics
        for metric, value in performance_metrics.items():
            if metric in self.performance_metrics:
                self.performance_metrics[metric].append(value)
    
    def add_user_feedback(self, interaction_id: str, feedback: str, rating: int):
        """Add user feedback to learning event"""
        for event in self.learning_history:
            if event.interaction_id == interaction_id:
                event.user_feedback = feedback
                event.performance_metrics["user_rating"] = rating / 5.0
                break
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning history"""
        if not self.learning_history:
            return {"message": "No learning data available yet"}
        
        insights = {
            "total_interactions": len(self.learning_history),
            "average_metrics": {},
            "common_improvement_areas": {},
            "recent_performance": {}
        }
        
        # Calculate averages
        for metric, values in self.performance_metrics.items():
            if values:
                insights["average_metrics"][metric] = sum(values) / len(values)
        
        # Common improvement areas
        all_areas = [area for event in self.learning_history for area in event.improvement_areas]
        area_counts = {}
        for area in all_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        insights["common_improvement_areas"] = area_counts
        
        return insights