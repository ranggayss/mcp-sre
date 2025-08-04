#main.py
import os
import random
import uuid
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Import semua modul dari kode MCP Anda
from agentic_new import *
from supabase_vector_db import SupabaseVectorDb

from graph_service import graph_service
from PyPDF2 import PdfReader
from io import BytesIO
import traceback
import logging 
from fastapi.logger import logger as fastapi_logger
import json
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Integrate with FastAPI's logger
fastapi_logger.handlers = logger.handlers
fastapi_logger.setLevel(logger.level)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

print(f"Using Google API Key: {GOOGLE_API_KEY[:5]}...")  # Debugging

model_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

class ProcessPDFRequest(BaseModel):
    pdf_url: str
    session_id: str
    node_id: Optional[str] = None
    metadata: Optional[dict[str,str]] = {}

class GenerateEdgesRequest(BaseModel):
    all_nodes_data: List[Dict[str, Any]]

app = FastAPI(title="MCP Agentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "documents"
TABLE_NAME = "documents"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))

class SupabaseVectorWrapper:
    """Wrapper untuk SupabaseVectroDb agar kompatibel"""

    def __init__(self, table_name: str = "documents"):
        self.vector_db = SupabaseVectorDb(table_name=table_name)
        self.collection_name = table_name

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Wrapper untuk add_documents"""
        return self.vector_db.add_documents(
            texts=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(self, query: str, k: int = 5, filter_metadata: Optional[Dict] = None):
        return self.vector_db.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
    
    def get_collection_info(self):
        """Wrapper get collection info"""
        return self.vector_db.get_collection_info()
    
    def delete_documents(self, ids: List[str]):
        """Wrapper untuk delete documents"""
        return self.vector_db.delete_documents(ids)
    
    async def health_check(self, collection_name: Optional[str] = None) -> bool:
        try:
            print(f"[SupabaseHealthCheck] Called for: {collection_name} vs actual: {self.collection_name}")

            if collection_name and collection_name != self.collection_name:
                print(f"[SupabaseHealthCheck] Collection mismatch!")
                return False

            test = self.vector_db.similarity_search("test", k=1)
            print(f"[SupabaseHealthCheck] Similarity search results: {len(test)}")
            return True
        except Exception as e:
            print(f"[SupabaseHealthCheck] ERROR: {e}")
            return False

def init_system():
    """Initialize semua modul"""
    print("Initializing system...")
    print("Initializing with supabase...")

    try:
        supabase_db = SupabaseVectorWrapper(table_name=TABLE_NAME)
        print("v supabase vector db initalized success")
    except Exception as e:
        print(f"X failed to initialize Supabase: {e}")
        raise
    
    mcp_orchestrator = MCPOrchestrator()
    mcp_orchestrator.register_provider(VectorDBMCPProvider(supabase_db, TABLE_NAME))
    mcp_orchestrator.register_provider(WebSearchMCPProvider())
    mcp_orchestrator.register_provider(GraphDBMCPProvider(graph_service))
    mcp_orchestrator.register_provider(GraphReasoningProvider(graph_service))

    
    perception_module = EnhancedPerceptionModule()
    reasoning_module = EnhancedReasoningModule(mcp_orchestrator)
    action_module = EnhancedActionModule()
    learning_module = LearningModule()

    print(f"Action module: {action_module}") 
    
    return {
        "vector_db": supabase_db,
        "mcp": mcp_orchestrator,
        "perception": perception_module,
        "reasoning": reasoning_module,
        "action": action_module,
        "learning": learning_module
    }

system = init_system()
print("System initialized successfully")

class ChatRequest(BaseModel):
    question: str
    session_id: str
    mode: Literal['general', 'single_node', 'multi_nodes'] = 'general'
    node_id: Optional[str] = None
    node_ids: Optional[List[str]] = None
    force_web: bool = False
    context_node_ids: Optional[List[str]] = None
    context_edge_ids: Optional[List[str]] = None
    context_article_ids: Optional[List[str]] = None

    class Config:
        extra = "forbid"

class ProcessTextRequest(BaseModel):
    text: str
    session_id: str
    metadata: Optional[dict] = None

class SuggestionRequest(BaseModel):
    query: str
    context: Optional[dict] = None
    suggestion_type: Literal["input", "followup"] = "input"  # or "followup"
    chat_history: Optional[List[Dict[str, str]]] = None

class FollowupRequest(BaseModel):
    lastMessage: str
    conversationHistory: Optional[List[dict]] = []
    context: Optional[dict] = None
    suggestion_type: str = "followup"

async def handle_chat(request: ChatRequest):
    """Endpoint utama untuk chat"""
    try:
        print(f"Received chat request: {request.model_dump()}")

        if not request.session_id:
            raise HTTPException(
                status_code=422,
                detail="session_id is required"
            )

        # Validasi mode
        valid_modes = ["general", "single_node", "multi_nodes"]
        if request.mode not in valid_modes:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode. Must be one of: {valid_modes}"
            )

        additional_context = {}
        context_node_ids = request.context_node_ids or []
        context_edge_ids = request.context_edge_ids or []
        context_article_ids = request.context_article_ids or []

        if (not context_node_ids and 
            not context_edge_ids and 
            not context_article_ids and 
            not request.node_id and 
            not request.node_ids):
            
            return {
                "success": True,
                "response": "Anda belum memilih node",
                "references": [],
                "usage_metadata": None,
                "metadata": {
                    "perception": {},
                    "reasoning": {},
                    "action": {"action_type": "no_context"}
                }
            }

        if context_node_ids or context_edge_ids:
            graph_context = await graph_service.get_graph_context(
                context_node_ids,
                context_edge_ids
            )
            additional_context["graph"] = graph_context

        # 1. Perception
        perception_data = await system["perception"].perceive(
            user_input = request.question, 
            context={
                **additional_context,
            })
        
        # 2. Reasoning
        reasoning_result_dict = await graph_service.reason(
            input = perception_data.user_input, 
            force_web = request.force_web,
            external_context={
                "node_ids": request.context_node_ids,
                "edge_ids": request.context_edge_ids,
            })
        
        reasoning_result = ReasoningResult(
            strategy=reasoning_result_dict.get("strategy", "hybrid"),
            confidence=reasoning_result_dict.get("confidence", 0.8),
            context_sources=reasoning_result_dict.get("context_sources", []),
            reasoning_chain=reasoning_result_dict.get("reasoning_chain", [])
        )

        external_filters = {}
        if request.context_node_ids is not None:
            external_filters["node_ids"] = request.context_node_ids
        if request.context_edge_ids is not None:
            external_filters["edge_ids"] = request.context_edge_ids

        # Filter BARU untuk VectorDB
        if context_article_ids:
            # PENTING: Kunci filternya harus 'article_id' karena itu yang disimpan di metadata Supabase
            external_filters["article_id"] = {"in": context_article_ids}
        
        # 3. Action
        unified_context = await system["mcp"].get_unified_context(
            query = request.question, 
            providers = reasoning_result.context_sources or [],
            external_filters=external_filters or None
        ) or {"providers": {}}

        # Pastikan unified_context tidak None
        if not unified_context:
            unified_context = {
                "query": request.question,
                "timestamp": datetime.now().isoformat(),
                "providers": {}
            }

        action_result = await system["action"].act(perception_data, reasoning_result, unified_context)
        
        # 4. Learning
        system["learning"].record_interaction(
            interaction_id = request.session_id,
            perception_data = perception_data,
            reasoning_result = reasoning_result,
            action_result = action_result,
            context_ids={
                "nodes": request.context_node_ids or [],
                "edges": request.context_edge_ids or [],
            }
        )
        
        return {
            "success": True,
            "response": action_result.response,
            "references": action_result.references,
            "usage_metadata": action_result.usage_metadata,
            "metadata": {
                "perception": perception_data.__dict__,
                "reasoning": reasoning_result.__dict__,
                "action": action_result.__dict__
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "message": "internal server error",
            "error": str(e),
            "type": type(e).__name__
        })

async def process_pdf_from_url(request: ProcessPDFRequest):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.pdf_url)
            response.raise_for_status()
            
            pdf_bytes = BytesIO(response.content)
            
            try:
                reader = PdfReader(pdf_bytes)
                full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                
                if not full_text.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="PDF is empty or text extraction failed"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF text extraction error: {str(e)}"
                )

        return await process_text_internal(full_text, request)
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download PDF: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

async def process_text_internal(text: str, request: ProcessPDFRequest) -> dict:
    try:
        # 1. Preprocessing teks
        processed_text = preprocess_indonesian_text(text)
        
        # 2. Split dokumen
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=lambda x: len(x.split())
        )
        chunks = text_splitter.split_text(processed_text)

        node_id = request.node_id
        

        base_metadata = {
            "source_url": request.pdf_url or "",  # Convert None to empty string
            "session_id": request.session_id or "",
            "node_id": request.node_id or "",
            "article_id": request.metadata.get("article_id"),
            "language": "id",
            "title": request.metadata.get("title")
        }
        
        additional_metadata = request.metadata or {}
        sanitized_additional_metadata = {
            k: v if v is not None else "" for k, v in additional_metadata.items()
        }
        
        final_metadata = {**base_metadata, **sanitized_additional_metadata}
        
        metadatas = [final_metadata for _ in chunks]

        chunks_ids = [
            str(uuid.uuid4()) for _ in chunks
        ]

        try:
            stored_ids = system["vector_db"].add_documents(
                documents=chunks,
                metadatas=metadatas,
                ids=chunks_ids
            )

            generated_article_node = await generate_article_summary_node(processed_text)
            logger.info(f"DEBUG: generated_article_node before returning: {generated_article_node}")

            generated_article_node['att_url'] = request.pdf_url

            token_usage = generated_article_node.pop('token_usage', {})
        
            return {
                "success": True,
                "num_chunks": len(chunks),
                # "document_ids": [f"{request.session_id}_{i}" for i in range(len(chunks))]
                "document_ids": stored_ids,
                "node_id": node_id,
                "session_id": request.session_id,
                "generated_article_node": generated_article_node,
                "token_usage": token_usage
            }
        
        except Exception as e:
            print(f"error saving to Supabase: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {str(e)}"
            )
        
    except Exception as e:
        print(f"Error in text processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text processing error: {str(e)}"
        )
    
def preprocess_indonesian_text(text: str) -> str:
    """
    Preprocessing khusus teks Bahasa Indonesia.
    Handle karakter khusus, normalisasi, dll.
    """
    # 1. Normalisasi karakter khusus
    text = (
        text.replace("â€œ", '"')  # Kutipan curly
        .replace("â€ ", '"')
        .replace("â€™", "'")  # Apostrof
        .replace("â€˜", "'")
        .replace("â€”", "-")  # Dash
    )
    
    # 2. Koreksi singkatan umum
    replacements = {
        ' tdk ': ' tidak ',
        ' yg ': ' yang ',
        ' dgn ': ' dengan ',
        ' pd ': ' pada ',
        ' jg ': ' juga '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # 3. Hapus whitespace berlebihan
    text = ' '.join(text.split())
    
    return text

# --- FUNGSI AI: Generate Satu Node Ringkasan Artikel ---
async def generate_article_summary_node(text: str) -> Dict[str, Any]:
    truncated_text = text[:50000] 

    prompt = f"""
Berikut adalah isi artikel ilmiah:

"{truncated_text}"

Buat **satu** ringkasan artikel ilmiah dalam format JSON dengan struktur:
{{
  "label": "Ringkasan Artikel",
  "type": "article",
  "title": "Judul Artikel (dari teks artikel, jika ada)",
  "content": "Rangkuman umum dari isi artikel (maksimal 1000 kata, dalam Bahasa Indonesia)",
  "att_goal": "Tujuan dari penelitian ini",
  "att_method": "Metodologi yang digunakan",
  "att_background": "Latar belakang penelitian",
  "att_future": "Arahan penelitian masa depan",
  "att_gaps": "Kekurangan atau gap dari penelitian"
}}

Pastikan semua field terisi. Jika informasi tidak ada dalam teks, tulis string kosong ("").
Berikan hanya **JSON murni** tanpa teks tambahan atau blok kode markdown (seperti ```json).
    """

    try:
        response = await model_flash.ainvoke(prompt)
        text_output = response.content.strip()

        #extract token usage
        token_usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            # logger.info(f"🔍 Found usage_metadata: {response.usage_metadata}")
            token_usage = {
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            logger.info(f"✅ Successfully extracted token usage: {token_usage}")
        else:
            logger.warning("⚠️ No usage_metadata found in response")
        
        cleaned_output = text_output.replace("```json", "").replace("```", "").strip()

        # logger.info(f"🧠 Raw Gemini (summary) output:\n{text_output[:500]}...")
        # logger.info(f"🧹 Cleaned JSON (summary):\n{cleaned_output[:500]}...")
        logger.info(f"📊 Token usage: {token_usage}")

        parsed_node = json.loads(cleaned_output)

        parsed_node['label'] = parsed_node.get('label', 'Ringkasan Artikel')
        parsed_node['type'] = parsed_node.get('type', 'article')
        parsed_node['title'] = parsed_node.get('title', 'Untitled Article')
        parsed_node['content'] = parsed_node.get('content', '')
        parsed_node['att_goal'] = parsed_node.get('att_goal', '')
        parsed_node['att_method'] = parsed_node.get('att_method', '')
        parsed_node['att_background'] = parsed_node.get('att_background', '')
        parsed_node['att_future'] = parsed_node.get('att_future', '')
        parsed_node['att_gaps'] = parsed_node.get('att_gaps', '')

        #add token usage to the result
        parsed_node['token_usage'] = token_usage
        
        return parsed_node
    except Exception as e:
        logger.error(f"❌ Error generating article summary node with AI: {traceback.format_exc()}")
        return {
            "label": "Ringkasan Artikel",
            "type": "article",
            "title": "Error Summary",
            "content": "",
            "att_goal": "",
            "att_method": "",
            "att_background": "",
            "att_future": "",
            "att_gaps": "",
            "token_usage": {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        }

# --- FUNGSI AI: Generate Edges Antar Nodes yang Diberikan ---
async def generate_edges_from_all_nodes(all_nodes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    default_token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    if len(all_nodes_data) < 2:
        logger.info("Less than 2 nodes for edge generation. Returning empty list.")
        # return []
        return {
            "edges": [],
            "token_usage": default_token_usage
        }

    prompt = f"Anda adalah asisten AI yang bertugas menganalisis hubungan semantik antar artikel ilmiah berdasarkan isi kontennya.\n"
    prompt += f"Semua konten di bawah ini ditulis dalam **Bahasa Indonesia**. Fokus pada kemiripan makna, bukan sekadar kemiripan kata.\n"
    prompt += f"Setiap artikel memiliki atribut berikut:\n- Judul\n- Tujuan\n- Metodologi\n- Latar Belakang\n- Arahan Penelitian Masa Depan\n- Gap/Kekurangan Penelitian\n\n"

    prompt += f"Tugas Anda adalah menganalisis kemungkinan **hubungan semantik** antar artikel. Jenis hubungan tersebut meliputi:\n"
    prompt += f"- same_background: artikel memiliki latar belakang atau konteks yang serupa\n"
    prompt += f"- extended_method: artikel B mengembangkan atau membangun dari metode artikel A\n"
    prompt += f"- shares_goal: artikel memiliki tujuan yang sama atau saling melengkapi\n"
    prompt += f"- follows_future_work: artikel mengikuti atau mewujudkan arahan masa depan dari artikel lain\n"
    prompt += f"- addresses_same_gap: kedua artikel mencoba mengatasi kekurangan atau gap penelitian yang sama\n\n"

    for idx, node in enumerate(all_nodes_data):
        prompt += f"Artikel {idx + 1} (ID: {node.get('id')}, Judul : \"{node.get('title', 'Untitled')}\"):\n"
        prompt += f"- Judul: {node.get('title', '')}\n"
        prompt += f"- Tujuan: {node.get('att_goal', '')}\n"
        prompt += f"- Metodologi: {node.get('att_method', '')}\n"
        prompt += f"- Latar Belakang: {node.get('att_background', '')}\n"
        prompt += f"- Arahan Masa Depan: {node.get('att_future', '')}\n"
        prompt += f"- Gap/Kekurangan: {node.get('att_gaps', '')}\n"
        prompt += f"- URL: {node.get('att_url', '')}\n\n"

    prompt += f"Sekarang kembalikan sebuah array JSON yang berisi relasi (\"edges\") antar artikel.\n"
    prompt += f"Format setiap elemen:\n\n"
    prompt += f"[\n"
    prompt += f"  {{\n"
    prompt += f"    \"from\": <id_artikel_sumber>,\n"
    prompt += f"    \"to\": <id_artikel_tujuan>,\n"
    prompt += f"    \"relation\": \"<jenis_relasi>\",\n"
    prompt += f"    \"label\": \"<deskripsi singkat dalam Bahasa Indonesia>\"\n"
    prompt += f"  }}\n"
    prompt += f"]\n\n"
    prompt += f"Jika tidak ada relasi, cukup kembalikan array kosong [] tanpa penjelasan tambahan.\n"
    prompt += f"Tolong **bungkus jawaban JSON dalam blok kode seperti berikut**:\n\n"
    prompt += "```json\n"
    prompt += "[ ... ]\n"
    prompt += "```\n"
    prompt += f"Gunakan **judul artikel** (bukan hanya ID atau \"Artikel 1\") dalam deskripsi relasi. \n"
    prompt += f"Contoh label yang baik:\n"
    prompt += f"- \"Artikel 'Analisis Usability Aplikasi XYZ' menggunakan metode yang disederhanakan dari artikel 'Studi SEM-PLS pada Aplikasi XYZ'\"\n\n"

    try:
        response = await model_flash.ainvoke(prompt) 
        text_output = response.content.strip()


        # Extract token usage information
        token_usage = default_token_usage.copy()
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata  # Dictionary
            token_usage = {
                'input_tokens': usage.get('input_tokens', 0),
                'output_tokens': usage.get('output_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            logger.info(f"✅ Successfully extracted edge token usage: {token_usage}")
        else:
            logger.warning("⚠️ No usage_metadata found in edge response")


        json_str = ""
        if "```json" in text_output and "```" in text_output:
            start_idx = text_output.find("```json") + len("```json")
            end_idx = text_output.find("```", start_idx)
            if start_idx != -1 and end_idx != -1:
                json_str = text_output[start_idx:end_idx].strip()
            else:
                json_str = text_output
        else:
            json_str = text_output

        # logger.info(f"🧠 Raw Gemini (edges) output:\n{text_output[:500]}...")
        # logger.info(f"🧹 Extracted JSON (edges):\n{json_str[:500]}...")
        logger.info(f"📊 Edge generation token usage: {token_usage}")

        edges = json.loads(json_str)
        if isinstance(edges, list):
            return {
                "edges": edges,
                "token_usage": token_usage
            }
        
        logger.warning(f"AI response for edges was not a list: {edges}")
        return {
            "edges": edges,
            "token_usage": token_usage
        }
    except Exception as e:
        logger.error(f"❌ Failed to generate edges with AI: {traceback.format_exc()}")
        return {
            "edges": edges,
            "token_usage": token_usage
        }


# @app.post("/api/suggestions")
async def get_suggestions(request: SuggestionRequest):
    try:
        context = request.context or {}
        node_ids = context.get("nodeIds", [])
        edge_ids = context.get("edgeIds", [])

        # 1. Fetch node metadata from your graph context endpoint
        context_data = {}
        if node_ids:
            context_data = await graph_service.get_graph_context(node_ids=node_ids, edge_ids=edge_ids)
        
        # 2. Extract node titles or labels
        node_titles = [n.get("title") or n.get("label") for n in context_data.get("nodes", []) if n.get("title") or n.get("label")]
        # 1. Ambil judul dari node kalau ada
        if node_titles:
            topic_description = ", ".join(node_titles[:3])

        # 2. Kalau tidak ada node, ambil query user kalau cukup informatif
        elif request.query and len(request.query.strip()) > 5 and request.query.lower() not in ["general", "umum", "topik umum"]:
            topic_description = "topik riset ilmiah atau brainstorming"

        # 3. Fallback terakhir: topik default
        else:
            topic_description = random.choice([
                "potensi riset interdisipliner di era digital",
                "inovasi dalam metode penelitian akademik",
                "tren terbaru dalam kecerdasan buatan",
                "tantangan etis dalam publikasi ilmiah",
                "kolaborasi riset global di masa depan"
            ])


        # 3. Buat prompt kontekstual
        prompt = f"""Kamu adalah asisten brainstorming. Buat 5 saran eksploratif berdasarkan topik: "{topic_description}"

Saran bisa berupa:
- pertanyaan kritis
- perbandingan ide
- gap yang belum dijawab
- metode alternatif
- contoh penerapan di bidang lain

Format:
1. ...
2. ...
3. ...
        """

        print(f"🧠 Prompt yang digunakan:\n{prompt[:200]}...")

        # 4. Generate suggestion
        suggestions = await system["action"].generate_suggestions(prompt)

        clean_suggestions = []
        for s in suggestions:
            if s and len(s.strip()) > 5:
                if s[0].isdigit() and s[1:3] == '. ':
                    clean_suggestions.append(s[3:].strip())
                else:
                    clean_suggestions.append(s.strip())

        return {
            "success": True,
            "suggestions": clean_suggestions[:5] or [
                f"Apa gap dari {topic_description}?",
                f"Studi kasus {topic_description}",
                f"Pendekatan alternatif untuk {topic_description}",
                f"Permasalahan utama dalam {topic_description}",
                f"Aplikasi nyata dari {topic_description}"
            ]
        }

    except Exception as e:
        print(f"❌ Error while generating suggestions: {e}")
        return {"success": False, "suggestions": []}

    
# @app.post("/api/suggestions/followup")
async def get_followup_suggestions(request: FollowupRequest):
    try:
        # Buat context dari conversation history
        conversation_context = ""
        if request.conversationHistory:
            recent_messages = request.conversationHistory[-3:]  # Ambil 3 terakhir
            conversation_context = "\n".join([
                f"{'User' if msg.get('sender') == 'user' else 'AI'}: {msg.get('text', '')[:100]}..."
                for msg in recent_messages
            ])
        
        # Buat context dari nodes/edges
        node_context = ""
        if request.context and (request.context.get("nodeIds") or request.context.get("edgeIds")):
            node_context = f"\nContext: Berdasarkan {len(request.context.get('nodeIds', []))} nodes dan {len(request.context.get('edgeIds', []))} edges"
        
        prompt = f"""Berdasarkan jawaban AI terakhir, buat 5 pertanyaan lanjutan yang relevan dan mendalam:

Jawaban AI terakhir: "{request.lastMessage[:300]}..."

Konteks percakapan sebelumnya:
{conversation_context}
{node_context}

Buat pertanyaan follow-up yang:
- Menggali lebih dalam dari jawaban yang diberikan
- Mengeksplorasi aspek praktis atau implementasi
- Menanyakan contoh konkret atau studi kasus
- Mempertanyakan hubungan dengan konsep lain
- Menanyakan tentang tantangan atau limitasi

Format: berikan 5 pertanyaan dengan format numerik (1. ... 2. ...)
Setiap pertanyaan maksimal 10 kata dan berupa kalimat tanya."""
        
        suggestions = await system["action"].generate_followup_suggestions(prompt)
        
        # Filter dan bersihkan suggestions
        clean_suggestions = []
        for s in suggestions:
            if len(s) > 10:
                # Hapus prefix nomor jika ada
                if s[0].isdigit() and len(s) > 3 and s[1:3] == '. ':
                    clean_suggestions.append(s[3:].strip())
                else:
                    clean_suggestions.append(s.strip())
        
        return {
            "success": True,
            "suggestions": clean_suggestions[:3] or [
                f"Bagaimana cara mengimplementasikan konsep ini?",
                f"Apa tantangan utama dalam penerapannya?",
                f"Bisakah berikan contoh kasus nyata?",
                f"Bagaimana perkembangan terbaru di bidang ini?",
                f"Apa perbedaannya dengan pendekatan lain?"
            ]
        }
        
    except Exception as e:
        print(f"❌ Followup Error: {str(e)}")
        return {
            "success": False,
            "suggestions": [
                "Bisakah dijelaskan lebih detail?",
                "Bagaimana cara praktis menerapkannya?",
                "Apa contoh kasus penggunaannya?",
                "Bagaimana tren perkembangannya?",
                "Apa kelebihan dan kekurangannya?"
            ]
        }

# --- Endpoint Tambahan ---
@app.post("/api/feedback")
async def submit_feedback(interaction_id: str, feedback: str, rating: int):
    """Catat feedback pengguna"""
    try:
        system["learning"].add_user_feedback(interaction_id, feedback, rating)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system-info")
async def system_info():
    """Get info tentang sistem"""
    return {
        "providers": list(system["mcp"].providers.keys()),
        "learning_metrics": system["learning"].get_learning_insights()
    }

@app.get("/api/debug/chroma-metadata")
async def debug_chroma_metadata(limit: int = 3):
    """Enhanced debug endpoint with proper error handling"""
    try:
        # 1. Get ChromaDB collection
        collection = system["chroma_db"].client.get_collection(COLLECTION_NAME)
        
        # 2. Get records with proper type handling
        # records = collection.get(
        #     limit=min(limit, 10),
        #     include=["metadatas", "documents"]
        # )

        if limit == -1:
            records = collection.get(include=["metadatas", "documents"])
        else:
            records = collection.get(limit=limit, include=["metadatas", "documents"])

        total_chars = sum(len(doc) for doc in records["documents"] if doc)

        # 3. Safely format samples
        samples = []
        for i in range(min(len(records["ids"]), min(limit, 10))):
            doc = records["documents"][i] if i < len(records["documents"]) else None
            meta = records["metadatas"][i] if i < len(records["metadatas"]) else {}
            
            samples.append({
                "id": records["ids"][i],
                "metadata": meta,
                "document_preview": f"{doc[:100]}..." if doc else None
            })

        return {
            "success": True,
            "collection": COLLECTION_NAME,
            "count": len(records["ids"]),
            "total_characters": total_chars,
            "samples": samples
        }

    except Exception as e:
        logger.error(f"Debug error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Failed to retrieve metadata: {str(e)}",
            "details": {
                "collection": COLLECTION_NAME,
                "available_collections": system["chroma_db"].client.list_collections()
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)