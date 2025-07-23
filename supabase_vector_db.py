import json
import logging
import os
import numpy as np
from supabase import create_client, Client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any, Optional
import uuid
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

print(f"Using Google API Key: {GOOGLE_API_KEY[:5]}...")  # Debugging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Inisialisasi Sistem ---
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

class SupabaseVectorDb:
    def __init__(
        self,
        table_name: str = "documents",
        supabase_url: str = None,
        supabase_key: str = None
    ):
        """
        Inisialisasi Supabase Vector Database
        
        Args:
            table_name: Nama tabel di Supabase
            supabase_url: URL Supabase project
            supabase_key: Supabase service role key
        """
        self.table_name = table_name
        self.embedding_function = EMBEDDING_MODEL
        
        # Ambil kredensial dari environment variables jika tidak disediakan
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and Service Key harus disediakan")
        
        # Inisialisasi client Supabase
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Pastikan tabel ada
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """
        Pastikan tabel dengan struktur yang tepat ada di Supabase
        
        Anda perlu membuat tabel ini di Supabase SQL Editor:
        
        CREATE TABLE IF NOT EXISTS documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}',
            embedding vector(768), -- Sesuaikan dimensi dengan model embedding
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Buat index untuk pencarian vector
        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
        ON documents USING ivfflat (embedding vector_cosine_ops);
        
        -- Enable RLS jika diperlukan
        ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
        """
        print(f"Pastikan tabel '{self.table_name}' sudah dibuat di Supabase dengan struktur yang benar")
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Tambahkan dokumen ke Supabase Vector Database
        
        Args:
            texts: List teks dokumen
            metadatas: List metadata untuk setiap dokumen
            ids: List ID untuk setiap dokumen (opsional)
            
        Returns:
            List ID dokumen yang disimpan
        """
        if not texts:
            return []
        
        # Generate IDs jika tidak disediakan
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Generate metadata default jika tidak disediakan
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # Generate embeddings untuk semua teks
        embeddings = self.embedding_function.embed_documents(texts)
        
        # Siapkan data untuk insert
        documents_data = []
        for i, (text, metadata, doc_id, embedding) in enumerate(zip(texts, metadatas, ids, embeddings)):
            documents_data.append({
                "id": doc_id,
                "content": text,
                "metadata": metadata,
                "embedding": embedding
            })
        
        # Insert ke Supabase
        try:
            result = self.client.table(self.table_name).insert(documents_data).execute()
            print(f"Berhasil menambahkan {len(documents_data)} dokumen")
            return ids
        except Exception as e:
            print(f"Error saat menambahkan dokumen: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None # Ini adalah filter yang masuk
    ) -> List[Dict[str, Any]]:
        """
        Cari dokumen berdasarkan similarity dengan query, meneruskan filter metadata
        langsung ke fungsi RPC Supabase.
        """
        query_embedding = self.embedding_function.embed_query(query)
        
        # --- HAPUS BAGIAN INI ---
        # supabase_query = self.client.table(self.table_name).select("*")
        # if filter_metadata:
        #     for key, value in filter_metadata.items():
        #         supabase_query = supabase_query.eq(f"metadata->{key}", value)
        # --- END HAPUS ---

        try:
            # Siapkan filter_params untuk dikirim ke fungsi RPC.
            # Jika filter_metadata kosong atau None, kirim objek JSON kosong.
            rpc_filter_params = filter_metadata if filter_metadata is not None else {}
            
            result = self.client.rpc(
                "similarity_search", 
                {
                    "query_embedding": query_embedding,
                    "similarity_threshold": 0.1,
                    "match_count": k,
                    "table_name": self.table_name,
                    "filter_params": rpc_filter_params # <--- TERUSKAN FILTER DI SINI!
                }
            ).execute()
            
            return result.data
        except Exception as e:
            print(f"Error saat melakukan similarity search: {e}")
            # Anda juga harus memperbarui _fallback_similarity_search agar menangani filter_metadata.
            return self._fallback_similarity_search(query_embedding, k, filter_metadata)

        
    def _get_all_documents(self, k: int, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get all documents when query is empty"""
        try:
            query = self.client.table(self.table_name).select("*").limit(k)
            
            if filter_metadata:
                for key, value in filter_metadata.items():
                    query = query.eq(f"metadata->{key}", value)
            
            result = query.execute()
            documents = result.data or []
            
            # Parse metadata if it's stored as string
            for doc in documents:
                if isinstance(doc.get('metadata'), str):
                    try:
                        doc['metadata'] = json.loads(doc['metadata'])
                    except:
                        doc['metadata'] = {}
            
            return documents
            
        except Exception as e:
            print(f"Error getting all documents: {e}")
            return []
    
    def _fallback_similarity_search(self, query_embedding: List[float], k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        print("[FALLBACK] Performing fallback similarity search in Python.")
        try:
            supabase_query = self.client.table(self.table_name).select("id, content, metadata, embedding")
            
            if filter_metadata:
                for key, value_obj in filter_metadata.items():
                    if isinstance(value_obj, dict) and "in" in value_obj:
                        # Ini akan bekerja untuk filter {"article_id": {"in": ["ID1", "ID2"]}}
                        supabase_query = supabase_query.in_(f"metadata->>{key}", value_obj["in"])
                    else:
                        # Ini untuk filter sederhana {"key": "value"}
                        # Mengasumsikan nilai string untuk kesederhanaan.
                        # Jika nilai bisa berupa angka/boolean, Anda mungkin perlu konversi di sini juga.
                        supabase_query = supabase_query.eq(f"metadata->>{key}", value_obj)


            all_documents_from_db = supabase_query.execute().data

            if not all_documents_from_db:
                print("[FALLBACK] No documents found after applying filters in fallback.")
                return []

            # Pastikan query_embedding adalah list of float
            if not all(isinstance(x, (int, float)) for x in query_embedding):
                print(f"[FALLBACK ERROR] query_embedding contains non-numeric values: {query_embedding}")
                return []

            results_with_similarity = []
            for doc in all_documents_from_db:
                doc_embedding_raw = doc.get('embedding')
                if doc_embedding_raw:
                    # --- FOKUS PERBAIKAN DI SINI ---
                    # Asumsikan embedding disimpan sebagai string representasi array atau list
                    # Anda perlu mengkonversinya ke list float.
                    try:
                        # Ini mencoba parse dari format string "[1.2, 3.4, ...]"
                        doc_embedding = json.loads(doc_embedding_raw) 
                    except json.JSONDecodeError:
                        # Jika disimpan sebagai string "{1.2,3.4,...}" (Postgres array literal)
                        doc_embedding = [float(x) for x in doc_embedding_raw.strip('{}').split(',')]
                    
                    if not all(isinstance(x, (int, float)) for x in doc_embedding):
                         print(f"[FALLBACK ERROR] doc_embedding for ID {doc.get('id')} contains non-numeric values: {doc_embedding}")
                         continue # Lewati dokumen ini jika embedding-nya tidak valid
                    # --- END FOKUS ---

                    # Pastikan kedua embedding memiliki panjang yang sama untuk dot product
                    if len(query_embedding) != len(doc_embedding):
                        logger.warning(f"Embedding length mismatch for doc ID {doc.get('id')}. Skipping.")
                        continue

                    # Hitung similarity secara manual (cosine similarity)
                    dot_product = sum(x * y for x, y in zip(query_embedding, doc_embedding))
                    norm_query = sum(x*x for x in query_embedding)**0.5
                    norm_doc = sum(y*y for y in doc_embedding)**0.5

                    if norm_query == 0 or norm_doc == 0:
                        similarity = 0 # Hindari pembagian nol
                    else:
                        similarity = dot_product / (norm_query * norm_doc)
                    
                    # Konversi cosine similarity ke "jarak" jika 1.0 = sempurna
                    # LangChain sering menggunakan 1 - cosine_similarity sebagai jarak
                    # Kalau fungsi RPC Anda mengembalikan 1 - (embedding <=> $1), itu sudah "jarak".
                    # Jadi, sesuaikan dengan apa yang Anda inginkan untuk 'similarity' atau 'distance'.
                    # Jika similarity, biarkan saja:
                    results_with_similarity.append({
                        "id": doc.get('id'),
                        "content": doc.get('content'),
                        "metadata": doc.get('metadata'),
                        "similarity": similarity # Atau 1 - similarity jika ingin jarak
                    })
            
            # Urutkan berdasarkan similarity (tertinggi ke terendah) dan ambil k teratas
            sorted_results = sorted(results_with_similarity, key=lambda x: x['similarity'], reverse=True)
            
            # Filter berdasarkan similarity_threshold jika Anda memiliki satu di RPC
            final_results = [r for r in sorted_results if r['similarity'] >= 0.1] # Gunakan threshold yang sama dengan RPC
            
            return final_results[:k]
        except Exception as e:
            logger.error(f"Error in fallback similarity search: {e}", exc_info=True)
            return []

    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Hitung cosine similarity antara dua vector"""
        a_np = np.array(a)
        b_np = np.array(b)
        
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Hapus dokumen berdasarkan ID
        
        Args:
            ids: List ID dokumen yang akan dihapus
            
        Returns:
            True jika berhasil
        """
        try:
            result = self.client.table(self.table_name).delete().in_("id", ids).execute()
            print(f"Berhasil menghapus {len(ids)} dokumen")
            return True
        except Exception as e:
            print(f"Error saat menghapus dokumen: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi tentang koleksi
        
        Returns:
            Dictionary dengan informasi koleksi
        """
        try:
            # Hitung jumlah dokumen
            count_result = self.client.table(self.table_name).select("*", count="exact").execute()
            
            return {
                "table_name": self.table_name,
                "document_count": count_result.count,
                "embedding_model": "models/embedding-001"
            }
        except Exception as e:
            print(f"Error saat mendapatkan info koleksi: {e}")
            return {}