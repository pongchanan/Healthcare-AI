import duckdb
import os
from src.config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize global clients
# Qdrant client (Local file persistent)
qdrant_client = QdrantClient(path=Config.VECTOR_DB_PATH)

def get_sql_connection():
    return duckdb.connect(Config.SQL_DB_PATH)

async def query_sql_db(query: str):
    """
    Executes a read-only SQL query against DuckDB.
    """
    try:
        conn = get_sql_connection()
        # Ensure read-only if possible or just execute
        results = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        conn.close()
        
        # Convert to list of dicts for easier consumption
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        print(f" [SQL Error] {e}")
        return []

async def query_vector_db(query_text: str, collection_name: str = "medical_docs", limit: int = 3):
    """
    Searches the Qdrant vector store using Ollama embeddings.
    """
    try:
        from src.config import Config
        import httpx
        
        # Get embedding from Ollama
        vector = []
        try:
            url = f"{Config.OLLAMA_BASE_URL}/api/embeddings"
            payload = {
                "model": Config.SYNTHESIZER_MODEL, 
                "prompt": query_text
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    vector = response.json().get("embedding")
        except Exception as e:
            print(f" [Embedding Error] {e}")
        
        if not vector:
             # Look for existing collection to guess dim if failed? Or just return empty
             # For robustness, we try to proceed if we can, but real search needs vector.
             return []

        # qdrant-client v1.10+ uses query_points for search in some contexts or if search is deprecated/missing
        # result structure: QueryResponse(points=[ScoredPoint(...), ...])
        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit
        ).points

        return [{"score": hit.score, "payload": hit.payload} for hit in search_result]
    except Exception as e:
        print(f" [Vector DB Error] {e}")
        return []
