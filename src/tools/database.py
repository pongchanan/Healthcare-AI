import duckdb
import os
import pickle
from src.config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Global caches for singletons
BM25_DATA = None
CROSS_ENCODER_MODEL = None

def get_sql_connection():
    return duckdb.connect(Config.SQL_DB_PATH)

async def query_sql_db(query: str):
    """Executes a read-only SQL query against DuckDB."""
    try:
        conn = get_sql_connection()
        results = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]
        conn.close()
        return [dict(zip(columns, row)) for row in results]
    except Exception as e:
        print(f" [SQL Error] {e}")
        return []

def load_bm25():
    """Lazy load BM25 index"""
    global BM25_DATA
    if BM25_DATA is None:
        try:
            with open("data/bm25_data.pkl", "rb") as f:
                BM25_DATA = pickle.load(f)
            print(" [System] BM25 Index loaded.")
        except Exception:
            print(" [System] BM25 not found. Hybrid search will be partial.")
            BM25_DATA = {}
    return BM25_DATA

def load_reranker():
    """Lazy load Cross-Encoder"""
    global CROSS_ENCODER_MODEL
    if CROSS_ENCODER_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            # TinyBERT is super fast (10-20ms per doc)
            CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2') 
            print(" [System] CrossEncoder loaded.")
        except Exception as e:
            print(f" [System] Failed to load CrossEncoder: {e}")
    return CROSS_ENCODER_MODEL

async def query_vector_db(query_text: str, collection_name: str = "medical_docs", limit: int = 10):
    """Standard Vector Search"""
    try:
        from src.config import Config
        import httpx
        
        vector = []
        try:
            url = f"{Config.OLLAMA_BASE_URL}/api/embeddings"
            payload = {
                "model": Config.EMBEDDING_MODEL, 
                "prompt": f"search_query: {query_text}"
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    vector = response.json().get("embedding")
        except Exception as e:
            print(f" [Embedding Error] {e}")
        
        if not vector: return []

        client = QdrantClient(path=Config.VECTOR_DB_PATH)
        search_result = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit
        ).points

        return [{"score": hit.score, "payload": hit.payload, "id": hit.id} for hit in search_result]
    except Exception as e:
        print(f" [Vector DB Error] {e}")
        return []

async def hybrid_search(query_text: str, limit: int = 5):
    """
    Combines Vector Search + BM25 using Reciprocal Rank Fusion (RRF).
    """
    # 1. Get Vector Results
    vector_results = await query_vector_db(query_text, limit=20) # Get more for fusion
    
    # 2. Get BM25 Results
    bm25_data = load_bm25()
    bm25_results = []
    
    if bm25_data and "bm25" in bm25_data:
        try:
            from pythainlp.tokenize import word_tokenize
            tokenized_query = word_tokenize(query_text, engine="newmm")
            bm25 = bm25_data["bm25"]
            docs = bm25_data["documents"]
            
            scores = bm25.get_scores(tokenized_query)
            # Get top N indices
            top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
            
            for i in top_n:
                if scores[i] > 0:
                    bm25_results.append({
                        "score": scores[i],
                        "payload": docs[i],
                        "id": i # Use index as faux ID for local docs
                    })
        except Exception as e:
            print(f" [BM25 Error] {e}")

    # 3. RRF Fusion
    # Map doc content hash to score
    fusion_scores = {}
    k = 60
    
    # Process Vector
    for rank, doc in enumerate(vector_results):
        # Use content as unique key since IDs might differ
        key = hash(doc['payload']['content'])
        fusion_scores[key] = {
            "score": (1 / (k + rank + 1)),
            "payload": doc['payload']
        }
        
    # Process BM25
    for rank, doc in enumerate(bm25_results):
        key = hash(doc['payload']['content'])
        if key in fusion_scores:
            fusion_scores[key]["score"] += (1 / (k + rank + 1))
        else:
            fusion_scores[key] = {
                "score": (1 / (k + rank + 1)),
                "payload": doc['payload']
            }
            
    # Sort by fused score
    final_results = sorted(fusion_scores.values(), key=lambda x: x['score'], reverse=True)
    return final_results[:limit]

async def rerank_results(query: str, chunks: list, top_k: int = 3):
    """
    Re-ranks chunks using CrossEncoder.
    """
    reranker = load_reranker()
    if not reranker or not chunks:
        return chunks[:top_k]
        
    try:
        pairs = [[query, chunk['payload']['content']] for chunk in chunks]
        scores = reranker.predict(pairs)
        
        # Attach new scores
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
            
        # Resort
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        print(f" [Rerank Error] {e}")
        return chunks[:top_k]
