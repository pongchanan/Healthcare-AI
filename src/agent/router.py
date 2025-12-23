import re
import httpx
import json
from src.config import Config

async def classify_intent(query: str):
    """
    Decides WHICH tool to use.
    Priority 1: Regex (0ms) - for IDs or known keywords.
    Priority 2: Tiny LLM (Typhoon 1B) - for complex questions.
    """
    
    # 1. Regex Heuristics (Instant)
    if re.search(r'\b(id|รหัส|number)\s*:?\s*\d+', query, re.IGNORECASE):
        return "api_lookup"
    
    if any(keyword in query for keyword in ["avg", "average", "count", "total", "กี่คน", "เฉลี่ย"]):
        return "sql_query"

    # 2. Fallback to Typhoon 1B Router (Fast) via Ollama/Local API
    try:
        payload = {
            "model": Config.ROUTER_MODEL,
            "prompt": f"Classify query: '{query}'. Options: [1] Vector Search (Knowledge) [2] SQL (Stats/Table) [3] API (Realtime) [4] Hybrid. Reply ONLY with digit.",
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.post(f"{Config.OLLAMA_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                if '1' in result: return "vector_search"
                if '2' in result: return "sql_query"
                if '3' in result: return "api_lookup"
            
    except Exception as e:
        print(f" [Router Error] {e}")
    
    return "hybrid" # Default safe fallback
