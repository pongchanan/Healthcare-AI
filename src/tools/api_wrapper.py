import httpx
import asyncio
from cachetools import TTLCache
from src.config import Config

# In-memory cache: Stores results for 5 minutes (300s)
api_cache = TTLCache(maxsize=1000, ttl=300)

async def fetch_patient_live_data(patient_id: str):
    """
    Wraps the external API.
    1. Checks Cache first (Speed).
    2. Calls API if miss.
    3. Handles timeouts/errors gracefully.
    """
    if patient_id in api_cache:
        print(f" [Cache Hit] Returning data for {patient_id}")
        return api_cache[patient_id]

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{Config.BAD_API_ENDPOINT}/patients/{patient_id}")
            response.raise_for_status()
            data = response.json()
            
            api_cache[patient_id] = data
            return data
        
    except httpx.HTTPError as e:
        print(f" [API Error] {e}")
        return {"error": "Data unavailable", "details": str(e)}
    except Exception as e:
        print(f" [System Error] {e}")
        return {"error": "System error", "details": str(e)}
