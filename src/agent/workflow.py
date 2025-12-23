import asyncio
import httpx
from src.tools.api_wrapper import fetch_patient_live_data
from src.tools.database import query_vector_db, query_sql_db
from src.agent.router import classify_intent
from src.config import Config

async def run_agent_pipeline(query: str):
    """
    Main execution pipeline.
    Strategy: Route -> Parallel Fetch -> Synthesize
    """
    
    # Step 1: Route (Intent Classification)
    intent = await classify_intent(query)
    print(f" [Router] Intent: {intent}")

    # Step 2: Parallel Execution (Asyncio Gather)
    tasks = []
    
    if intent in ["api_lookup", "hybrid"]:
        # Extract ID (simplified logic)
        words = query.split()
        pid = "123" # Default
        for w in words:
            if w.isdigit():
                pid = w
                break
        tasks.append(fetch_patient_live_data(pid))
        
    if intent in ["vector_search", "hybrid"]:
        tasks.append(query_vector_db(query))
        
    if intent in ["sql_query", "hybrid"]:
        # Logic to generate SQL would go here. For now, we run a safe example query if intent is SQL-heavy
        # In a real app, we'd use an LLM to generate the SQL
        tasks.append(query_sql_db("SELECT * FROM patients LIMIT 5")) 

    # Wait for all tools to finish
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Step 3: Final Synthesis
    # Pack all results into context
    final_context = f"Retrieved Data: {str(results)}"
    
    system_prompt = "You are a helpful Thai-English medical assistant. Answer based ONLY on context."
    if "Thai" in query or any(char in query for char in "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"):
        system_prompt += " Answer in Thai."
    
    try:
        payload = {
            "model": Config.SYNTHESIZER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {final_context}\n\nQuestion: {query}"}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client: # Longer timeout for generation
            response = await client.post(f"{Config.OLLAMA_BASE_URL}/api/chat", json=payload)
            if response.status_code == 200:
                return response.json()['message']['content']
            else:
                return f"Error from model: {response.text}"
                
    except Exception as e:
        return f"Error generating response: {str(e)}"
