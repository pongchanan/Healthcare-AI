import asyncio
import httpx
from src.tools.api_wrapper import fetch_patient_live_data
from src.tools.database import query_vector_db, query_sql_db
from src.agent.router import classify_intent
from src.config import Config

async def run_fast_qa_pipeline(query: str):
    """
    Optimized pipeline for sub-0.5s latency multiple-choice QA.
    Bypasses router and parallel fetch. Assumes Vector Search is the only need.
    """
    # Step 1: Retrieval (FAST)
    # We skip intent classification and go straight to vector search
    # Assuming the query is the question itself.
    context_chunks = await query_vector_db(query)
    # context_chunks is a list of dicts: {'score': float, 'payload': {'content': str, ...}}
    context_text = "\n".join([chunk.get('payload', {}).get('content', '') for chunk in context_chunks[:2]]) # Limit to top 2 chunks for speed

    # Step 2: Synthesis (FAST)
    # using 1B model with strict single-token prompt
    prompt = f"""Context: {context_text}
Question: {query}
Answer ONLY with the correct Thai letter (ก, ข, ค, or ง). Do not explain.
Answer:"""

    try:
        payload = {
            "model": Config.SYNTHESIZER_MODEL, # 1B Model
            "prompt": prompt, # Use text completion endpoint or chat with strict prompt
            "stream": False,
            "options": {
                "temperature": 0.0, # Deterministic
                "num_predict": 5 # Max 5 tokens to prevent rambling
            }
        }
        
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(f"{Config.OLLAMA_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                # Post-processing to ensure only ก/ข/ค/ง
                valid_answers = ["ก", "ข", "ค", "ง"]
                for ans in valid_answers:
                    if ans in answer:
                        return ans
                return answer # Fallback if it didn't listen
            else:
                return "Error"
    except Exception as e:
        return f"Error: {e}"

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
