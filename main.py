from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.workflow import run_agent_pipeline, run_fast_qa_pipeline
import time
import os

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    print("\n\n🔥 [SYSTEM] API v2.5 - Chat Mode & Nomic Embeddings Loaded 🔥\n\n")

@app.post("/api/ask")
async def ask_agent(request: QueryRequest):
    start_time = time.time()
    
    try:
        # FAST MODE for 0.5s latency target
        answer = await run_fast_qa_pipeline(request.question)
        # answer = await run_agent_pipeline(request.question)
        
        process_time = (time.time() - start_time) * 1000
        print(f" [Log] Processed in {process_time:.2f}ms")
        
        return {
            "answer": answer,
            "latency_ms": process_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
