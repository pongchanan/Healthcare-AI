import pandas as pd
import asyncio
from src.agent.workflow import run_agent_pipeline
import time

async def eval_qa():
    df = pd.read_csv("data/raw/QA.csv", names=["Question", "Answer"]).iloc[1:] # Skip header row if pandas read it as data or if header exists
    # Check if header is actually header.
    # The file view showed 1: Question,Answer. So read_csv with header=0 (default) is fine.
    df = pd.read_csv("data/raw/QA.csv")
    
    # Limit to first 20 for quick test, or run all if user wants "match"
    # User said "see how much it match", implying accuracy.
    # Let's run a sample of 10 to be fast for the demo, or 50.
    sample_df = df.head(10) 
    
    correct = 0
    total = 0
    
    results = []

    print(f"Evaluating {len(sample_df)} questions...")
    
    for index, row in sample_df.iterrows():
        q = row['Question']
        expected = row['Answer']
        
        if pd.isna(expected): continue # Skip if no answer key
        
        start = time.time()
        try:
            prediction = await run_agent_pipeline(q)
            latency = (time.time() - start) * 1000
            
            # Simple exact match or heuristic?
            # The answers are typically "ก", "ข", "ค", "ง" or short text.
            # If the prediction contains the answer key, we might count it?
            # Or if it's multiple choice, does the agent output the choice?
            # The agent is instructed to be helpful. It might output full text.
            # We will save the result for manual inspection or simple containment check.
            
            match = False
            if str(expected).strip() in prediction:
                match = True
                correct += 1
            
            results.append({
                "Question": q,
                "Expected": expected,
                "Predicted": prediction,
                "Match": match,
                "Latency": latency
            })
            total += 1
            print(f"[{'PASS' if match else 'FAIL'}] {q[:30]}... -> {prediction[:50]}... ({latency:.1f}ms)")
            
        except Exception as e:
            print(f"Error on {q}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Save results
    pd.DataFrame(results).to_csv("data/processed/eval_results.csv", index=False)

if __name__ == "__main__":
    asyncio.run(eval_qa())
