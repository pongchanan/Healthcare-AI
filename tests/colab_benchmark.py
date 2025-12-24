import csv
import time
import requests
import statistics
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000/api/ask"
DATA_PATH = "data/raw/QA.csv"
MAX_QUESTIONS = 50  # Test first 50 for speed, set to None for all

def test_question(row):
    question = row['Question']
    correct_answer = row['Answer'].strip()
    
    # Skip if no answer key provided
    if not correct_answer:
        return None

    start_ts = time.time()
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=10)
        latency = (time.time() - start_ts) * 1000
        
        if response.status_code == 200:
            result = response.json()
            model_answer = result.get('answer', '').strip()
            server_latency = result.get('latency_ms', 0)
            
            # Check correctness (Exact match of ก,ข,ค,ง)
            is_correct = model_answer == correct_answer
            
            return {
                "question": question[:30] + "...",
                "expected": correct_answer,
                "got": model_answer,
                "correct": is_correct,
                "latency": latency,
                "server_latency": server_latency
            }
        else:
            return {"error": f"Status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def run_benchmark():
    print(f"Loading data from {DATA_PATH}...")
    questions = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Answer'].strip(): # Only valid rows
                questions.append(row)
    
    print(f"Found {len(questions)} valid questions. Testing {MAX_QUESTIONS if MAX_QUESTIONS else 'all'}...")
    
    target_questions = questions[:MAX_QUESTIONS] if MAX_QUESTIONS else questions
    results = []
    
    # Run sequentially to measure true latency (parallel would stress test throughput, not single-req latency)
    for i, q in enumerate(target_questions):
        res = test_question(q)
        if res and "error" not in res:
            results.append(res)
            status = "✅" if res['correct'] else "❌"
            print(f"[{i+1}/{len(target_questions)}] {status} Latency: {res['latency']:.0f}ms | Exp: {res['expected']} | Got: {res['got']}")
        elif res:
            print(f"[{i+1}/{len(target_questions)}] ⚠️ Error: {res['error']}")

    if not results:
        print("No results to analyze.")
        return

    # metrics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total) * 100
    avg_latency = statistics.mean(r['latency'] for r in results)
    
    print("\n" + "="*30)
    print(" 📊 BENCHMARK REPORT")
    print("="*30)
    print(f"Total Questions: {total}")
    print(f"Accuracy:        {accuracy:.2f}%")
    print(f"Avg Latency:     {avg_latency:.2f} ms")
    print("="*30)

if __name__ == "__main__":
    run_benchmark()
