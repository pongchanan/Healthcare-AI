# Healthcare-AI System 🏥

A high-performance, privacy-focused AI agent designed for medical contexts. This system is built to provide accurate answers by combining structured data (SQL), unstructured knowledge (Medical PDFs/Vectors), and real-time patient APIs, all while targeting <50ms decision latency.

---

## 🚀 How It Works (The "Mental Model")

Think of this AI as a **Smart Doctor's Assistant** that has three superpowers:
1.  **Memory:** It can read and recall thousands of medical textbooks instantly.
2.  **Access:** It can look up live patient records securely.
3.  **Synthesis:** It can reason like a human to combine facts into a helpful answer.

### The Workflow: From Question to Answer

When you ask a question (e.g., *"Does patient John Doe have the flu?"*), the system follows this journey:

1.  **The Ear (Entry Point):**
    Your question enters the system via the API.

2.  **The Brain (The Router):**
    Before thinking hard, a fast "Router" (like a triage nurse) decides what *kind* of problem this is:
    - **Is it a fact?** (e.g., "What are symptoms of Dengue?") → Uses **Knowledge Search**.
    - **Is it a stat?** (e.g., "How many patients are sick?") → Uses **SQL Calculator**.
    - **Is it specific?** (e.g., "Check Jane's blood pressure") → Uses **Patient API**.
    - **Is it complex?** (e.g., "Based on Jane's history, is she at risk?") → Uses **Hybrid Mode** (All of the above).

3.  **The Hands (Tools):**
    Based on the decision, the system uses specific tools:
    - **Vector Search (Qdrant):** Only looks for "meaning" in text. It knows that "H1N1" and "Flu" are related.
    - **SQL Database (DuckDB):** Lightning-fast data crunching for tables and CSVs.
    - **API Wrapper:** Securely fetches real-time data.

4.  **The Voice (Typhoon Synthesizer):**
    Finally, all the gathered clues are sent to a robust LLM (Large Language Model) that speaks fluent Thai and English. It writes the final answer for you.

---

## 🏗 System Architecture

The system is split into two halves to ensure **Speed** during the day and **Learning** at night.

### 1. Offline Layer ("The Learning Phase")
*This runs in the background to prepare data.*
*   **Ingestion Pipeline:**
    *   Reads **PDFs & Images** -> Converts them into mathematical "Vectors" -> Stores in **Qdrant**.
    *   Reads **CSVs** -> Automatically detects schema -> Loads into **DuckDB**.
*   **Why?** By processing heavy data beforehand, we don't make the user wait when they ask a question.

### 2. Runtime Layer ("The Action Phase")
*This runs when you ask a question.*
*   **Tech Stack:** FastAPI (Server), Asynchronous Python (Speed), Ollama (Local AI).
*   **Desing:**
    *   **No Mocks:** Every tool connects to a real engine.
    *   **Parallelism:** If the AI needs to check the API and the Database, it does both *at the same time* to save milliseconds.

---

## 📂 Project Structure

- **`src/`**
    - **`pipelines/`**: The offline workers (Ingestion, SQL Loading).
    - **`agent/`**: The brain (Router, Workflow logic).
    - **`tools/`**: The hands (Database connectors, API wrappers).
    - **`config.py`**: The central settings control.
- **`data/`**
    - **`raw/`**: Drop your PDFs, CSVs, and Images here.
    - **`processed/`**: The system stores optimized DB files here.
    - **`vector_store/`**: The AI's long-term memory.
- **`main.py`**: The web server launchpad.

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the `typhoon` models.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place your medical PDFs and CSVs (e.g., `patients.csv`, `guidelines.pdf`) into `data/raw/`.

### 3. Build the Brain (Ingestion)
Run these once to teach the AI your data:
```bash
# Load structured data (CSVs)
python -m src.pipelines.sql_loader

# Process unstructured data (PDFs/Vectors)
python -m src.pipelines.ingestion
```

### 4. Start the Agent
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Evaluate
Run the test script to see how smart the AI is:
```bash
python eval.py
```

---

## 🛡 Design Philosophy
- **Local-First:** Designed to run sensitive medical data on-premise without sending it to the cloud.
- **Speed-Obsessed:** Every millisecond counts in a clinical setting.
- **Explainable:** We use "Chain of Thought" routing so we know *why* the AI made a decision.
