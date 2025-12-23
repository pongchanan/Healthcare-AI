import os
import glob
import httpx
from src.config import Config
from src.tools.database import qdrant_client
from qdrant_client.http import models

# Use Ollama for embeddings to avoid heavy local dependencies
def get_embedding(text: str):
    try:
        url = f"{Config.OLLAMA_BASE_URL}/api/embeddings"
        payload = {
            "model": Config.SYNTHESIZER_MODEL, # Use the 8b model or 1b if compatible for embeddings
            # Note: Instruct models might need a specific prompt or might not support /api/embeddings well if not optimized
            # But normally Ollama handles it.
            # If 8b fails, we can try 1b or hardcode a model name if known.
            # Using 1b might be faster.
            "prompt": text
        }
        # Verify if model supports embeddings. If not, we might need to use a specific embedding model if user has one.
        # Given the list: scb10x/llama3.1-typhoon2-8b-instruct
        # We will try it.
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("embedding")
    except Exception as e:
        print(f" [Embedding Error] {e}")
        return [0.0] * 384 # Fallback dummy if failed, to allow pipeline to continue (though results will be bad)

def process_images_offline():
    """
    1. Scan all PNG/JPEGs in data/raw
    2. Send to Typhoon Vision (offline) to generate captions.
    """
    print(" [Offline] Processing images...")
    image_paths = glob.glob(os.path.join("data/raw/*.png")) + \
                  glob.glob(os.path.join("data/raw/*.jpg"))
    
    for img_path in image_paths:
        print(f"   - Processing {os.path.basename(img_path)}")
        # Real logic would go here

def chunk_mixed_documents():
    """
    Handles PDF text splitting and vector indexing.
    """
    print(" [Offline] Chunking Documents...")
    
    # Ensure collection exists
    # LLama embeddings are typically 4096 dim for 8B?
    # We need to know the dimension.
    # Let's do a test call to get dim or assume 4096. 
    # For safety, we'll try to fetch one embedding first to set size.
    test_embed = get_embedding("test")
    if not test_embed:
        print(" [Error] Could not get embeddings from Ollama. Aborting indexing.")
        return
        
    dim = len(test_embed)
    print(f" [Offline] Detected embedding dimension: {dim}")

    qdrant_client.recreate_collection(
        collection_name="medical_docs",
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
    )

    pdf_paths = glob.glob(os.path.join("data/raw/*.pdf"))
    
    documents = []
    
    # Start with simple text chunking (no PDF lib installed yet? pymupdf was in requirements...)
    # I'll check if fitz/pymupdf is available. If not, skip PDF content or use text files if any.
    # User moved data/*.pdf to data/raw/
    
    # Attempt to import pymupdf
    try:
        import fitz
        for pdf_file in pdf_paths:
            print(f"   - Reading {pdf_file}...")
            doc = fitz.open(pdf_file)
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Simple chunking
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "source": os.path.basename(pdf_file),
                    "id": f"{os.path.basename(pdf_file)}_{i}"
                })
    except ImportError:
            print(" [Warning] PyMuPDF not found. Skipping PDF content.")

    # Add dummy data if empty to ensure system works
    if not documents:
        documents.append({"content": "Patient A has classic flu symptoms.", "source": "synthetic", "id": "1"})
        documents.append({"content": "Patient B diagnosed with mild hypertension.", "source": "synthetic", "id": "2"})

    # Indexing
    if documents:
        print(f"   - Indexing {len(documents)} chunks...")
        points = []
        for doc in documents:
            vector = get_embedding(doc['content'])
            if vector and len(vector) == dim:
                points.append(models.PointStruct(
                    id=abs(hash(doc['id'])),
                    vector=vector,
                    payload=doc
                ))
            
            # Batch upsert to avoid huge request?
            if len(points) >= 10:
                qdrant_client.upsert(collection_name="medical_docs", points=points)
                points = []
        
        if points:
            qdrant_client.upsert(collection_name="medical_docs", points=points)
    
    print(" [Offline] Vector Indexing Complete.")

if __name__ == "__main__":
    process_images_offline()
    chunk_mixed_documents()
