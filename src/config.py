import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Runtime Models (Optimized for Speed)
    # Using environment variables or defaults provided in the prompt
    ROUTER_MODEL = os.getenv("ROUTER_MODEL", "scb10x/llama3.2-typhoon2-1b-instruct:latest") # Tiny, fast decision maker
    SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "scb10x/llama3.1-typhoon2-8b-instruct:latest") # General knowledge & Thai fluency
    
    # Offline Models (Optimized for Accuracy)
    VISION_MODEL = os.getenv("VISION_MODEL", "typhoon-v2-vision") 
    
    # Database Paths
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")
    SQL_DB_PATH = os.getenv("SQL_DB_PATH", "./data/processed/medical_data.duckdb")
    
    # Caching
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour
    
    # API Settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # BAD_API_ENDPOINT should be a real endpoint if available, or handled gracefully
    BAD_API_ENDPOINT = os.getenv("BAD_API_ENDPOINT", "http://localhost:8080/api/v1") 
    TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "EMPTY")
