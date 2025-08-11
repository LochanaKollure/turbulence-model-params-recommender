import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "turbulence-docs")
    
    embedding_dimension: int = 1536
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()