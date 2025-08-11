import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import settings
import time
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self.dimension = settings.embedding_dimension
        self.index = None
        
        # Initialize OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        
    def create_index(self) -> bool:
        """Create Pinecone index if it doesn't exist."""
        try:
            # Check if index already exists
            if self.index_name in self.pc.list_indexes().names():
                logger.info(f"Index {self.index_name} already exists")
                self.index = self.pc.Index(self.index_name)
                return True
            
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Created index {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def connect_to_index(self) -> bool:
        """Connect to existing Pinecone index."""
        try:
            if self.index_name not in self.pc.list_indexes().names():
                logger.error(f"Index {self.index_name} does not exist")
                return False
                
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to index: {str(e)}")
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI's text-embedding-ada-002."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        try:
            if not self.index:
                logger.error("Index not connected")
                return False
            
            vectors = []
            for i, doc in enumerate(documents):
                # Generate embedding for document text
                embedding = self.generate_embedding(doc['text'])
                
                # Create unique ID
                doc_id = f"doc_{int(time.time())}_{i}"
                
                # Prepare metadata (Pinecone has metadata size limits)
                metadata = {
                    'source': doc['metadata'].get('source', '')[:500],  # Truncate if too long
                    'type': doc['metadata'].get('type', ''),
                    'chunk_index': doc['metadata'].get('chunk_index', 0),
                    'total_chunks': doc['metadata'].get('total_chunks', 1),
                    'text_preview': doc['text'][:200]  # First 200 chars for preview
                }
                
                vectors.append({
                    'id': doc_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            return False
    
    def query_similar_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query for similar documents."""
        try:
            if not self.index:
                logger.error("Index not connected")
                return []
            
            # Generate query embedding
            query_embedding = self.generate_embedding(query_text)
            
            # Query Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            for match in response['matches']:
                results.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata'],
                    'text_preview': match['metadata'].get('text_preview', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query documents: {str(e)}")
            return []
    
    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get full text of a document by ID."""
        try:
            # In a real implementation, you might want to store full text separately
            # For now, we'll return the preview text from metadata
            response = self.index.fetch(ids=[doc_id])
            if doc_id in response['vectors']:
                return response['vectors'][doc_id]['metadata'].get('text_preview', '')
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document text: {str(e)}")
            return None
    
    def delete_index(self) -> bool:
        """Delete the entire index (use with caution)."""
        try:
            self.pc.delete_index(self.index_name)
            self.index = None
            logger.info(f"Deleted index {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            if not self.index:
                return {}
            
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats.get('total_vector_count', 0),
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}