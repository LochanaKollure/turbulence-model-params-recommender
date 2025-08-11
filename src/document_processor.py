import requests
import PyPDF2
import io
from typing import List, Dict, Any
from urllib.parse import urlparse
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text content from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'application/pdf' in content_type:
                return self._extract_text_from_pdf_bytes(response.content)
            else:
                # Assume HTML content and extract basic text
                return self._extract_text_from_html(response.text)
                
        except Exception as e:
            raise Exception(f"Failed to extract text from URL {url}: {str(e)}")
    
    def extract_text_from_pdf_file(self, pdf_path: str) -> str:
        """Extract text from a local PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                return self._extract_text_from_pdf_bytes(file.read())
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
    
    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return self._clean_text(text)
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF bytes: {str(e)}")
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Basic HTML text extraction."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        # Replace HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            })
            
            result.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        return result
    
    def process_document(self, source: str, doc_type: str = "auto") -> List[Dict[str, Any]]:
        """Process a document from URL or file path."""
        metadata = {
            'source': source,
            'type': doc_type
        }
        
        if doc_type == "auto":
            # Auto-detect type
            if source.startswith(('http://', 'https://')):
                doc_type = "url"
            elif source.endswith('.pdf'):
                doc_type = "pdf"
            else:
                doc_type = "text"
        
        if doc_type == "url":
            text = self.extract_text_from_url(source)
            metadata['type'] = "url"
        elif doc_type == "pdf":
            text = self.extract_text_from_pdf_file(source)
            metadata['type'] = "pdf"
        else:
            with open(source, 'r', encoding='utf-8') as f:
                text = f.read()
            metadata['type'] = "text"
        
        return self.chunk_text(text, metadata)