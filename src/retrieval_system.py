from typing import List, Dict, Any, Optional
from src.vector_store import VectorStore
from src.turbulence_models import get_model, TurbulenceModel
import logging

logger = logging.getLogger(__name__)

class RetrievalSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.connected = False
    
    def initialize(self) -> bool:
        """Initialize connection to vector store."""
        try:
            if self.vector_store.connect_to_index():
                self.connected = True
                logger.info("Retrieval system initialized successfully")
                return True
            else:
                logger.error("Failed to connect to vector store")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize retrieval system: {str(e)}")
            return False
    
    def construct_search_query(self, 
                             turbulence_model: str, 
                             user_description: str = "",
                             focus_area: str = "") -> str:
        """Construct an optimized search query for document retrieval."""
        
        # Get model information
        model_info = get_model(turbulence_model)
        if not model_info:
            return f"{turbulence_model} turbulence model parameters {user_description}"
        
        # Build comprehensive query
        query_parts = []
        
        # Add model name variations
        query_parts.append(model_info.full_name)
        query_parts.append(model_info.name.replace("_", "-"))
        
        # Add model category and type
        query_parts.append(f"{model_info.category} turbulence model")
        
        # Add parameter-specific terms
        param_terms = []
        for param in model_info.parameters:
            param_terms.append(param.name)
            if param.description:
                # Extract key terms from description
                desc_words = param.description.lower().split()
                key_words = [word for word in desc_words if len(word) > 4 and word.isalpha()]
                param_terms.extend(key_words[:2])  # Take first 2 key words
        
        query_parts.extend(param_terms[:10])  # Limit parameter terms
        
        # Add application context if available
        if model_info.applications:
            query_parts.extend(model_info.applications[:3])  # Top 3 applications
        
        # Add user description
        if user_description.strip():
            query_parts.append(user_description.strip())
        
        # Add focus area
        if focus_area.strip():
            query_parts.append(focus_area.strip())
        
        # Combine and optimize query
        query = " ".join(query_parts)
        
        # Clean up the query
        query = query.replace("_", " ")
        query = " ".join(query.split())  # Remove extra spaces
        
        return query[:500]  # Limit query length
    
    def retrieve_relevant_documents(self, 
                                  turbulence_model: str,
                                  user_description: str = "",
                                  focus_area: str = "",
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to the specified turbulence model and context."""
        
        if not self.connected:
            logger.error("Retrieval system not initialized")
            return []
        
        try:
            # Construct optimized search query
            search_query = self.construct_search_query(
                turbulence_model, user_description, focus_area
            )
            
            logger.info(f"Searching with query: {search_query[:100]}...")
            
            # Retrieve documents
            results = self.vector_store.query_similar_documents(search_query, top_k=top_k)
            
            # Enhance results with additional context
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    'id': result['id'],
                    'relevance_score': result['score'],
                    'content_preview': result['text_preview'],
                    'source': result['metadata'].get('source', 'Unknown'),
                    'document_type': result['metadata'].get('type', 'Unknown'),
                    'chunk_info': {
                        'index': result['metadata'].get('chunk_index', 0),
                        'total': result['metadata'].get('total_chunks', 1)
                    }
                }
                enhanced_results.append(enhanced_result)
            
            logger.info(f"Retrieved {len(enhanced_results)} relevant documents")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []
    
    def get_context_for_generation(self, 
                                 turbulence_model: str,
                                 user_description: str = "",
                                 focus_area: str = "",
                                 max_context_length: int = 4000) -> Dict[str, Any]:
        """Get formatted context for LLM parameter generation."""
        
        # Retrieve relevant documents
        documents = self.retrieve_relevant_documents(
            turbulence_model, user_description, focus_area, top_k=8
        )
        
        if not documents:
            logger.warning("No relevant documents found")
            return {
                'context_text': "",
                'sources': [],
                'relevance_scores': [],
                'total_documents': 0
            }
        
        # Format context text
        context_parts = []
        sources = []
        relevance_scores = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Add document header
            header = f"\n--- Document {i+1} (Relevance: {doc['relevance_score']:.3f}) ---\n"
            content = doc['content_preview']
            
            # Check if adding this document would exceed limit
            if current_length + len(header) + len(content) > max_context_length:
                if current_length == 0:  # If first document is too long, truncate it
                    available_space = max_context_length - len(header) - 50  # Leave space for truncation note
                    content = content[:available_space] + "...[truncated]"
                    context_parts.append(header + content)
                    sources.append(doc['source'])
                    relevance_scores.append(doc['relevance_score'])
                break
            
            context_parts.append(header + content)
            sources.append(doc['source'])
            relevance_scores.append(doc['relevance_score'])
            current_length += len(header) + len(content)
        
        context_text = "".join(context_parts)
        
        return {
            'context_text': context_text,
            'sources': sources,
            'relevance_scores': relevance_scores,
            'total_documents': len(sources)
        }
    
    def expand_query_with_synonyms(self, base_query: str) -> str:
        """Expand query with turbulence modeling synonyms and related terms."""
        
        # Turbulence modeling synonyms and expansions
        expansions = {
            'k-epsilon': ['k-ε', 'k epsilon', 'k_epsilon', 'kinetic energy dissipation'],
            'k-omega': ['k-ω', 'k omega', 'k_omega', 'kinetic energy frequency'],
            'SST': ['Shear Stress Transport', 'Menter SST', 'k-omega SST'],
            'spalart': ['Spalart-Allmaras', 'SA model', 'one equation model'],
            'reynolds': ['Reynolds stress', 'RSM', 'second moment closure'],
            'viscosity': ['turbulent viscosity', 'eddy viscosity', 'kinematic viscosity'],
            'dissipation': ['energy dissipation', 'epsilon', 'turbulent dissipation'],
            'production': ['turbulence production', 'kinetic energy production'],
            'constants': ['coefficients', 'parameters', 'closure constants'],
            'wall': ['near-wall', 'wall treatment', 'wall function'],
            'boundary': ['boundary conditions', 'BC', 'wall BC'],
            'separation': ['flow separation', 'separated flow', 'adverse gradient'],
            'pressure': ['pressure gradient', 'adverse pressure gradient', 'APG']
        }
        
        query_lower = base_query.lower()
        expanded_terms = []
        
        for term, synonyms in expansions.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Add top 2 synonyms
        
        if expanded_terms:
            return base_query + " " + " ".join(expanded_terms)
        
        return base_query
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the retrieval system."""
        status = {
            'connected': self.connected,
            'index_stats': {}
        }
        
        if self.connected:
            status['index_stats'] = self.vector_store.get_index_stats()
        
        return status