from typing import Dict, Any, Optional, List
from src.retrieval_system import RetrievalSystem
from src.parameter_generator import ParameterGenerator
from src.turbulence_models import get_model_names, get_model
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG pipeline that orchestrates document retrieval and parameter generation
    for turbulence model recommendations.
    """
    
    def __init__(self):
        self.retrieval_system = RetrievalSystem()
        self.parameter_generator = ParameterGenerator()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the RAG pipeline components."""
        try:
            logger.info("Initializing RAG pipeline...")
            print("🔧 Initializing RAG pipeline components...")
            
            # Initialize retrieval system
            if not self.retrieval_system.initialize():
                logger.error("Failed to initialize retrieval system")
                print("❌ Retrieval system initialization failed")
                return False
            
            self.initialized = True
            logger.info("RAG pipeline initialized successfully")
            print("✅ RAG pipeline components ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            print(f"❌ Pipeline initialization error: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available turbulence models."""
        return get_model_names()
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if the turbulence model is supported."""
        return model_name in get_model_names()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific turbulence model."""
        model = get_model(model_name)
        if not model:
            return None
        
        return {
            'name': model.name,
            'full_name': model.full_name,
            'description': model.description,
            'category': model.category,
            'parameters': [
                {
                    'name': p.name,
                    'description': p.description,
                    'type': p.parameter_type.value,
                    'default_value': p.default_value,
                    'typical_range': p.typical_range,
                    'units': p.units
                } for p in model.parameters
            ],
            'applications': model.applications,
            'limitations': model.limitations
        }
    
    def generate_recommendations(self,
                               turbulence_model: str,
                               user_description: str = "",
                               focus_area: str = "",
                               include_debug_info: bool = False) -> Dict[str, Any]:
        """
        Generate parameter recommendations for a turbulence model.
        
        Args:
            turbulence_model: Name of the turbulence model
            user_description: User's description of the flow case/application
            focus_area: Specific focus area or constraint
            include_debug_info: Whether to include debugging information
            
        Returns:
            Dictionary containing parameter recommendations and metadata
        """
        
        if not self.initialized:
            logger.error("RAG pipeline not initialized")
            return self._create_error_response("Pipeline not initialized")
        
        # Validate model
        if not self.validate_model(turbulence_model):
            return self._create_error_response(f"Unsupported turbulence model: {turbulence_model}")
        
        try:
            logger.info(f"Generating recommendations for {turbulence_model}")
            print(f"🎨 Starting parameter generation for {turbulence_model}")
            
            # Step 1: Retrieve relevant documents
            logger.info("Retrieving relevant documents...")
            print("🔍 Searching knowledge base...")
            context_info = self.retrieval_system.get_context_for_generation(
                turbulence_model=turbulence_model,
                user_description=user_description,
                focus_area=focus_area,
                max_context_length=4000
            )
            
            if context_info['total_documents'] == 0:
                logger.warning("No relevant documents found")
                print("⚠️ No matching documents found in knowledge base")
                # Continue anyway with empty context
            else:
                logger.info(f"Retrieved context from {context_info['total_documents']} documents")
                avg_relevance = sum(context_info['relevance_scores']) / len(context_info['relevance_scores']) if context_info['relevance_scores'] else 0
                print(f"📚 Retrieved {context_info['total_documents']} documents (avg relevance: {avg_relevance:.3f})")
            
            # Step 2: Generate parameter recommendations
            logger.info("Generating parameter recommendations...")
            print("🤖 AI generating parameter recommendations...")
            recommendations = self.parameter_generator.generate_parameters(
                turbulence_model=turbulence_model,
                user_description=user_description,
                context_text=context_info['context_text'],
                focus_area=focus_area
            )
            
            # Step 3: Compile final response
            response = {
                'success': True,
                'model_name': turbulence_model,
                'recommendations': recommendations,
                'retrieval_info': {
                    'documents_used': context_info['total_documents'],
                    'sources': context_info['sources'],
                    'average_relevance': (
                        sum(context_info['relevance_scores']) / len(context_info['relevance_scores'])
                        if context_info['relevance_scores'] else 0.0
                    )
                }
            }
            
            # Add debug information if requested
            if include_debug_info:
                response['debug_info'] = {
                    'context_length': len(context_info['context_text']),
                    'relevance_scores': context_info['relevance_scores'],
                    'retrieval_query': self.retrieval_system.construct_search_query(
                        turbulence_model, user_description, focus_area
                    )
                }
            
            logger.info("Successfully generated recommendations")
            print("✨ Parameter recommendations generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            print(f"❌ Recommendation generation failed: {str(e)}")
            return self._create_error_response(str(e))
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of the RAG pipeline components."""
        
        retrieval_status = self.retrieval_system.get_system_status()
        
        return {
            'initialized': self.initialized,
            'retrieval_system': retrieval_status,
            'available_models': len(get_model_names()),
            'supported_models': get_model_names()
        }
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents directly (for debugging/exploration)."""
        
        if not self.initialized:
            logger.error("RAG pipeline not initialized")
            return []
        
        try:
            results = self.retrieval_system.vector_store.query_similar_documents(query, top_k=top_k)
            print(f"🔎 Document search completed: {len(results)} results for '{query[:30]}{'...' if len(query) > 30 else ''}'")
            return results
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            print(f"❌ Document search failed: {str(e)}")
            return []
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'model_name': '',
            'recommendations': {
                'error': True,
                'error_message': error_message,
                'parameters': {},
                'overall_confidence': 0.0,
                'key_considerations': [],
                'sensitivity_warnings': [],
                'validation_recommendations': []
            },
            'retrieval_info': {
                'documents_used': 0,
                'sources': [],
                'average_relevance': 0.0
            }
        }
    
    def format_recommendations_summary(self, response: Dict[str, Any]) -> str:
        """Format recommendations for quick summary display."""
        
        if not response.get('success'):
            return f"❌ Error: {response.get('error', 'Unknown error')}"
        
        recommendations = response.get('recommendations', {})
        if recommendations.get('error'):
            return f"❌ Generation Error: {recommendations.get('error_message', 'Unknown error')}"
        
        model_name = response.get('model_name', 'Unknown')
        confidence = recommendations.get('overall_confidence', 0)
        param_count = len(recommendations.get('parameters', {}))
        doc_count = response.get('retrieval_info', {}).get('documents_used', 0)
        
        return f"✅ Generated {param_count} parameters for {model_name} (Confidence: {confidence:.2f}, Documents: {doc_count})"
    
    def export_recommendations(self, response: Dict[str, Any], format_type: str = "json") -> str:
        """Export recommendations in various formats."""
        
        if format_type.lower() == "json":
            import json
            return json.dumps(response, indent=2)
        
        elif format_type.lower() == "markdown":
            return self.parameter_generator.format_parameters_for_display(
                response.get('recommendations', {})
            )
        
        elif format_type.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['Parameter', 'Value', 'Confidence', 'Rationale'])
            
            # Parameters
            for param_name, param_data in response.get('recommendations', {}).get('parameters', {}).items():
                writer.writerow([
                    param_name,
                    param_data.get('value', ''),
                    param_data.get('confidence', ''),
                    param_data.get('rationale', '')
                ])
            
            return output.getvalue()
        
        else:
            return "Unsupported format type"