import streamlit as st
import sys
import os
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline
from src.turbulence_models import get_model_names, get_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Turbulence Model Parameter Recommender",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .parameter-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .confidence-high { border-left-color: #28a745; }
    .confidence-medium { border-left-color: #ffc107; }
    .confidence-low { border-left-color: #dc3545; }
    
    .source-item {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_pipeline():
    """Initialize the RAG pipeline with caching."""
    print("🚀 Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    if pipeline.initialize():
        print("✅ RAG pipeline initialized successfully")
        return pipeline
    else:
        print("❌ Failed to initialize RAG pipeline")
        st.error("Failed to initialize RAG pipeline. Check your API keys and Pinecone configuration.")
        return None

def display_model_info(model_name):
    """Display information about the selected turbulence model."""
    model = get_model(model_name)
    if not model:
        return
    
    with st.expander("ℹ️ Model Information", expanded=False):
        st.markdown(f"""
        **Full Name**: {model.full_name}
        
        **Description**: {model.description}
        
        **Category**: {model.category}
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Applications:**")
            for app in model.applications:
                st.markdown(f"• {app}")
        
        with col2:
            st.markdown("**Limitations:**")
            for lim in model.limitations:
                st.markdown(f"• {lim}")
        
        st.markdown("**Parameters:**")
        for param in model.parameters:
            st.markdown(f"""
            - **{param.name}**: {param.description}
              - Default: {param.default_value or 'varies'} | Range: {param.typical_range or 'model-dependent'}
            """)

def display_recommendations(response):
    """Display parameter recommendations in a formatted way."""
    if not response.get('success'):
        st.error(f"Error: {response.get('error', 'Unknown error')}")
        return
    
    recommendations = response.get('recommendations', {})
    
    if recommendations.get('error'):
        st.error(f"Generation Error: {recommendations.get('error_message', 'Unknown error')}")
        return
    
    # Overall confidence
    overall_confidence = recommendations.get('overall_confidence', 0)
    confidence_color = "🟢" if overall_confidence >= 0.8 else "🟡" if overall_confidence >= 0.6 else "🔴"
    
    st.markdown(f"""
    ### 🎯 Parameter Recommendations
    **Overall Confidence**: {confidence_color} {overall_confidence:.2f}
    """)
    
    # Parameters
    parameters = recommendations.get('parameters', {})
    if parameters:
        st.markdown("#### Parameters")
        
        for param_name, param_data in parameters.items():
            confidence = param_data.get('confidence', 0)
            css_class = "confidence-high" if confidence >= 0.8 else "confidence-medium" if confidence >= 0.6 else "confidence-low"
            
            st.markdown(f"""
            <div class="parameter-card {css_class}">
                <h4>{param_name}</h4>
                <p><strong>Recommended Value:</strong> {param_data.get('value', 'N/A')}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                <p><strong>Rationale:</strong> {param_data.get('rationale', 'No rationale provided')}</p>
                {f'<p><strong>⚠️ Warning:</strong> {param_data["validation_warning"]}</p>' if 'validation_warning' in param_data else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Key considerations
    considerations = recommendations.get('key_considerations', [])
    if considerations:
        st.markdown("#### 🔍 Key Considerations")
        for consideration in considerations:
            st.markdown(f"• {consideration}")
    
    # Sensitivity warnings
    warnings = recommendations.get('sensitivity_warnings', [])
    if warnings:
        st.markdown("#### ⚠️ Sensitivity Warnings")
        for warning in warnings:
            st.warning(warning)
    
    # Validation recommendations
    validations = recommendations.get('validation_recommendations', [])
    if validations:
        st.markdown("#### ✅ Validation Recommendations")
        for validation in validations:
            st.markdown(f"• {validation}")

def display_sources(response):
    """Display document sources used for recommendations."""
    retrieval_info = response.get('retrieval_info', {})
    sources = retrieval_info.get('sources', [])
    
    if sources:
        st.markdown("#### 📚 Sources Used")
        st.markdown(f"**Documents Retrieved**: {retrieval_info.get('documents_used', 0)}")
        st.markdown(f"**Average Relevance**: {retrieval_info.get('average_relevance', 0):.3f}")
        
        for i, source in enumerate(sources):
            st.markdown(f"""
            <div class="source-item">
                <strong>Source {i+1}:</strong> {source}
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🌪️ Turbulence Model Parameter Recommender</h1>', unsafe_allow_html=True)
    st.markdown("Generate expert parameter recommendations for CFD turbulence models using RAG-enhanced AI")
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    if not pipeline:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = pipeline.get_available_models()
        model_display_names = {
            'k_epsilon': 'k-ε (Standard)',
            'k_omega_sst': 'k-ω SST',
            'spalart_allmaras': 'Spalart-Allmaras',
            'reynolds_stress': 'Reynolds Stress Model'
        }
        
        selected_model = st.selectbox(
            "Select Turbulence Model",
            available_models,
            format_func=lambda x: model_display_names.get(x, x),
            index=0
        )
        
        # Log model selection
        if 'previous_model' not in st.session_state or st.session_state.previous_model != selected_model:
            print(f"🔄 User selected turbulence model: {model_display_names.get(selected_model, selected_model)}")
            st.session_state.previous_model = selected_model
        
        # Advanced options
        st.subheader("Advanced Options")
        include_debug = st.checkbox("Include Debug Information", value=False)
        export_format = st.selectbox("Export Format", ["JSON", "Markdown", "CSV"])
        
        # Pipeline status
        st.subheader("System Status")
        status = pipeline.get_pipeline_status()
        if status['initialized']:
            st.success("✅ Pipeline Ready")
            st.info(f"Models Available: {status['available_models']}")
            
            # Index stats
            retrieval_status = status.get('retrieval_system', {})
            if retrieval_status.get('connected'):
                index_stats = retrieval_status.get('index_stats', {})
                st.info(f"Documents in DB: {index_stats.get('total_vector_count', 0)}")
        else:
            st.error("❌ Pipeline Not Ready")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display selected model info
        display_model_info(selected_model)
        
        # Input form
        st.subheader("📝 Application Details")
        
        with st.form("parameter_form"):
            user_description = st.text_area(
                "Describe your CFD application, flow conditions, and geometry:",
                placeholder="e.g., External flow over a wing at high Reynolds number with potential separation...",
                height=100
            )
            
            focus_area = st.text_input(
                "Specific focus area or constraints (optional):",
                placeholder="e.g., accurate separation prediction, wall heat transfer..."
            )
            
            generate_button = st.form_submit_button("🚀 Generate Recommendations", type="primary")
        
        # Generate recommendations
        if generate_button:
            print(f"🎯 Generating recommendations for {model_display_names.get(selected_model, selected_model)}")
            print(f"📝 User description length: {len(user_description)} characters")
            if focus_area:
                print(f"🎪 Focus area: {focus_area[:50]}{'...' if len(focus_area) > 50 else ''}")
            
            with st.spinner("Generating parameter recommendations..."):
                response = pipeline.generate_recommendations(
                    turbulence_model=selected_model,
                    user_description=user_description,
                    focus_area=focus_area,
                    include_debug_info=include_debug
                )
                
                # Log generation result
                if response.get('success'):
                    recommendations = response.get('recommendations', {})
                    param_count = len(recommendations.get('parameters', {}))
                    confidence = recommendations.get('overall_confidence', 0)
                    docs_used = response.get('retrieval_info', {}).get('documents_used', 0)
                    print(f"✅ Generated {param_count} parameters (confidence: {confidence:.2f}, docs: {docs_used})")
                else:
                    print(f"❌ Generation failed: {response.get('error', 'Unknown error')}")
                
                # Store in session state for export
                st.session_state['last_response'] = response
                
                # Display results
                display_recommendations(response)
    
    with col2:
        st.subheader("📊 Results Summary")
        
        # Display sources if available
        if 'last_response' in st.session_state:
            response = st.session_state['last_response']
            
            # Quick summary
            summary = pipeline.format_recommendations_summary(response)
            if summary.startswith("✅"):
                st.success(summary)
            else:
                st.error(summary)
            
            # Sources
            display_sources(response)
            
            # Export options
            st.subheader("📤 Export")
            if st.button("Export Results"):
                print(f"📤 Exporting results in {export_format} format for {selected_model}")
                exported_data = pipeline.export_recommendations(response, export_format.lower())
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"turbulence_params_{selected_model}.{export_format.lower()}",
                    mime="application/json" if export_format.lower() == "json" else "text/plain"
                )
                print(f"✅ Export prepared: {len(exported_data)} characters")
            
            # Debug info
            if include_debug and response.get('debug_info'):
                st.subheader("🐛 Debug Information")
                with st.expander("Debug Details"):
                    st.json(response['debug_info'])
        
        # Document search
        st.subheader("🔍 Document Search")
        search_query = st.text_input("Search documents:")
        if search_query:
            print(f"🔍 Document search query: {search_query}")
            with st.spinner("Searching..."):
                search_results = pipeline.search_documents(search_query, top_k=3)
                print(f"📚 Found {len(search_results)} search results")
                
                for i, result in enumerate(search_results):
                    st.markdown(f"""
                    **Result {i+1}** (Score: {result['score']:.3f})  
                    Source: {result['metadata'].get('source', 'Unknown')}  
                    {result['text_preview'][:200]}...
                    """)

if __name__ == "__main__":
    main()