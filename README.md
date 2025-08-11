# Turbulence Model Parameter Recommender

A Retrieval Augmented Generation (RAG) system that provides expert parameter recommendations for CFD turbulence models using AI-powered analysis of research literature.

## Features

- **Expert Parameter Recommendations**: AI-generated parameter values based on research literature and application context
- **Multiple Turbulence Models**: Support for k-ε, k-ω SST, Spalart-Allmaras, and Reynolds Stress models
- **Document-Informed Decisions**: Semantic search through turbulence modeling research papers and documentation
- **Interactive Web Interface**: User-friendly Streamlit interface for model selection and parameter generation
- **Confidence Assessment**: Confidence scores for each parameter recommendation with detailed rationale
- **Export Capabilities**: Export recommendations in JSON, Markdown, or CSV formats

## System Architecture

```
User Input → Document Retrieval → LLM Generation → Parameter Recommendations
     ↓              ↓                    ↓                    ↓
Flow Details → Pinecone Vector DB → OpenAI GPT-4o → Structured Output
```

## Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Turb_Model_Param_Recom

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy the environment template and add your API keys:

```bash
cp .env.template .env
```

Edit `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=turbulence-docs
```

### 4. Database Setup

Populate the vector database with turbulence modeling documents:

```bash
python scripts/populate_database.py --mode populate
```

This will create a Pinecone index and populate it with sample turbulence modeling documents.

### 5. Launch Application

```bash
streamlit run ui/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

## Usage

1. **Select Turbulence Model**: Choose from k-ε, k-ω SST, Spalart-Allmaras, or Reynolds Stress models
2. **Describe Application**: Provide details about your CFD case, flow conditions, and geometry
3. **Specify Focus Area** (optional): Add specific constraints or areas of interest
4. **Generate Recommendations**: Click to get AI-powered parameter recommendations
5. **Review Results**: Examine parameter values, confidence scores, and rationales
6. **Export**: Download results in your preferred format

## Project Structure

```
Turb_Model_Param_Recom/
├── src/
│   ├── config.py                  # Configuration management
│   ├── turbulence_models.py       # Model definitions and schemas
│   ├── document_processor.py      # Document ingestion and processing
│   ├── vector_store.py            # Pinecone vector database interface
│   ├── retrieval_system.py        # Document retrieval logic
│   ├── parameter_generator.py     # LLM parameter generation
│   └── rag_pipeline.py            # Main RAG orchestration
├── ui/
│   └── streamlit_app.py           # Web interface
├── scripts/
│   └── populate_database.py       # Database population utility
├── data/                          # Document storage
├── tests/                         # Test files
└── requirements.txt               # Dependencies
```

## Supported Turbulence Models

### k-ε (Standard k-epsilon)
- **Parameters**: Cmu, C1ε, C2ε, σk, σε
- **Applications**: Free shear flows, wall-bounded flows with mild pressure gradients
- **Best For**: General-purpose industrial applications

### k-ω SST (Shear Stress Transport)
- **Parameters**: β*, α1, β1, α2, β2, σk1, σω1, σk2, σω2, a1
- **Applications**: Adverse pressure gradient flows, separated flows, airfoil aerodynamics
- **Best For**: Aerospace applications, flows with separation

### Spalart-Allmaras
- **Parameters**: Cb1, Cb2, Cv1, Cw1, Cw2, Cw3, σ
- **Applications**: Aerodynamic flows, external aerodynamics
- **Best For**: Aircraft design, external flow applications

### Reynolds Stress Model (RSM)
- **Parameters**: Cμ, C1ε, C2ε, C1, C2, σε, σk
- **Applications**: Complex flows with strong anisotropy, swirling flows
- **Best For**: Complex geometries, secondary flow prediction

## Advanced Usage

### Adding Custom Documents

Add your own research papers or documentation:

```bash
# Add local PDF files
python scripts/populate_database.py --mode add --files path/to/your/paper.pdf

# Add documents from URLs
python scripts/populate_database.py --mode populate --urls http://example.com/paper.pdf
```

### API Usage

Use the RAG pipeline programmatically:

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()
pipeline.initialize()

# Generate recommendations
response = pipeline.generate_recommendations(
    turbulence_model="k_omega_sst",
    user_description="External flow over airfoil with potential separation",
    focus_area="accurate separation prediction"
)

print(response['recommendations'])
```

### Testing Document Retrieval

Test the document search functionality:

```bash
python scripts/populate_database.py --mode test
```

## Customization

### Adding New Turbulence Models

1. Define the model in `src/turbulence_models.py`:
   ```python
   "your_model": TurbulenceModel(
       name="your_model",
       full_name="Your Model Full Name",
       description="Model description",
       # ... add parameters and metadata
   )
   ```

2. The system will automatically support the new model in the UI and API.

### Modifying System Prompts

Edit the prompt templates in `src/parameter_generator.py` to customize the AI behavior for specific use cases or domains.

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your `.env` file contains valid API keys
2. **Pinecone Connection**: Verify your Pinecone environment and index settings
3. **Document Processing**: Check document URLs are accessible and PDFs are not encrypted
4. **Memory Issues**: For large documents, adjust chunk sizes in `src/config.py`

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- Adjust `chunk_size` and `chunk_overlap` in configuration for your document types
- Tune `max_retrieval_docs` based on context window requirements
- Use document filtering for domain-specific applications

## Acknowledgments

- OpenAI for GPT-4o language model
- Pinecone for vector database infrastructure
- Streamlit for the web interface framework
- NASA Turbulence Modeling Resource for reference materials