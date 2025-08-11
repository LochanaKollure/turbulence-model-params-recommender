"""
Script to populate the Pinecone vector database with turbulence modeling documents.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample turbulence modeling document URLs and papers
SAMPLE_DOCUMENTS = [
    # NASA Turbulence Modeling Resource
    #"https://turbmodels.larc.nasa.gov/spalart.html",
    #"https://turbmodels.larc.nasa.gov/ke-chien.html", 
    #"https://turbmodels.larc.nasa.gov/sst.html",
    
    # Example academic papers (these are placeholder URLs - in real implementation, 
    # you would use actual PDF URLs from journals)
    #"https://www.cfd-online.com/Wiki/Spalart-Allmaras_model",
    #"https://www.cfd-online.com/Wiki/K-epsilon_models",
    #"https://www.cfd-online.com/Wiki/SST_k-omega_model",


    "https://doc.comsol.com/5.5/doc/com.comsol.help.cfd/cfd_ug_fluidflow_single.06.093.html",
    "https://resources.system-analysis.cadence.com/blog/msa2024-what-is-the-spalart-allmaras-turbulence-model",
    "https://journal.gpps.global/Calibration-of-Spalart-Allmaras-model-for-simulation-of-corner-flow-separation-in,135174,0,2.html",
    "https://www.cfdsupport.com/openfoam-training-by-cfd-support/node398/",
    "https://www.sciencedirect.com/topics/engineering/allmaras-model",
    "https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf",
    "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?params=/context/etd/article/3052/&path_info=Tong_Oisin.pdf",
    "https://www.researchgate.net/publication/291815988_Review_of_the_Spalart-Allmaras_turbulence_model_and_its_modifications_to_three-dimensional_supersonic_configurations",

]

def populate_database(document_urls: List[str] = None):
    """Populate the vector database with documents."""
    
    if document_urls is None:
        document_urls = SAMPLE_DOCUMENTS
    
    # Initialize processors
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Create or connect to index
    logger.info("Setting up vector store...")
    if not vector_store.create_index():
        logger.error("Failed to create/connect to index")
        return False
    
    # Process each document
    all_chunks = []
    successful_docs = 0
    failed_docs = 0
    
    for url in document_urls:
        try:
            logger.info(f"Processing document: {url}")
            chunks = doc_processor.process_document(url, doc_type="url")
            
            if chunks:
                all_chunks.extend(chunks)
                successful_docs += 1
                logger.info(f"Successfully processed {len(chunks)} chunks from {url}")
            else:
                logger.warning(f"No content extracted from {url}")
                failed_docs += 1
                
        except Exception as e:
            logger.error(f"Failed to process {url}: {str(e)}")
            failed_docs += 1
            continue
    
    if not all_chunks:
        logger.error("No documents were successfully processed")
        return False
    
    # Upload to vector store
    logger.info(f"Uploading {len(all_chunks)} chunks to vector store...")
    if vector_store.upsert_documents(all_chunks):
        logger.info("Successfully populated vector database")
        
        # Get and display stats
        stats = vector_store.get_index_stats()
        logger.info(f"Index stats: {stats}")
        
        logger.info(f"Summary: {successful_docs} successful, {failed_docs} failed documents")
        return True
    else:
        logger.error("Failed to upload documents to vector store")
        return False

def add_local_documents(file_paths: List[str]):
    """Add local documents to the database."""
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Connect to existing index
    if not vector_store.connect_to_index():
        logger.error("Failed to connect to index")
        return False
    
    all_chunks = []
    for file_path in file_paths:
        try:
            logger.info(f"Processing local file: {file_path}")
            chunks = doc_processor.process_document(file_path)
            all_chunks.extend(chunks)
            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            continue
    
    if all_chunks and vector_store.upsert_documents(all_chunks):
        logger.info(f"Successfully added {len(all_chunks)} chunks to database")
        return True
    
    return False

def test_retrieval():
    """Test document retrieval functionality."""
    vector_store = VectorStore()
    
    if not vector_store.connect_to_index():
        logger.error("Failed to connect to index for testing")
        return
    
    # Test queries
    test_queries = [
        "k-epsilon turbulence model constants",
        "Spalart-Allmaras model parameters",
        "SST k-omega model coefficients",
        "turbulent viscosity",
        "pressure gradient effects"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        results = vector_store.query_similar_documents(query, top_k=3)
        
        for i, result in enumerate(results):
            logger.info(f"  Result {i+1}: Score={result['score']:.3f}, Source={result['metadata'].get('source', 'Unknown')}")
            logger.info(f"    Preview: {result['text_preview'][:150]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate turbulence model database")
    parser.add_argument("--mode", choices=["populate", "add", "test"], default="populate",
                       help="Mode: populate (fresh), add (to existing), or test")
    parser.add_argument("--files", nargs="+", help="Local files to add (for add mode)")
    parser.add_argument("--urls", nargs="+", help="URLs to process (for populate mode)")
    
    args = parser.parse_args()
    
    if args.mode == "populate":
        urls = args.urls if args.urls else SAMPLE_DOCUMENTS
        success = populate_database(urls)
        sys.exit(0 if success else 1)
        
    elif args.mode == "add":
        if not args.files:
            logger.error("No files specified for add mode")
            sys.exit(1)
        success = add_local_documents(args.files)
        sys.exit(0 if success else 1)
        
    elif args.mode == "test":
        test_retrieval()
        sys.exit(0)