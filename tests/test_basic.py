"""
Basic integration tests for the RAG system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from src.turbulence_models import get_model_names, get_model
from src.rag_pipeline import RAGPipeline

class TestTurbulenceModels(unittest.TestCase):
    
    def test_get_model_names(self):
        """Test that model names are returned."""
        models = get_model_names()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn("k_epsilon", models)
        self.assertIn("k_omega_sst", models)
    
    def test_get_model(self):
        """Test getting specific model information."""
        model = get_model("k_epsilon")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "k_epsilon")
        self.assertGreater(len(model.parameters), 0)
    
    def test_invalid_model(self):
        """Test handling of invalid model names."""
        model = get_model("nonexistent_model")
        self.assertIsNone(model)

class TestRAGPipeline(unittest.TestCase):
    
    def setUp(self):
        """Set up test pipeline."""
        self.pipeline = RAGPipeline()
    
    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        self.assertIsNotNone(self.pipeline)
        self.assertFalse(self.pipeline.initialized)
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = self.pipeline.get_available_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
    
    def test_validate_model(self):
        """Test model validation."""
        self.assertTrue(self.pipeline.validate_model("k_epsilon"))
        self.assertFalse(self.pipeline.validate_model("invalid_model"))
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.pipeline.get_model_info("k_epsilon")
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], "k_epsilon")
        self.assertIn('parameters', info)
        self.assertIn('applications', info)
    
    @patch('src.rag_pipeline.RetrievalSystem')
    def test_initialization_mock(self, mock_retrieval):
        """Test pipeline initialization with mocked components."""
        # Mock successful initialization
        mock_retrieval_instance = Mock()
        mock_retrieval_instance.initialize.return_value = True
        mock_retrieval.return_value = mock_retrieval_instance
        
        pipeline = RAGPipeline()
        result = pipeline.initialize()
        
        self.assertTrue(result)
        self.assertTrue(pipeline.initialized)

class TestDocumentProcessor(unittest.TestCase):
    
    def setUp(self):
        from src.document_processor import DocumentProcessor
        self.processor = DocumentProcessor()
    
    def test_processor_creation(self):
        """Test document processor can be created."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.chunk_size, 1000)
        self.assertEqual(self.processor.chunk_overlap, 200)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is   a   test\n\n\n  with   extra   spaces  "
        clean_text = self.processor._clean_text(dirty_text)
        self.assertEqual(clean_text, "This is a test with extra spaces")
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = "This is a test document. " * 100  # Create long text
        chunks = self.processor.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('metadata', chunk)

if __name__ == '__main__':
    # Run basic tests that don't require API keys
    unittest.main(verbosity=2)