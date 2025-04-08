"""Unit tests for Vectara tools."""

import json
import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_tests.unit_tests.tools import ToolsUnitTests

from langchain_vectara.tools import (
    VectaraGeneration,
    VectaraIngest,
    VectaraSearch,
)
from langchain_vectara.vectorstores import (
    CorpusConfig,
    GenerationConfig,
    SearchConfig,
    VectaraQueryConfig,
)


@pytest.fixture(autouse=True)
def mock_openai():
    """Create a pytest fixture to mock OpenAI dependencies."""
    with patch("langchain_community.tools.vectorstore.tool.OpenAI") as mock_openai:
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        yield


# Create a mock Vectara class that inherits from VectorStore
class MockVectara(VectorStore):
    """Mock Vectara class for testing."""

    def add_texts(self, *args, **kwargs):
        """Mock implementation."""
        return ["doc1", "doc2"]

    def similarity_search(self, *args, **kwargs):
        """Mock implementation of the required abstract method."""
        return [MagicMock(page_content="Test content", metadata={"source": "test"})]

    def similarity_search_with_score(self, *args, **kwargs):
        """Mock implementation."""
        return [
            (MagicMock(page_content="Test content", metadata={"source": "test"}), 0.85)
        ]

    def as_rag(self, *args, **kwargs):
        """Mock implementation."""
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Test answer",
            "fcs": 0.95,
        }
        return mock_rag

    @classmethod
    def from_texts(cls, *args, **kwargs):
        """Mock implementation."""
        return cls()


class TestVectaraSearchToolUnit(ToolsUnitTests):
    """Test VectaraSearch tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self):
        """Return the constructor for VectaraSearch."""
        return VectaraSearch

    @property
    def tool_constructor_params(self):
        """Return parameters for initializing VectaraSearch."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_search",
            "description": "Test search",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self):
        """Return example parameters for invoking VectaraSearch."""
        return {"query": "test query"}

    @property
    def tool_invoke_params_example(self):
        """Return example parameters for invoking VectaraSearch."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self):
        """Return parameters for testing initialization from environment variables."""
        mock_vectorstore = MockVectara()

        # Return as a tuple of (env_vars, init_args, expected_attrs)
        return (
            {},  # No direct env vars for the tool itself
            {
                "name": "test_search",
                "description": "Test search",
                "vectorstore": mock_vectorstore,
                "corpus_key": "test-corpus-123",
            },
            {
                "name": "test_search",
                "description": "Test search",
                "corpus_key": "test-corpus-123",
            },
        )


class TestVectaraGenerationToolUnit(ToolsUnitTests):
    """Test VectaraGeneration tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self):
        """Return the constructor for VectaraGeneration."""
        return VectaraGeneration

    @property
    def tool_constructor_params(self):
        """Return parameters for initializing VectaraGeneration."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_generation",
            "description": "Test generation",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self):
        """Return example parameters for invoking VectaraGeneration."""
        return {"query": "test query"}

    @property
    def tool_invoke_params_example(self):
        """Return example parameters for invoking VectaraGeneration."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self):
        """Return parameters for testing initialization from environment variables."""
        mock_vectorstore = MockVectara()

        # Return as a tuple of (env_vars, init_args, expected_attrs)
        return (
            {},  # No direct env vars for the tool itself
            {
                "name": "test_generation",
                "description": "Test generation",
                "vectorstore": mock_vectorstore,
                "corpus_key": "test-corpus-123",
            },
            {
                "name": "test_generation",
                "description": "Test generation",
                "corpus_key": "test-corpus-123",
            },
        )


class TestVectaraIngestToolUnit(ToolsUnitTests):
    """Test VectaraIngest tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self):
        """Return the constructor for VectaraIngest."""
        return VectaraIngest

    @property
    def tool_constructor_params(self):
        """Return parameters for initializing VectaraIngest."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_ingest",
            "description": "Test ingest",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self):
        """Return example parameters for invoking VectaraIngest."""
        return {
            "documents": ["Document 1", "Document 2"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}],
        }

    @property
    def tool_invoke_params_example(self):
        """Return example parameters for invoking VectaraIngest."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self):
        """Return parameters for testing initialization from environment variables."""
        mock_vectorstore = MockVectara()

        # Return as a tuple of (env_vars, init_args, expected_attrs)
        return (
            {},  # No direct env vars for the tool itself
            {
                "name": "test_ingest",
                "description": "Test ingest",
                "vectorstore": mock_vectorstore,
                "corpus_key": "test-corpus-123",
            },
            {
                "name": "test_ingest",
                "description": "Test ingest",
                "corpus_key": "test-corpus-123",
            },
        )


class TestVectaraTools(unittest.TestCase):
    """Test Vectara tools functionality."""

    @patch("langchain_vectara.Vectara")
    def test_vectara_search_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with config parameter."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.similarity_search_with_score = MagicMock(
            return_value=[
                (
                    MagicMock(page_content="Test content", metadata={"source": "test"}),
                    0.85,
                )
            ]
        )

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(corpora=[corpus_config], limit=10)

        query_config = VectaraQueryConfig(search=search_config)

        result = tool._run(
            query="test query",
            config=query_config,
        )

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "config" in call_kwargs
        assert isinstance(call_kwargs["config"], VectaraQueryConfig)

        results_data = json.loads(result)
        assert isinstance(results_data, list)
        assert len(results_data) == 1
        assert results_data[0]["content"] == "Test content"
        assert results_data[0]["source"] == "test"
        assert results_data[0]["score"] == 0.85
        assert "index" in results_data[0]
        assert "metadata" in results_data[0]

    @patch("langchain_vectara.Vectara")
    def test_vectara_search_without_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with default configuration."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.similarity_search_with_score = MagicMock(
            return_value=[
                (
                    MagicMock(page_content="Test content", metadata={"source": "test"}),
                    0.92,
                )
            ]
        )

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]

        config = call_kwargs["config"]
        assert config.search.corpora is not None
        assert len(config.search.corpora) == 1
        assert config.search.corpora[0].corpus_key == "test-corpus-123"

        results_data = json.loads(result)
        assert isinstance(results_data, list)
        assert len(results_data) == 1
        assert results_data[0]["content"] == "Test content"
        assert results_data[0]["source"] == "test"
        assert results_data[0]["score"] == 0.92

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with config parameter."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text",
            "fcs": 0.95,
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(corpora=[corpus_config], limit=10)

        generation_config = GenerationConfig(
            max_used_search_results=8,
            response_language="eng",
            generation_preset_name="test-prompt",
            enable_factual_consistency_score=True,
        )

        query_config = VectaraQueryConfig(
            search=search_config, generation=generation_config
        )

        result = tool._run(
            query="test query",
            config=query_config,
        )

        mock_vectorstore.as_rag.assert_called_once_with(query_config)
        mock_rag.invoke.assert_called_once_with("test query")

        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text"
        assert summary_data["factual_consistency_score"] == 0.95

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_with_defaults(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with default configuration."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text",
            "fcs": 0.95,
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        mock_vectorstore.as_rag.assert_called_once()

        config = mock_vectorstore.as_rag.call_args[0][0]
        assert isinstance(config, VectaraQueryConfig)
        assert config.search.corpora is not None
        assert len(config.search.corpora) == 1
        assert config.search.corpora[0].corpus_key == "test-corpus-123"
        assert config.generation is not None
        assert (
            config.generation.generation_preset_name
            == "vectara-summary-ext-24-05-med-omni"
        )

        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text"
        assert summary_data["factual_consistency_score"] == 0.95

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_without_answer(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with no answer in response."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": None,  # Explicitly set answer to None
            "fcs": "N/A",  # Set fcs to "N/A" to match expected behavior
            "context": [
                (
                    MagicMock(
                        page_content="Document content", metadata={"source": "test"}
                    ),
                    0.9,
                )
            ],
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] is None
        assert summary_data["factual_consistency_score"] == "N/A"

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_with_no_results(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with empty response."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = None
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        assert result == "No results found"

    @patch("langchain_vectara.Vectara")
    def test_vectara_ingest(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with basic parameters."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_texts = MagicMock(return_value=["doc1", "doc2"])

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        result = tool._run(
            documents=documents,
            metadatas=metadatas,
        )

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents,
            metadatas=metadatas,
            ids=None,
            corpus_key="test-corpus-123",
        )

        assert "Successfully ingested 2 documents" in result
        assert "test-corpus-123" in result
        assert "doc1, doc2" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_ingest_with_all_parameters(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with all available parameters."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_texts = MagicMock(return_value=["custom1", "custom2"])

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        documents = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["custom1", "custom2"]
        doc_metadata = {"batch": "test-batch", "department": "engineering"}
        doc_type = "structured"

        result = tool._run(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            doc_metadata=doc_metadata,
            doc_type=doc_type,
        )

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents,
            metadatas=metadatas,
            ids=ids,
            corpus_key="test-corpus-123",
            doc_metadata=doc_metadata,
            doc_type=doc_type,
        )

        assert "Successfully ingested 2 documents" in result
        assert "test-corpus-123" in result
        assert "custom1, custom2" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_ingest_with_override_corpus(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with corpus_key override."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_texts = MagicMock(return_value=["doc1", "doc2"])

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="default-corpus",
        )

        documents = ["Document 1", "Document 2"]

        result = tool._run(documents=documents, corpus_key="override-corpus")

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=documents, metadatas=None, ids=None, corpus_key="override-corpus"
        )

        assert "Successfully ingested 2 documents" in result
        assert "override-corpus" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_search_without_corpus_key(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with no corpus_key provided."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        # Create a custom check for corpus_key
        def mock_similarity_search_with_score(query, **kwargs):
            if "config" in kwargs:
                config = kwargs["config"]
                if not config.search.corpora or not config.search.corpora[0].corpus_key:
                    raise ValueError("A corpus_key is required for search")
            return [
                (
                    MagicMock(page_content="Test content", metadata={"source": "test"}),
                    0.85,
                )
            ]

        mock_vectorstore.similarity_search_with_score = MagicMock(
            side_effect=mock_similarity_search_with_score
        )

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key=None,
        )

        result = tool._run(
            query="test query",
        )

        # Match the actual error format from the tool
        assert "Error" in result
        assert "corpus_key is required for search" in result

        empty_config = VectaraQueryConfig(search=SearchConfig())

        result = tool._run(query="test query", config=empty_config)

        # Match the actual error format from the tool
        assert "Error searching Vectara" in result
        assert "corpus_key is required for search" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_without_corpus_key(
        self, mock_vectara: MagicMock
    ) -> None:
        """Test VectaraGeneration tool with no corpus_key provided."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        # Create a custom as_rag method that checks for corpus_key
        def mock_as_rag(config):
            if not config.search.corpora or not config.search.corpora[0].corpus_key:
                raise ValueError("A corpus_key is required for generation")

            mock_rag = MagicMock()
            mock_rag.invoke.return_value = {
                "answer": "Test answer",
                "fcs": 0.95,
            }
            return mock_rag

        mock_vectorstore.as_rag = MagicMock(side_effect=mock_as_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key=None,
        )

        result = tool._run(
            query="test query",
        )

        # Match the actual error format from the tool
        assert "Error" in result
        assert "corpus_key is required for generation" in result

        empty_config = VectaraQueryConfig(
            search=SearchConfig(), generation=GenerationConfig()
        )

        result = tool._run(query="test query", config=empty_config)

        # Match the actual error format from the tool
        assert "Error generating response from Vectara" in result
        assert "corpus_key is required for generation" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_without_context(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with response missing context field."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text without context",
            "fcs": 0.80,
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text without context"
        assert summary_data["factual_consistency_score"] == 0.80

    @patch("langchain_vectara.Vectara")
    def test_vectara_generation_with_empty_dict(self, mock_vectara: MagicMock) -> None:
        """Test VectaraGeneration tool with empty dictionary response."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {}
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)

        tool = VectaraGeneration(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        result = tool._run(
            query="test query",
        )

        # For an empty dict, the tool is expected to return "No results found"
        assert result == "No results found"
