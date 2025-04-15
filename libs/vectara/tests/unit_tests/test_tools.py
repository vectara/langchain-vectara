"""Unit tests for Vectara tools."""

import json
import unittest
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_tests.unit_tests.tools import ToolsUnitTests

from langchain_vectara.tools import (
    VectaraAddFiles,
    VectaraIngest,
    VectaraRAG,
    VectaraSearch,
)
from langchain_vectara.vectorstores import (
    ChunkingStrategy,
    CorpusConfig,
    File,
    GenerationConfig,
    SearchConfig,
    TableExtractionConfig,
    Vectara,
    VectaraQueryConfig,
)

# Type alias for the return type of init_from_env_params
EnvParamsType = Tuple[Dict[str, str], Dict[str, Any], Dict[str, Any]]


@pytest.fixture(autouse=True)
def mock_openai() -> Generator[None, None, None]:
    """Create a pytest fixture to mock OpenAI dependencies."""
    with patch("langchain_community.tools.vectorstore.tool.OpenAI") as mock_openai:
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        yield


# Create a mock Vectara class that inherits from Vectara
class MockVectara(Vectara):
    """Mock Vectara class for testing."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Mock init that doesn't require any parameters."""
        # Don't call super().__init__ to avoid API calls
        pass

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Mock implementation."""
        return ["doc1", "doc2"]

    def add_files(
        self,
        files_list: List[File],
        corpus_key: str,
        **kwargs: Any,
    ) -> List[str]:
        """Mock implementation."""
        return ["file1", "file2"]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Mock implementation of the required abstract method."""
        return [MagicMock(page_content="Test content", metadata={"source": "test"})]

    def similarity_search_with_score(
        self, query: str, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Mock implementation."""
        return [
            (MagicMock(page_content="Test content", metadata={"source": "test"}), 0.85)
        ]

    def as_rag(self, config: Any, **kwargs: Any) -> Any:
        """Mock implementation."""
        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {  # type: ignore[attr-defined]
            "answer": "Test answer",
            "fcs": 0.95,
        }
        return mock_rag

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "MockVectara":
        """Mock implementation."""
        return cls()


class TestVectaraSearchToolUnit(ToolsUnitTests):
    """Test VectaraSearch tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self) -> Type[VectaraSearch]:
        """Return the constructor for VectaraSearch."""
        return VectaraSearch

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        """Return parameters for initializing VectaraSearch."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_search",
            "description": "Test search",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self) -> Dict[str, str]:
        """Return example parameters for invoking VectaraSearch."""
        return {"query": "test query"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, str]:
        """Return example parameters for invoking VectaraSearch."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self) -> EnvParamsType:
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


class TestVectaraRAGToolUnit(ToolsUnitTests):
    """Test VectaraRAG tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self) -> Type[VectaraRAG]:
        """Return the constructor for VectaraRAG."""
        return VectaraRAG

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        """Return parameters for initializing VectaraRAG."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_generation",
            "description": "Test generation",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self) -> Dict[str, str]:
        """Return example parameters for invoking VectaraRAG."""
        return {"query": "test query"}

    @property
    def tool_invoke_params_example(self) -> Dict[str, str]:
        """Return example parameters for invoking VectaraRAG."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self) -> EnvParamsType:
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
    def tool_constructor(self) -> Type[VectaraIngest]:
        """Return the constructor for VectaraIngest."""
        return VectaraIngest

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        """Return parameters for initializing VectaraIngest."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_ingest",
            "description": "Test ingest",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
            "doc_type": "structured",
        }

    @property
    def tool_invoke_params(self) -> Dict[str, Any]:
        """Return example parameters for invoking VectaraIngest."""
        return {
            "texts": ["Text 1", "Text 2"],
            "metadatas": [{"source": "test1"}, {"source": "test2"}],
        }

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        """Return example parameters for invoking VectaraIngest."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self) -> EnvParamsType:
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


class TestVectaraAddFilesToolUnit(ToolsUnitTests):
    """Test VectaraAddFiles tool with LangChain standard unit tests."""

    @property
    def tool_constructor(self) -> Type[VectaraAddFiles]:
        """Return the constructor for VectaraAddFiles."""
        return VectaraAddFiles

    @property
    def tool_constructor_params(self) -> Dict[str, Any]:
        """Return parameters for initializing VectaraAddFiles."""
        mock_vectorstore = MockVectara()
        return {
            "name": "test_add_files",
            "description": "Test add files",
            "vectorstore": mock_vectorstore,
            "corpus_key": "test-corpus-123",
        }

    @property
    def tool_invoke_params(self) -> Dict[str, Any]:
        """Return example parameters for invoking VectaraAddFiles."""
        return {
            "files": [
                File(file_path="/tmp/test1.pdf", metadata={"source": "test1"}),
                File(file_path="/tmp/test2.docx", metadata={"source": "test2"}),
            ]
        }

    @property
    def tool_invoke_params_example(self) -> Dict[str, Any]:
        """Return example parameters for invoking VectaraAddFiles."""
        return self.tool_invoke_params

    @property
    def init_from_env_params(self) -> EnvParamsType:
        """Return parameters for testing initialization from environment variables."""
        mock_vectorstore = MockVectara()

        # Return as a tuple of (env_vars, init_args, expected_attrs)
        return (
            {},  # No direct env vars for the tool itself
            {
                "name": "test_add_files",
                "description": "Test add files",
                "vectorstore": mock_vectorstore,
                "corpus_key": "test-corpus-123",
            },
            {
                "name": "test_add_files",
                "description": "Test add files",
                "corpus_key": "test-corpus-123",
            },
        )


class TestVectaraTools(unittest.TestCase):
    """Test Vectara tools functionality."""

    # -----------------
    # VectaraRAG Tests
    # -----------------

    @patch("langchain_vectara.Vectara")
    def test_vectara_rag_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with a custom query configuration.

        This test verifies that the VectaraRAG tool properly uses a custom
        VectaraQueryConfig with specific search and generation parameters.
        """
        # Set up mock vectorstore
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {  # type: ignore[attr-defined]
            "answer": "Summary text",
            "fcs": 0.95,
            "context": [
                (
                    MagicMock(
                        page_content="Document content",
                        metadata={"source": "test", "title": "Test Document"},
                    ),
                    0.9,
                ),
                (
                    MagicMock(
                        page_content="Additional content",
                        metadata={"source": "test2", "title": "Second Document"},
                    ),
                    0.8,
                ),
            ],
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(
            corpora=[corpus_config],
            limit=10,
        )

        generation_config = GenerationConfig(
            max_used_search_results=8,
            response_language="eng",
            generation_preset_name="test-custom-prompt",
            enable_factual_consistency_score=True,
        )

        query_config = VectaraQueryConfig(
            search=search_config, generation=generation_config
        )

        tool = VectaraRAG(
            name="test_rag",
            description="Test RAG generation",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
            config=query_config,
        )

        result = tool._run(query="test query")

        mock_vectorstore.as_rag.assert_called_once_with(query_config)
        mock_rag.invoke.assert_called_once_with("test query")

        summary_data = json.loads(result)
        assert isinstance(summary_data, dict)
        assert summary_data["summary"] == "Summary text"
        assert summary_data["factual_consistency_score"] == 0.95

    @patch("langchain_vectara.Vectara")
    def test_vectara_rag_with_defaults(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with default configuration."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {  # type: ignore[attr-defined]
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
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
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
    def test_vectara_rag_without_answer(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with no answer in response."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {  # type: ignore[attr-defined]
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
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
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
    def test_vectara_rag_with_no_results(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with empty response."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = None  # type: ignore[attr-defined]
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
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
    def test_vectara_rag_without_corpus_key(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with no corpus_key provided."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        # Create a custom as_rag method that checks for corpus_key
        def mock_as_rag(config: Any) -> Any:
            if not config.search.corpora or not config.search.corpora[0].corpus_key:
                raise ValueError("A corpus_key is required for generation")

            mock_rag = MagicMock()
            mock_rag.invoke.return_value = {
                "answer": "Test answer",
                "fcs": 0.95,
            }
            return mock_rag

        mock_vectorstore.as_rag = MagicMock(side_effect=mock_as_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
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

        tool = VectaraRAG(
            name="test_generation",
            description="Test generation",
            vectorstore=mock_vectorstore,
            corpus_key=None,
            config=empty_config,
        )

        result = tool._run(query="test query")

        # Match the actual error format from the tool
        assert "Error generating response from Vectara" in result
        assert "corpus_key is required for generation" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_rag_without_context(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with response missing the context field.

        This test verifies that the VectaraRAG tool correctly handles API responses
        that don't include the 'context' field (which normally contains the source
        documents). Even without context documents, the tool should still return a
        properly formatted response with the answer and factual consistency score.
        """
        # Set up mock vectorstore
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {
            "answer": "Summary text without context",
            "fcs": 0.80,
        }
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
            name="test_rag",
            description="Test RAG generation",
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
    def test_vectara_rag_with_empty_dict(self, mock_vectara: MagicMock) -> None:
        """Test VectaraRAG tool with empty dictionary response."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_rag = MagicMock()
        mock_rag.invoke.return_value = {}
        mock_vectorstore.as_rag = MagicMock(return_value=mock_rag)  # type: ignore[method-assign]

        tool = VectaraRAG(
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

    # -----------------------
    # VectaraSearch Tests
    # -----------------------

    @patch("langchain_vectara.Vectara")
    def test_vectara_search_with_config(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with config parameter."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.similarity_search_with_score = MagicMock(  # type: ignore[method-assign]
            return_value=[
                (
                    MagicMock(page_content="Test content", metadata={"source": "test"}),
                    0.85,
                )
            ]
        )

        corpus_config = CorpusConfig(
            corpus_key="test-corpus-123",
            metadata_filter="doc.type = 'article'",
            lexical_interpolation=0.2,
        )

        search_config = SearchConfig(corpora=[corpus_config], limit=10)

        tool = VectaraSearch(
            name="test_search",
            description="Test search",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
            search_config=search_config,
        )

        result = tool._run(
            query="test query",
        )

        mock_vectorstore.similarity_search_with_score.assert_called_once()
        call_kwargs = mock_vectorstore.similarity_search_with_score.call_args[1]
        assert "search" in call_kwargs
        assert isinstance(call_kwargs["search"], SearchConfig)

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

        mock_vectorstore.similarity_search_with_score = MagicMock(  # type: ignore[method-assign]
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

        search_config = call_kwargs["search"]
        assert search_config.corpora is not None
        assert len(search_config.corpora) == 1
        assert search_config.corpora[0].corpus_key == "test-corpus-123"

        results_data = json.loads(result)
        assert isinstance(results_data, list)
        assert len(results_data) == 1
        assert results_data[0]["content"] == "Test content"
        assert results_data[0]["source"] == "test"
        assert results_data[0]["score"] == 0.92

    @patch("langchain_vectara.Vectara")
    def test_vectara_search_without_corpus_key(self, mock_vectara: MagicMock) -> None:
        """Test VectaraSearch tool with no corpus_key provided."""
        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        # Create a custom check for corpus_key
        def mock_similarity_search_with_score(
            query: str, **kwargs: Any
        ) -> List[Tuple[Any, float]]:
            search_config = kwargs.get("search_config")
            if search_config:
                if not search_config.corpora[0].corpus_key:
                    raise ValueError("A corpus_key is required for search")

            return [
                (
                    MagicMock(page_content="Test content", metadata={"source": "test"}),
                    0.85,
                )
            ]

        mock_vectorstore.similarity_search_with_score = MagicMock(  # type: ignore[method-assign]
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

    # -----------------------
    # VectaraIngest Tests
    # -----------------------

    @patch("langchain_vectara.Vectara")
    def test_vectara_ingest(self, mock_vectara: MagicMock) -> None:
        """Test VectaraIngest tool with basic parameters."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_texts = MagicMock(return_value=["doc1", "doc2"])  # type: ignore[method-assign]

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        result = tool._run(
            texts=texts,
            metadatas=metadatas,
        )

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=texts,
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

        mock_vectorstore.add_texts = MagicMock(return_value=["custom1", "custom2"])  # type: ignore[method-assign]

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["custom1", "custom2"]
        doc_metadata = {"batch": "test-batch", "department": "engineering"}
        doc_type = "structured"

        result = tool._run(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            doc_metadata=doc_metadata,
            doc_type=doc_type,
        )

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=texts,
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

        mock_vectorstore.add_texts = MagicMock(return_value=["doc1", "doc2"])  # type: ignore[method-assign]

        tool = VectaraIngest(
            name="test_ingest",
            description="Test ingest",
            vectorstore=mock_vectorstore,
            corpus_key="default-corpus",
        )

        texts = ["Text 1", "Text 2"]

        result = tool._run(texts=texts, corpus_key="override-corpus")

        mock_vectorstore.add_texts.assert_called_once_with(
            texts=texts, metadatas=None, ids=None, corpus_key="override-corpus"
        )

        assert "Successfully ingested 2 documents" in result
        assert "override-corpus" in result

    # -----------------------
    # VectaraAddFiles Tests
    # -----------------------

    @patch("langchain_vectara.Vectara")
    def test_vectara_add_files(self, mock_vectara: MagicMock) -> None:
        """Test VectaraAddFiles tool with basic parameters."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_files = MagicMock(return_value=["file1", "file2"])  # type: ignore[method-assign]

        tool = VectaraAddFiles(
            name="test_add_files",
            description="Test add files",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        files = [
            File(file_path="/tmp/test1.pdf", metadata={"source": "test1"}),
            File(file_path="/tmp/test2.docx", metadata={"source": "test2"}),
        ]

        result = tool._run(
            files=files,
        )

        # Check that the mock was called with expected parameters
        mock_vectorstore.add_files.assert_called_once_with(
            files_list=files, corpus_key="test-corpus-123"
        )

        assert "Successfully uploaded 2 files" in result
        assert "test-corpus-123" in result
        assert "file1, file2" in result

    @patch("langchain_vectara.Vectara")
    def test_vectara_add_files_with_all_parameters(
        self, mock_vectara: MagicMock
    ) -> None:
        """Test VectaraAddFiles tool with all available parameters."""

        mock_vectorstore = MockVectara()
        mock_vectara.return_value = mock_vectorstore

        mock_vectorstore.add_files = MagicMock(return_value=["custom1", "custom2"])  # type: ignore[method-assign]

        tool = VectaraAddFiles(
            name="test_add_files",
            description="Test add files",
            vectorstore=mock_vectorstore,
            corpus_key="test-corpus-123",
        )

        files = [
            File(
                file_path="/tmp/test1.pdf",
                metadata={"source": "test1"},
                filename="renamed1.pdf",
                chunking_strategy=ChunkingStrategy(max_chars_per_chunk=1000),
                table_extraction_config=TableExtractionConfig(extract_tables=True),
            ),
            File(
                file_path="/tmp/test2.docx",
                metadata={"source": "test2"},
                filename="renamed2.docx",
                chunking_strategy=ChunkingStrategy(max_chars_per_chunk=1000),
                table_extraction_config=TableExtractionConfig(extract_tables=True),
            ),
        ]

        custom_corpus_key = "override-corpus-key"

        result = tool._run(files=files, corpus_key=custom_corpus_key)

        # Check that the mock was called with expected parameters
        mock_vectorstore.add_files.assert_called_once_with(
            files_list=files, corpus_key=custom_corpus_key
        )

        assert "Successfully uploaded 2 files" in result
        assert "override-corpus-key" in result
        assert "custom1, custom2" in result
