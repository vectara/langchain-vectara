"""Tools for interacting with Vectara."""

import json
from typing import Any, Dict, List, Optional, Type

from langchain_community.tools.vectorstore.tool import BaseVectorStoreTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_vectara.vectorstores import (
    CorpusConfig,
    File,
    GenerationConfig,
    SearchConfig,
    Vectara,
    VectaraQueryConfig,
)


class VectaraIngestInput(BaseModel):
    """Input for the Vectara ingest tool."""

    texts: List[str] = Field(description="List of texts to ingest into Vectara")
    metadatas: Optional[List[Dict]] = Field(
        default=None, description="Optional metadata for each document"
    )
    ids: Optional[List[str]] = Field(
        default=None, description="Optional list of IDs associated with each document"
    )
    corpus_key: Optional[str] = Field(
        default=None, description="Corpus key where documents will be ingested"
    )
    doc_metadata: Optional[Dict] = Field(
        default=None, description="Optional metadata at the document level"
    )
    doc_type: Optional[str] = Field(
        default="structured",
        description=(
            "Optional document type ('core' or 'structured'). Defaults to 'structured'"
        ),
    )


class VectaraSearch(BaseVectorStoreTool, BaseTool):
    """Tool for searching the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_vectara.tools import VectaraSearch
            from langchain_vectara import Vectara
            from langchain_vectara.vectorstores import (
                VectaraQueryConfig,
                SearchConfig,
                CorpusConfig,
            )

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create a SearchConfig
            corpus_config = CorpusConfig(
                corpus_key="your-corpus-id",
                metadata_filter="doc.type = 'article'",
                lexical_interpolation=0.2
            )

            search_config = SearchConfig(
                corpora=[corpus_config],
                limit=10
            )

            # Create the tool with config
            tool = VectaraSearch(
                name="vectara_search",
                description="Search for information in the Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-key",
                search_config=search_config  # Pass config at initialization
            )

            # Use the tool
            results = tool.run("What is RAG?")
    """

    name: str = "vectara_search"
    description: str = (
        "Search for information in your Vectara corpus using semantic search. "
        "This tool understands the meaning of your query beyond simple keyword "
        "matching. "
        "Useful for retrieving specific information from your documents based on "
        "meaning and context. "
    )
    vectorstore: Vectara

    # Default corpus_key if not provided in the config
    corpus_key: Optional[str] = None

    search_config: Optional[SearchConfig] = None

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need to answer questions about {name} using "
            "semantic search. "
            "This tool understands the meaning and context of your query, not just "
            "keywords. "
            "Whenever you need information about {description} you should use this. "
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara search.

        Args:
            query: The query to search for
            run_manager: Optional callback manager for the run

        Returns:
            JSON string containing search results
        """
        try:
            if not self.corpus_key and not self.search_config:
                return (
                    "Error: A corpus_key is required for search. "
                    "You can provide it either directly to the tool or in the "
                    "search_config object."
                )

            if not self.search_config:
                self.search_config = SearchConfig()

                if self.corpus_key:
                    corpus_config = CorpusConfig(corpus_key=self.corpus_key)
                    self.search_config.corpora = [corpus_config]

            results = self.vectorstore.similarity_search_with_score(
                query, search=self.search_config, generation=None
            )

            if not results:
                return "No results found"

            # Directly serialize structured results with scores
            return json.dumps(
                [
                    {
                        "index": i,
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Vectara"),
                        "metadata": doc.metadata,
                        "score": score,
                    }
                    for i, (doc, score) in enumerate(results)
                ],
                indent=2,
            )
        except Exception as e:
            return f"Error searching Vectara: {str(e)}"


class VectaraRAG(BaseVectorStoreTool, BaseTool):
    """Tool for generating summaries from the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_community.tools import VectaraRAG
            from langchain_vectara import Vectara
            from langchain_vectara.vectorstores import (
                VectaraQueryConfig,
                SearchConfig,
                CorpusConfig,
                GenerationConfig,
            )

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create a VectaraQueryConfig with search and generation settings
            corpus_config = CorpusConfig(
                corpus_key="your-corpus-key",
                metadata_filter="doc.type = 'article'",
                lexical_interpolation=0.2
            )

            search_config = SearchConfig(
                corpora=[corpus_config],
                limit=10
            )

            generation_config = GenerationConfig(
                max_used_search_results=10,
                response_language="eng",
                generation_preset_name="vectara-summary-ext-24-05-med-omni"
            )

            query_config = VectaraQueryConfig(
                search=search_config,
                generation=generation_config
            )

             # Create the tool
            tool = VectaraRAG(
                name="vectara_rag",
                description="Generate summaries from your Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-id", # optional, provide the corpus key if
                                             # you have not provided the config
                config=query_config,
            )


            # Use the tool with the config
            results = tool.run({
                "query": "What is RAG?",
                "config": query_config
            })
    """

    name: str = "vectara_rag"
    description: str = (
        "Generate AI responses from your Vectara corpus using semantic search. "
        "This tool understands the meaning of your query and generates a concise "
        "summary from the most relevant results. "
    )
    vectorstore: Vectara

    # Default corpus_key if not provided in the config
    corpus_key: Optional[str] = None

    config: Optional[VectaraQueryConfig] = None

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need AI-generated answers about {name}. "
            "This tool understands the meaning of your query and generates a concise "
            "response from relevant documents. "
            "Whenever you need a comprehensive overview about {description} you should "
            "use this. "
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara generation."""
        try:
            if not self.config and not self.corpus_key:
                return (
                    "Error: A corpus_key is required for generation. "
                    "You can provide it either directly to the tool or in the "
                    "config object."
                )

            if not self.config:
                search_config = SearchConfig()

                if self.corpus_key:
                    corpus_config = CorpusConfig(corpus_key=self.corpus_key)
                    search_config.corpora = [corpus_config]

                generation_config = GenerationConfig(
                    max_used_search_results=7,
                    response_language="eng",
                    generation_preset_name="vectara-summary-ext-24-05-med-omni",
                    enable_factual_consistency_score=True,
                )

                self.config = VectaraQueryConfig(
                    search=search_config, generation=generation_config
                )

            rag = self.vectorstore.as_rag(self.config)  # type: ignore[attr-defined]
            result = rag.invoke(query)

            if not result:
                return "No results found"

            return json.dumps(
                {
                    "summary": result.get("answer"),
                    "factual_consistency_score": result.get("fcs", None),
                },
                indent=2,
            )

        except Exception as e:
            return f"Error generating response from Vectara: {str(e)}"


class VectaraIngest(BaseVectorStoreTool, BaseTool):
    """Tool for ingesting documents into the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_community.tools import VectaraIngest
            from langchain_vectara import Vectara

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create the tool
            tool = VectaraIngest(
                name="vectara_ingest",
                description="Ingest documents into the Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-key"  # Required for ingestion
            )

            # Use the tool with additional parameters
            result = tool.run({
                "texts": ["Text 1", "Text 2"],
                "metadatas": [{"source": "file1"}, {"source": "file2"}],
                "ids": ["doc1", "doc2"],
                "doc_metadata": {"batch": "batch1"},
                "doc_type": "structured"
            })
    """

    name: str = "vectara_ingest"
    description: str = (
        "Ingest documents into your Vectara corpus for semantic search. "
        "Useful for adding new information to your knowledge base that can be "
        "queried using natural language. "
        "Documents will be processed to understand their meaning and context. "
        "Input should be a list of texts to ingest."
    )
    args_schema: Type[BaseModel] = VectaraIngestInput
    vectorstore: Vectara

    # Required corpus_key for ingestion
    corpus_key: str = Field(
        ...,  # This makes it required
        description="Corpus key where documents will be ingested",
    )

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need to add new information to {name} for "
            "semantic search. "
            "Documents added will be processed for meaning and context to enable "
            "natural language querying. "
            "Whenever you need to ingest new documents about {description} "
            "you should use this. "
            "Input should be a list of text documents to ingest."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        corpus_key: Optional[str] = None,
        doc_metadata: Optional[Dict] = None,
        doc_type: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara ingest.

        Args:
            texts: List of texts to ingest into Vectara
            metadatas: Optional metadata for each document
            ids: Optional list of IDs for each document
            corpus_key: Optional corpus key override
            doc_metadata: Optional metadata at the document level
            doc_type: Optional document type ('core' or 'structured')
            run_manager: Optional callback manager

        Returns:
            String describing the ingestion result
        """
        try:
            active_corpus_key = corpus_key or self.corpus_key

            if not active_corpus_key:
                return "Error: corpus_key is required for ingestion"

            # Create a dictionary of kwargs for add_texts
            add_texts_kwargs: Dict[str, Any] = {"corpus_key": active_corpus_key}

            if doc_metadata is not None:
                add_texts_kwargs["doc_metadata"] = doc_metadata

            if doc_type is not None:
                add_texts_kwargs["doc_type"] = doc_type

            doc_ids = self.vectorstore.add_texts(
                texts=texts, metadatas=metadatas, ids=ids, **add_texts_kwargs
            )

            return (
                f"Successfully ingested {len(doc_ids)} documents into Vectara "
                f"corpus {active_corpus_key} with IDs: {', '.join(doc_ids)}"
            )
        except Exception as e:
            return f"Error ingesting documents to Vectara: {str(e)}"


class VectaraAddFilesInput(BaseModel):
    """Input for the Vectara add files tool."""

    files: List[File] = Field(description="List of File objects to upload to Vectara")
    corpus_key: Optional[str] = Field(
        default=None, description="Corpus key where files will be uploaded"
    )

    model_config = {"arbitrary_types_allowed": True}


class VectaraAddFiles(BaseVectorStoreTool, BaseTool):
    """Tool for uploading files to the Vectara platform.

    Example:
        .. code-block:: python

            from langchain_vectara.tools import VectaraAddFiles
            from langchain_vectara.vectorstores import File, TableExtractionConfig,
            ChunkingStrategy
            from langchain_vectara import Vectara  # Import from langchain-vectara

            # Initialize the Vectara vectorstore
            vectara = Vectara(
                vectara_api_key="your-api-key"
            )

            # Create the tool
            tool = VectaraAddFiles(
                name="vectara_add_files",
                description="Upload files to the Vectara corpus",
                vectorstore=vectara,
                corpus_key="your-corpus-key"  # Required for file upload
            )

            # Prepare file objects
            file1 = File(
                file_path="/path/to/file1.pdf",
                metadata={"source": "file1"},
                table_extraction_config=TableExtractionConfig(extract_tables=True),
                chunking_strategy=ChunkingStrategy(max_chars_per_chunk=1000)
            )
            file2 = File(
                file_path="/path/to/file2.docx",
                metadata={"source": "file2"}
            )

            # Use the tool
            result = tool.run({
                "files": [file1, file2]
            })
    """

    name: str = "vectara_add_files"
    description: str = (
        "Upload files to your Vectara corpus for semantic search. "
        "Supports various file formats including PDFs, DOC, DOCX, TXT, HTML, and more. "
        "Files will be processed automatically with text and metadata extraction. "
        "Input should be a list of File objects with file_path and optional metadata, "
        "chunking_strategy, and table_extraction_config."
    )
    args_schema: Type[BaseModel] = VectaraAddFilesInput
    vectorstore: Vectara

    # Required corpus_key for file upload
    corpus_key: str = Field(
        ...,
        description="Corpus key where files will be uploaded",
    )

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def get_description(name: str, description: str) -> str:
        """Get the description for the tool.

        Args:
            name: The name of the Vectara corpus or knowledge base
            description: Additional description of the knowledge base

        Returns:
            A formatted description for the tool
        """
        template: str = (
            "Useful for when you need to upload files to {name} for "
            "semantic search. "
            "Supports PDFs, DOC, DOCX, TXT, HTML and more. "
            "Documents are automatically processed with text extraction and chunking. "
            "Whenever you need to add files about {description} "
            "you should use this. "
            "Input should be File objects with file paths and optional settings."
        )
        return template.format(name=name, description=description)

    def _run(
        self,
        files: List[File],
        corpus_key: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the Vectara add files.

        Args:
            files: List of File objects to upload
            corpus_key: Optional corpus key override
            run_manager: Optional callback manager

        Returns:
            String describing the upload result
        """
        try:
            active_corpus_key = corpus_key or self.corpus_key

            if not active_corpus_key:
                return "Error: corpus_key is required for file upload"

            doc_ids = self.vectorstore.add_files(
                files_list=files,
                corpus_key=active_corpus_key,
            )

            if not doc_ids:
                return "No files were successfully uploaded."

            return (
                f"Successfully uploaded {len(doc_ids)} files to Vectara "
                f"corpus {active_corpus_key} with IDs: {', '.join(doc_ids)}"
            )
        except Exception as e:
            return f"Error uploading files to Vectara: {str(e)}"
