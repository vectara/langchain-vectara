"""Integration tests for Vectara tools."""

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Type, Union

import pytest
import requests
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_tests.integration_tests.tools import ToolsIntegrationTests

from langchain_vectara import Vectara
from langchain_vectara.tools import (
    VectaraAddFiles,
    VectaraIngest,
    VectaraRAG,
    VectaraSearch,
)
from langchain_vectara.vectorstores import (
    CorpusConfig,
    File,
    GenerationConfig,
    SearchConfig,
    VectaraQueryConfig,
)

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://www.vectara.com/integrations/langchain
# 2. Create a corpus in your Vectara account
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY and VECTARA_CORPUS_KEY
#

test_prompt_name = "vectara-summary-ext-24-05-med-omni"


@pytest.fixture(scope="module")
def vectara() -> Generator[Vectara, None, None]:
    api_key = os.getenv("VECTARA_API_KEY")
    if not api_key:
        pytest.skip("VECTARA_API_KEY environment variable not set")
    vectara_instance = Vectara(vectara_api_key=api_key)

    yield vectara_instance

    cleanup_documents(vectara_instance, os.getenv("VECTARA_CORPUS_KEY"))


def cleanup_documents(vectara: Vectara, corpus_key: Union[str, None]) -> None:
    """
    Fetch all documents from the corpus and delete them after tests are completed.
    """

    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents"
    headers = {"Accept": "application/json", "x-api-key": str(vectara._vectara_api_key)}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return

    data = response.json()
    document_ids = [doc["id"] for doc in data.get("documents", [])]

    if not document_ids:
        return

    vectara.delete(ids=document_ids, corpus_key=corpus_key)


@pytest.fixture(scope="module")
def corpus_key() -> str:
    corpus_key = os.getenv("VECTARA_CORPUS_KEY")
    if not corpus_key:
        pytest.skip("VECTARA_CORPUS_KEY environment variable not set")
    return corpus_key


@pytest.fixture(scope="module")
def add_test_docs(vectara: Vectara, corpus_key: str) -> List[str]:
    """Add test documents for tool tests and return their IDs."""

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog sleeps while the quick brown fox jumps.",
        "A fox is quick and brown in color.",
    ]

    metadatas = [
        {"source": "langchain tool"},
        {"source": "langchain tool"},
        {"source": "langchain tool"},
    ]

    doc_ids = vectara.add_texts(
        texts=texts,
        metadatas=metadatas,
        corpus_key=corpus_key,
        doc_metadata={"test_type": "langchain tool"},
    )

    return doc_ids


@pytest.fixture
def temp_files(tmp_path: Path) -> Dict[str, str]:
    """Fixture to create test files in various formats."""
    # Using a unique identifier for file contents
    unique_id = str(uuid.uuid4())

    file_contents = {
        "test.txt": f"This is a test file about a purple elephant.Test ID: {unique_id}",
        "test.md": f"# Test Markdown\n\nThis document mentions a singing giraffe."
        f"Test ID: {unique_id}",
    }

    created_files = {}
    for filename, content in file_contents.items():
        file_path = tmp_path / filename
        file_path.write_text(content)
        created_files[filename] = str(file_path)

    return created_files


def test_vectara_search_tool(
    vectara: Vectara, corpus_key: str, add_test_docs: List[str]
) -> None:
    """Test the VectaraSearch tool functionality."""

    search_tool = VectaraSearch(
        name="animal_search",
        description="Search for information about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
    )

    result = search_tool.run("fox and dog")
    result_obj = eval(result)
    assert isinstance(result_obj, list)
    assert len(result_obj) > 0

    # Search with config
    search_config = SearchConfig(
        corpora=[
            CorpusConfig(
                corpus_key=corpus_key,
            )
        ],
        limit=1,
    )

    search_tool = VectaraSearch(
        name="animal_search",
        description="Search for information about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
        search_config=search_config,
    )

    result_with_config = search_tool.run({"query": "What animal is mentioned?"})
    result_obj_config = eval(result_with_config)

    assert isinstance(result_obj_config, list)
    assert len(result_obj_config) == 1


def test_vectara_rag_tool(
    vectara: Vectara, corpus_key: str, add_test_docs: List[str]
) -> None:
    """Test the VectaraRAG tool functionality."""
    # Initialize RAG tool
    rag_tool = VectaraRAG(
        name="animal_knowledge",
        description="Get information about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
    )

    # Basic RAG query targeting our test documents specifically
    result = rag_tool.run("What color is the fox mentioned?")

    assert isinstance(result, str)

    # RAG query with custom config
    generation_config = GenerationConfig(
        max_used_search_results=2,
        response_language="eng",
        generation_preset_name=test_prompt_name,
    )

    custom_config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        generation=generation_config,
    )

    rag_tool = VectaraRAG(
        name="animal_knowledge",
        description="Get information about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
        config=custom_config,
    )

    result_with_config = rag_tool.run(
        {"query": "Describe the animals mentioned in the documents."}
    )

    assert isinstance(result_with_config, str)


def test_vectara_ingest_tool(vectara: Vectara, corpus_key: str) -> None:
    """Test the VectaraIngest tool functionality."""
    # Initialize tool
    ingest_tool = VectaraIngest(
        name="ingest_tool",
        description="Add new documents about planets",
        vectorstore=vectara,
        corpus_key=corpus_key,
    )

    # Test ingest functionality
    texts = ["Mars is a red planet.", "Venus has a thick atmosphere."]

    metadatas = [{"source": "planet Mars"}, {"source": "planet Venus"}]

    result = ingest_tool.run(
        {
            "texts": texts,
            "metadatas": metadatas,
            "doc_metadata": {"test_case": "langchain tool"},
        }
    )

    assert "Successfully ingested" in result

    # Verify ingestion with a search
    search_tool = VectaraSearch(
        name="verify_ingest",
        description="Verify ingested documents",
        vectorstore=vectara,
        corpus_key=corpus_key,
        search_config=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
    )

    result = search_tool.run({"query": "What planet?"})
    result_obj = eval(result)

    assert len(result_obj) > 0


def test_vectara_add_files_tool(
    vectara: Vectara, corpus_key: str, temp_files: Dict[str, str]
) -> None:
    """Test the VectaraAddFiles tool functionality."""

    add_files_tool = VectaraAddFiles(
        name="add_files_tool",
        description="Upload files about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
    )

    file_obj1 = File(
        file_path=temp_files["test.txt"], metadata={"source": "langchain tool"}
    )

    file_obj2 = File(
        file_path=temp_files["test.md"], metadata={"source": "langchain tool"}
    )

    result = add_files_tool.run({"files": [file_obj1, file_obj2]})

    assert "Successfully uploaded" in result


def test_vectara_tools_chain_integration(
    vectara: Vectara, corpus_key: str, add_test_docs: List[str]
) -> None:
    """Test Vectara tools in a real chain with an actual LLM if available."""
    # Skip test if real chain integration is not possible
    try:
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable not set")
    except ImportError:
        pytest.skip("langchain_openai not installed")

    # Initialize real LLM
    llm = ChatOpenAI(temperature=0)

    # Initialize search tool
    search_tool = VectaraSearch(
        name="animal_search",
        description="Search for information about animals",
        vectorstore=vectara,
        corpus_key=corpus_key,
        search_config=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
    )

    # Create a real chain
    template = """
    Answer the following question based on the search results:
    
    Question: {question}
    
    Search Results: {search_results}
    
    Answer the question concisely based on the information provided.
    """
    prompt = ChatPromptTemplate.from_template(template)

    def search(query: str) -> str:
        return search_tool.run({"query": f"{query}"})

    chain: RunnableSerializable[Any, str] = (
        {"question": lambda x: x, "search_results": search}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the chain with a real LLM
    result = chain.invoke("What color is the fox?")

    assert isinstance(result, str)
    assert len(result) > 0


def test_vectara_react_agent(
    vectara: Vectara, corpus_key: str, add_test_docs: List[str]
) -> None:
    """Test VectaraRAG tool in a ReAct agent framework with real API calls."""
    # Skip test if OpenAI integration is not possible
    try:
        from langchain_openai import ChatOpenAI

        # Check if OPENAI_API_KEY is set, otherwise use a placeholder for testing
        if not os.environ.get("OPENAI_API_KEY"):
            # Try to load from .env file directly as a fallback
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            if os.path.exists(env_path):
                try:
                    from dotenv import load_dotenv

                    load_dotenv(env_path, override=True)
                except ImportError:
                    pass

            # If still not available, skip the test
            if not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable not set")
    except ImportError:
        pytest.skip("langchain_openai not installed")

    api_auth_docs = [
        """API keys are long strings that act as a simple but effective form of
        authentication for APIs. They are passed with each request, usually in
        a header, and identify the calling project. API keys are easy to use but
        can be insecure if leaked.""",
        """JWT (JSON Web Tokens) are a more secure form of authentication that contain
        encoded JSON data. They can include user identity, expiration times, and
        other claims. JWTs are signed to ensure they haven't been altered and are
        verified on the server. They are ideal for session management and authorization.
        """,
        """When deciding between API keys and JWT tokens, consider your security needs.
        Use API keys for simple internal services or when you need minimal setup.
        Use JWT tokens for user authentication, when you need to store session data,
        or when working with distributed systems that need secure token validation.""",
    ]

    # Add the documents to Vectara
    vectara.add_texts(
        texts=api_auth_docs,
        corpus_key=corpus_key,
    )

    # Allow time for indexing
    time.sleep(3)

    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)],
        ),
        generation=GenerationConfig(
            max_used_search_results=5,
            response_language="eng",
            generation_preset_name="vectara-summary-ext-24-05-med-omni",
        ),
    )

    vectara_rag_tool = VectaraRAG(
        name="api_auth_tool",
        description="Get answers about API authentication methods, JWT tokens, "
        "and security best practices",
        vectorstore=vectara,
        corpus_key=corpus_key,
        config=config,
    )

    # Set up the agent with the VectaraRAG tool
    tools = [vectara_rag_tool]
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question as best you can. 

    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    (This Thought/Action/Action Input/Observation can repeat N times)

    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    Question: {input}
    Thought:{agent_scratchpad}
    """)

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    result = agent_executor.invoke(
        {
            "input": "What is an API key? What is a JWT token? When should I use one or"
            "the other?"
        }
    )

    assert result["output"] is not None


# Standard integration test implementation for VectaraSearch
class TestVectaraSearchIntegration(ToolsIntegrationTests):
    """Test VectaraSearch with standard integration test pattern."""

    @property
    def tool_constructor(self) -> Type[VectaraSearch]:
        """Return the constructor for VectaraSearch."""
        return VectaraSearch

    @property
    def tool_constructor_params(self) -> Dict:
        """Return arguments needed to construct the tool."""

        api_key = os.environ.get("VECTARA_API_KEY")
        vectara_instance = Vectara(vectara_api_key=api_key)

        corpus_key = os.environ.get("VECTARA_CORPUS_KEY")

        # Create a search config for the tool
        search_config = SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)])

        return {
            "name": "standard_search",
            "description": "Standard search tool",
            "vectorstore": vectara_instance,
            "corpus_key": corpus_key,
            "search_config": search_config,  # Pass config at initialization
        }

    @property
    def tool_invoke_params_example(self) -> Dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        For Vectara tools, this must include a "query" parameter.
        """
        return {"query": "fox animal"}


# Standard integration test implementation for VectaraRAG
class TestVectaraRAGIntegration(ToolsIntegrationTests):
    """Test VectaraRAG with standard integration test pattern."""

    @property
    def tool_constructor(self) -> Type[VectaraRAG]:
        """Return the constructor for VectaraRAG."""
        return VectaraRAG

    @property
    def tool_constructor_params(self) -> Dict:
        """Return arguments needed to construct the tool."""

        api_key = os.environ.get("VECTARA_API_KEY")
        vectara_instance = Vectara(vectara_api_key=api_key)

        corpus_key = os.environ.get("VECTARA_CORPUS_KEY")

        # Create a query config for the tool
        query_config = VectaraQueryConfig(
            search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
            generation=GenerationConfig(
                max_used_search_results=5,
                response_language="eng",
            ),
        )

        return {
            "name": "standard_rag",
            "description": "Standard RAG tool",
            "vectorstore": vectara_instance,
            "corpus_key": corpus_key,
            "config": query_config,  # Pass config at initialization
        }

    @property
    def tool_invoke_params_example(self) -> Dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        For Vectara tools, this must include a "query" parameter.
        """
        return {"query": "What color is the fox?"}


# Standard integration test implementation for VectaraIngest
class TestVectaraIngestIntegration(ToolsIntegrationTests):
    """Test VectaraIngest with standard integration test pattern."""

    @property
    def tool_constructor(self) -> Type[VectaraIngest]:
        """Return the constructor for VectaraIngest."""
        return VectaraIngest

    @property
    def tool_constructor_params(self) -> Dict:
        """Return arguments needed to construct the tool."""

        api_key = os.environ.get("VECTARA_API_KEY")
        vectara_instance = Vectara(vectara_api_key=api_key)

        corpus_key = os.environ.get("VECTARA_CORPUS_KEY")

        return {
            "name": "standard_ingest",
            "description": "Standard ingest tool",
            "vectorstore": vectara_instance,
            "corpus_key": corpus_key,  # Required for ingestion
        }

    @property
    def tool_invoke_params_example(self) -> Dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        For VectaraIngest tool, this must include "texts" parameter.
        """
        return {
            "texts": ["Sample document for ingest test"],
            "metadatas": [{"source": "integration test"}],
            "doc_metadata": {"test_type": "standard_integration"},
        }


# Standard integration test implementation for VectaraAddFiles
class TestVectaraAddFilesIntegration(ToolsIntegrationTests):
    """Test VectaraAddFiles with standard integration test pattern."""

    @property
    def tool_constructor(self) -> Type[VectaraAddFiles]:
        """Return the constructor for VectaraAddFiles."""
        return VectaraAddFiles

    @property
    def tool_constructor_params(self) -> Dict:
        """Return arguments needed to construct the tool."""

        api_key = os.environ.get("VECTARA_API_KEY")
        vectara_instance = Vectara(vectara_api_key=api_key)

        corpus_key = os.environ.get("VECTARA_CORPUS_KEY")

        return {
            "name": "standard_add_files",
            "description": "Standard file upload tool",
            "vectorstore": vectara_instance,
            "corpus_key": corpus_key,  # Required for file upload
        }

    @property
    def tool_invoke_params_example(self) -> Dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        For VectaraAddFiles tool, this must include "files" parameter.
        Note: For standard tests, we don't actually upload files but provide
        a valid structure to pass schema validation.
        """
        # Create a temporary file path for the example
        # This doesn't need to be a real file for the standard integration test
        sample_file = File(
            file_path="/tmp/example.txt", metadata={"source": "integration test"}
        )

        return {"files": [sample_file]}
