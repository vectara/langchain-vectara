# langchain-vectara

This package contains the LangChain integration with Vectara.

# Vectara

[Vectara](https://vectara.com/) is the trusted AI Assistant and Agent platform which focuses on enterprise readiness for mission-critical applications.
Vectara serverless RAG-as-a-service provides all the components of RAG behind an easy-to-use API, including:
1. A way to extract text from files (PDF, PPT, DOCX, etc)
2. ML-based chunking that provides state of the art performance.
3. The [Boomerang](https://vectara.com/how-boomerang-takes-retrieval-augmented-generation-to-the-next-level-via-grounded-generation/) embeddings model.
4. Its own internal vector database where text chunks and embedding vectors are stored.
5. A query service that automatically encodes the query into embedding, and retrieves the most relevant text segments, including support for [Hybrid Search](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) as well as multiple reranking options such as the [multi-lingual relevance reranker](https://www.vectara.com/blog/deep-dive-into-vectara-multilingual-reranker-v1-state-of-the-art-reranker-across-100-languages), [MMR](https://vectara.com/get-diverse-results-and-comprehensive-summaries-with-vectaras-mmr-reranker/), [UDF reranker](https://www.vectara.com/blog/rag-with-user-defined-functions-based-reranking). 
6. An LLM to for creating a [generative summary](https://docs.vectara.com/docs/learn/grounded-generation/grounded-generation-overview), based on the retrieved documents (context), including citations.

For more information:
- [Documentation](https://docs.vectara.com/docs/)
- [API Playground](https://docs.vectara.com/docs/rest-api/)
- [Quickstart](https://docs.vectara.com/docs/quickstart)

## Installation and Setup

```bash
pip install -U langchain-vectara
```

To get started, [sign up](https://vectara.com/integrations/langchain) for a free Vectara trial,
and follow the [quickstart](https://docs.vectara.com/docs/quickstart) guide to create a corpus and an API key. 
Once you have API Key, you can provide it as an argument to the Vectara `vectorstore`, or you can set it as an environment variable..

```bash
- export `VECTARA_API_KEY`="your-vectara-api-key"
```

## Vectara as a Vector Store

There exists a wrapper around the Vectara platform, allowing you to use it as a `vectorstore` in LangChain:

To import this vectorstore:
```python
from langchain_vectara import Vectara
```

To create an instance of the Vectara vectorstore:
```python
vectara = Vectara(
    vectara_api_key=api_key
)
```
The `api_key` is optional, and if its not supplied will be read from the environment variables.

### Indexing Data

After you have the vectorstore, you can use `add_texts` or `add_documents` as per the standard `VectorStore` interface, for example:


```python
doc_ids = vectara.add_texts(
    texts=["A bunch of scientists bring back dinosaurs and mayhem breaks loose", "Leo DiCaprio gets lost in a dream within a dream within a dream within a ..."],
    metadatas=[{"year": 1993, "rating": 7.7, "genre": "science fiction"}, {"year": 2010, "director": "Christopher Nolan", "rating": 8.2}],
    corpus_key="your-corpus-key",
)

```

Since Vectara supports file-upload in the platform, we also added the ability to upload files (PDF, TXT, HTML, PPT, DOC, etc) directly. 
When using this method, each file is uploaded directly to the Vectara backend, processed and chunked optimally there, so you don't have to use the LangChain document loader or chunking mechanism.

As an example:

You can specify whether to extract table data from any uploaded PDF file. If you do not set this option, the platform does not extract tables from PDF.

```python
from langchain_vectara.vectorstores import File, TableExtractionConfig

file_objects = [
    File(file_path="path/to/file1.pdf", metadata={"doc_type": "pdf"}, table_extraction_config=TableExtractionConfig(extract_tables=True)),
    File(file_path="path/to/file2.docx", metadata={"doc_type": "word"}),
]

doc_ids = vectara.add_files(file_objects, corpus_key="your-corpus-key")
```

## Vectara for Retrieval Augmented Generation (RAG)

Vectara is an end-to-end Generative AI platform that implements a complete RAG stack, including text extraction, chunking, embedding, vector store, retrieval, and response generation using an LLM.
There is no RAG-stack abstraction in LangChain, therefore the Vectara integration is provided via the  vectorstore class.

To use it as a complete RAG solution, you can use the `as_rag` method.
Following are the parameters that can be specified in the [`VectaraQueryConfig`](https://docs.vectara.com/docs/rest-api/query) object to control retrieval and summarization:
* search: Config for search results to return
    - corpora:  List of corpora to search within. Vectara supports searching within a single corpus or multiple corpora.
    - offset: Number of results to skip, useful for pagination.
    - limit: Maximum number of search results to return.
    - context_configuration: Context settings for search results.
    - reranker: Reranker to refine search results.
* generation: can be used to request an LLM summary in RAG
    - max_used_search_results: The maximum number of search results to be available to the prompt.
    - response_language: requested language for the summary
    - generation_preset_name: name of the prompt to use for summarization (see https://docs.vectara.com/docs/learn/grounded-generation/select-a-summarizer)
    - enable_factual_consistency_score: Score based on the HHEM that indicates the factual accuracy of the summary


For example:

```python
from langchain_vectara.vectorstores import (
    VectaraQueryConfig,
    SearchConfig,
    CorpusConfig,
    CustomerSpecificReranker,
    GenerationConfig,
)

search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key="your-corpus-key")],
    reranker=CustomerSpecificReranker(reranker_id="rnk_272725719", limit=100),
    limit=25
)

generation_config = GenerationConfig(
    max_used_search_results=7,
    response_language="eng",
    enable_factual_consistency_score=True,
)

config = VectaraQueryConfig(
    search=search_config,
    generation=generation_config,
)

```
Then you can use the `as_rag` method to create a RAG pipeline:

```python
query_str = "what did Biden say?"

rag = vectara.as_rag(config)
rag.invoke(query_str)['answer']
```

For streaming:

```python
output = {}
for chunk in rag.stream("what did he said about the covid?"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key == "answer":
            print(chunk[key], end="", flush=True)
```

The `as_rag` method returns a `VectaraRAG` object, which behaves just like any LangChain Runnable, including the `invoke` or `stream` methods.


### Hallucination Detection score

Vectara created [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) - an open source model that can be used to evaluate RAG responses for factual consistency. 
As part of Vectara's RAG pipeline, the "Factual Consistency Score" (or FCS) is automatically returned with every query. This calibrated score can range from 0.0 to 1.0. A higher score indicates a higher confidence that the summary is factually consistent, while a lower score indicates possible hallucinations.

```python
rag = vectara.as_rag(config)
resp = rag.invoke(query_str)
print(resp['answer'])
print(f"Vectara FCS = {resp['fcs']}")
```


## Vectara Chat

The RAG functionality can be used to create a chatbot with multi-turn question/response turns in a conversation. In this case, the full chat history is also maintained and used as needed by the Vectara platform.

For example:

```python
query_str = "what did Biden say?"
bot = vectara.as_chat(config)
bot.invoke(query_str)['answer']
```

The main difference is the following: with `as_chat` Vectara internally tracks the chat history and conditions each response on the full chat history.
There is no need to keep that history locally to LangChain, as Vectara will manage it internally.

## Vectara as a LangChain retriever only

If you want to use Vectara as a retriever only, you can use the `as_retriever` method, which returns a `VectaraRetriever` object.
```python
retriever = vectara.as_retriever(config=config)
retriever.invoke(query_str)
```

Like with as_rag, you provide a `VectaraQueryConfig` object to control the retrieval parameters.
If no summary is requested, the response will be a list of relevant documents, each with a relevance score.
If a summary is requested, the response will be a list of relevant documents as before, plus an additional document that includes the generative summary.

Of course you do not have to add any data, and instead just connect to an existing Vectara corpus where data may already be indexed.

### Querying the VectorStore

For the semantic search query the Vectara vectorstore using `similarity_search` method (or `similarity_search_with_score`), which takes a query string and returns a list of results:
```python
from langchain_vectara.vectorstores import (
    SearchConfig,
    CorpusConfig
    )

results_with_score = vectara.similarity_search_with_score(
    "what is LangChain?",
    search=SearchConfig(
        corpora=[CorpusConfig(corpus_key="your-corpus-key")])
)
```
The results are returned as a list of relevant documents, and a relevance score of each document.

In this case, we used the default retrieval parameters, but you can also specify the following additional arguments in the `SearchConfig` and `CorpusConfig`:
- `limit`: number of results to return (defaults to 10)
- `lexical_interpolation`: the [lexical matching](https://docs.vectara.com/docs/api-reference/search-apis/lexical-matching) factor for hybrid search (defaults to 0.0)
- `filter`: a [metadata filter](https://docs.vectara.com/docs/common-use-cases/filtering-by-metadata/filter-overview) to apply to the results (default None)
- `context_configuration`: [Context configuration](https://docs.vectara.com/docs/api-reference/search-apis/interpreting-responses/highlighting) settings for search results.
- `reranker`: Reranker to refine search results. Vectara has multiple [reranker](https://docs.vectara.com/docs/api-reference/search-apis/reranking)
   - Multilingual Reranker v1/CustomerSpecific
   - Maximal Marginal Relevance (MMR) reranker
   - User Defined Function reranker
   - Chain reranker

To get results without the relevance score, you can simply use the 'similarity_search' method:
```python   
results = vectara.similarity_search("what is LangChain?")
```

## Intelligent Query Rewriting
Intelligent Query Rewriting enhances search precision by automatically generating metadata filter expressions from natural language queries. This capability analyzes user queries, extracts relevant metadata filters, and rephrases the query to focus on the core information need. For more [details](https://docs.vectara.com/docs/search-and-retrieval/intelligent-query-rewriting).

Enable intelligent query rewriting on a per-query basis by setting the `intelligent_query_rewriting` parameter to `true` in `VectaraQueryConfig`.

### Example Usage
Consider a corpus containing movie data with the metadata filter attribute `doc.production_country` (Text).

Example user query: What are some of the highest grossing movies made in US, UK, or India?

#### Intelligent Query Rewriting processes this by

  - Extracting metadata filters: doc.production_country IN ('United States of America', 'United Kingdom', 'India')
  - Rephrasing the query to remove filter context: What are some of the highest grossing movies?

```python
from langchain_vectara.vectorstores import (
    VectaraQueryConfig,
    SearchConfig,
    CorpusConfig,
)

search_config = SearchConfig(
    corpora=[CorpusConfig(corpus_key="your-corpus-key")],
)

config = VectaraQueryConfig(
    search=search_config,
    intelligent_query_rewriting=True
)
```