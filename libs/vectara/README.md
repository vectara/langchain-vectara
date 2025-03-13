# langchain-vectara

This package contains the LangChain integration with Vectara.

# Vectara

>[Vectara](https://vectara.com/) provides a Trusted Generative AI platform, allowing organizations to rapidly create a ChatGPT-like experience (an AI assistant) 
> which is grounded in the data, documents, and knowledge that they have (technically, it is Retrieval-Augmented-Generation-as-a-service).

**Vectara Overview:**
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
    texts=["to be or not to be", "that is the question"],
    metadatas=[{"title": "test"}, {"title": "test"}],
    corpus_key="your-corpus-key",
)

```

Since Vectara supports file-upload in the platform, we also added the ability to upload files (PDF, TXT, HTML, PPT, DOC, etc) directly. 
When using this method, each file is uploaded directly to the Vectara backend, processed and chunked optimally there, so you don't have to use the LangChain document loader or chunking mechanism.

As an example:

```python
from langchain_vectara.vectorstores import File

file_objects = [
    File(file_path="path/to/file1.pdf", metadata={"doc_type": "pdf"}),
    File(file_path="path/to/file2.docx", metadata={"doc_type": "word"}),
]

doc_ids = vectara.add_files(file_objects, corpus_key="your-corpus-key")
```

Of course you do not have to add any data, and instead just connect to an existing Vectara corpus where data may already be indexed.

### Querying the VectorStore

To query the Vectara vectorstore, you can use the `similarity_search` method (or `similarity_search_with_score`), which takes a query string and returns a list of results:
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

## Vectara for Retrieval Augmented Generation (RAG)

Vectara provides a full RAG pipeline, including generative summarization. To use it as a complete RAG solution, you can use the `as_rag` method.
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

## Hallucination Detection score

Vectara created [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model) - an open source model that can be used to evaluate RAG responses for factual consistency. 
As part of the Vectara RAG, the "Factual Consistency Score" (or FCS), which is an improved version of the open source HHEM is made available via the API. 
This is automatically included in the output of the RAG pipeline

```python
rag = vectara.as_rag(config)
resp = rag.invoke(query_str)
print(resp['answer'])
print(f"Vectara FCS = {resp['fcs']}")
```