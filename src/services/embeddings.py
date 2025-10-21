import logging
import regex
from typing import Any, Dict, Optional

from firecrawl import Firecrawl
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.services.vector_store import get_vector_store
from src.shared.constants import (
    INVALID_UNICODE_CLEANUP_REGEX,
    VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD,
    VECTOR_EMBEDDINGS_QUERY_SYSTEM_PROMPT
)
from src.shared.enums import SourceType

logger = logging.getLogger(__name__)


def store_data_from_website(website: str, practice_id: str):
    """
    Scrapes a website and stores its content in Chroma.
    """
    if not settings.FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY not found in settings")

    vector_store = get_vector_store()
    firecrawl = Firecrawl(
        api_key=settings.FIRECRAWL_API_KEY,
    )

    print(f"Scraping {website}...")
    scraped_website = firecrawl.scrape(
        url=website,
        formats=["markdown"],
        exclude_tags=
            ["script", "style", "img", "a", "source", "track", "embed", "base", "col", "area", "form", "input"],
    )
    output_markdown = scraped_website.markdown
    if not output_markdown:
        print(f"Warning: No markdown content scraped from {website}. Skipping.")
        return
        
    cleaned_markdown = regex.sub(INVALID_UNICODE_CLEANUP_REGEX, '', output_markdown)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
    )
    docs = text_splitter.create_documents([cleaned_markdown])

    sanitized_url = regex.sub(r'[^a-zA-Z0-9]', '_', website)
    ids = []
    for i, doc in enumerate(docs):
        doc_id = f"{sanitized_url}_{i}"
        doc.metadata["doc_id"] = doc_id
        doc.metadata["practice_id"] = practice_id
        doc.metadata["source_type"] = SourceType.WEB_PAGE.value
        doc.metadata["source_page_title"] = getattr(scraped_website.metadata, 'title', 'No Title')
        doc.metadata["source_url"] = website
        ids.append(doc_id)

    # Check for existing documents to avoid duplication
    existing_docs_result = vector_store.get(ids=ids, include=[])
    existing_ids = set(existing_docs_result["ids"])

    docs_to_add = []
    ids_to_add = []
    docs_to_update = []
    ids_to_update = []
    for i, doc_id in enumerate(ids):
        if doc_id not in existing_ids:
            docs_to_add.append(docs[i])
            ids_to_add.append(doc_id)
        else:
            docs_to_update.append(docs[i])
            ids_to_update.append(doc_id)

    if docs_to_add:
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        print(
            f"Successfully added {len(docs_to_add)} new chunks from {website} to the collection."
        )
    if docs_to_update:
        vector_store.update_documents(documents=docs_to_update, ids=ids_to_update)
        print(
            f"Successfully updated {len(docs_to_update)} new chunks from {website} to the collection."
        )

    if not docs_to_add and not docs_to_update:
        print(
            f"All documents from {website} are already in the collection. No new documents to add."
        )


def retrieve_data(query: str, practice_id: str, filters: Optional[Dict[str, Any]] = None) -> str:
    """
    Retrieves data from the vector store based on a query and optional filters,
    and generates a response using an LLM.

    Args:
        query: The user's question.
        practice_id: The practice ID to filter the search results.
        filters: A dictionary of metadata to filter the search results.

    Returns:
        The content of the model's response.
    """
    vector_store = get_vector_store()

    search_filters = filters.copy() if filters else {}
    search_filters["practice_id"] = practice_id

    # Perform a similarity search with optional filters
    results_with_scores = vector_store.similarity_search_with_score(
        query=query, k=3, filter=search_filters
    )

    if not results_with_scores:
        logger.warning(f"No results found for query: '{query}' with filters: {search_filters}")
        return "No relevant information was found to answer your question."

    filtered_results_with_scores = [
        (doc, score) for doc, score in results_with_scores if score < VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD
    ]

    logger.info(f"Found {len(filtered_results_with_scores)} results for query: '{query}'")
    for doc, score in filtered_results_with_scores:
        doc_id = doc.metadata.get('doc_id', 'N/A')
        logger.info(f"  - Document ID: {doc_id}, Score (distance): {score:.4f}")

    results = [
        doc for doc, _ in filtered_results_with_scores
    ]

    if not results:
        logger.warning(
            f"No results found within similarity threshold ({VECTOR_EMBEDDINGS_SIMILARITY_THRESHOLD}) for query: '{query}'"
        )
        return "No relevant information was found to answer your question."

    context = "\n---\n".join([doc.page_content for doc in results])
    prompt = ChatPromptTemplate.from_template(VECTOR_EMBEDDINGS_QUERY_SYSTEM_PROMPT)
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0,
    )

    chain = prompt | model

    response = chain.invoke({"context": context, "question": query})

    return response.content
