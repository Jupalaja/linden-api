import regex
from firecrawl import Firecrawl
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.services.vector_store import get_vector_store
from src.shared.enums import SourceType

EXCLUDED_CHARS_REGEX_PATTERN = r'[\p{Cf}\p{Cn}\p{Co}\p{Cs}\p{So}]'


def store_data_from_website(website: str):
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
        
    cleaned_markdown = regex.sub(EXCLUDED_CHARS_REGEX_PATTERN, '', output_markdown)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
    )
    docs = text_splitter.create_documents([cleaned_markdown])

    sanitized_url = regex.sub(r'[^a-zA-Z0-9]', '_', website)
    ids = []
    for i, doc in enumerate(docs):
        doc.metadata["source_type"] = SourceType.WEB_PAGE.value
        doc.metadata["source_page_title"] = getattr(scraped_website.metadata, 'title', 'No Title')
        doc.metadata["source_url"] = website
        ids.append(f"{sanitized_url}_{i}")

    # Check for existing documents to avoid duplication
    existing_docs_result = vector_store.get(ids=ids, include=[])
    existing_ids = set(existing_docs_result['ids'])

    docs_to_add = []
    ids_to_add = []
    for i, doc_id in enumerate(ids):
        if doc_id not in existing_ids:
            docs_to_add.append(docs[i])
            ids_to_add.append(doc_id)

    if docs_to_add:
        vector_store.add_documents(documents=docs_to_add, ids=ids_to_add)
        print(f"Successfully added {len(docs_to_add)} new chunks from {website} to the collection.")
    else:
        print(f"All documents from {website} are already in the collection. No new documents to add.")
