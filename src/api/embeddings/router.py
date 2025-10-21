import logging
from fastapi import APIRouter, HTTPException

from src.services.embeddings import store_data_from_website, retrieve_data
from src.shared.enums import SourceType
from src.shared.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
    QueryEmbeddingsRequest,
    QueryEmbeddingsResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/create-embeddings", response_model=CreateEmbeddingsResponse)
async def create_embeddings(
    request: CreateEmbeddingsRequest,
):
    logger.info(f"Received create-embeddings request: {request.model_dump_json(indent=2)}")

    if request.sourceType == SourceType.WEB_PAGE:
        if not request.sourceData.webPageURL:
            raise HTTPException(status_code=400, detail="webPageURL is required for WEB_PAGE source type")
        try:
            store_data_from_website(request.sourceData.webPageURL)
            return CreateEmbeddingsResponse(status="success", message="Embeddings created successfully from web page.")
        except Exception as e:
            logger.error(f"Failed to create embeddings from web page: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Source type '{request.sourceType.value}' not supported.")


@router.post("/query-embeddings", response_model=QueryEmbeddingsResponse)
async def query_embeddings(
    request: QueryEmbeddingsRequest,
):
    logger.info(f"Received query-embeddings request: {request.model_dump_json(indent=2)}")
    filters = {"source_type": SourceType.WEB_PAGE.value}

    try:
        response_message = retrieve_data(query=request.query, filters=filters)
        return QueryEmbeddingsResponse(status="success", message=response_message)
    except Exception as e:
        logger.error(f"Failed to query embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
