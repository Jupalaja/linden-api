import logging
from fastapi import APIRouter, HTTPException, status

from src.services.embeddings import store_data_from_website
from src.shared.enums import SourceType
from src.shared.schemas import (
    CreateEmbeddingsRequest,
    CreateEmbeddingsResponse,
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
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="webPageURL is required for WEB_PAGE source type")
        try:
            store_data_from_website(request.sourceData.webPageURL, request.practiceId)
            return CreateEmbeddingsResponse(status="success", message="Embeddings created successfully from web page.")
        except Exception as e:
            logger.error(f"Failed to create embeddings from web page {request.sourceData.webPageURL} for practice {request.practiceId}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating embeddings from the web page.")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Source type '{request.sourceType.value}' not supported.")
