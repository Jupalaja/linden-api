import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.db import get_db
from src.shared.schemas import (
    InteractionRequest,
    InteractionResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chatflow", response_model=InteractionResponse)
async def handle(
    interaction_request: InteractionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Handles a user-assistant interaction for the chatflow operation,
    continuing a conversation by loading history from the database,
    appending the new message, and saving the updated history.
    """
    # TODO: Build handle function