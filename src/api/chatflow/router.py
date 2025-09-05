import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
import google.genai as genai
from google.genai import errors

from .handler import handle_chatflow
from .state import ChatflowState
from src.database import models
from src.database.db import get_db
from src.shared.schemas import (
    InteractionMessage,
    InteractionRequest,
    InteractionResponse,
)
from src.services.google_sheets import GoogleSheetsService

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
    logger.info(
        f"Handling 'chatflow' request for session: {interaction_request.sessionId}"
    )
    client: genai.Client = request.app.state.genai_client
    sheets_service: Optional[GoogleSheetsService] = request.app.state.sheets_service

    interaction = await db.get(models.Interaction, interaction_request.sessionId)

    history_messages: list[InteractionMessage] = []
    current_state = ChatflowState.IDLE
    interaction_data: Optional[dict] = None
    if interaction:
        history_messages = [
            InteractionMessage.model_validate(msg) for msg in interaction.messages
        ]
        if interaction.state:
            try:
                current_state = ChatflowState(interaction.state)
            except ValueError:
                logger.warning(
                    f"Invalid state '{interaction.state}' for chatflow found in DB for session {interaction_request.sessionId}. Resetting to IDLE."
                )
                current_state = ChatflowState.IDLE
        if interaction.interaction_data:
            interaction_data = interaction.interaction_data

    history_messages.append(interaction_request.message)

    try:
        (
            new_assistant_messages,
            next_state,
            tool_call_name,
            new_interaction_data,
        ) = await handle_chatflow(
            session_id=interaction_request.sessionId,
            history_messages=history_messages,
            current_state=current_state,
            interaction_data=interaction_data,
            client=client,
            sheets_service=sheets_service,
        )

        history_messages.extend(new_assistant_messages)

        if interaction:
            interaction.messages = [msg.model_dump(mode="json") for msg in history_messages]
            interaction.state = next_state.value
            interaction.interaction_data = new_interaction_data
        else:
            interaction = models.Interaction(
                session_id=interaction_request.sessionId,
                messages=[msg.model_dump(mode="json") for msg in history_messages],
                state=next_state.value,
                interaction_data=new_interaction_data,
            )
            db.add(interaction)

        await db.commit()

        return InteractionResponse(
            sessionId=interaction_request.sessionId,
            messages=new_assistant_messages,
            toolCall=tool_call_name,
            state=interaction.state,
        )
    except errors.APIError as e:
        logger.error(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e!s}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Check server logs and environment variables.",
        )