import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.api.chatflow.handler import handle_chatflow
from src.api.chatflow.state import ChatflowState
from src.database.db import get_db
from src.database.models import Interaction
from src.shared.schemas import (
    InteractionRequest,
    InteractionResponse,
    InteractionMessage,
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
    session_id = interaction_request.sessionId
    user_message = interaction_request.message

    # Find existing interaction
    result = await db.execute(
        select(Interaction).where(Interaction.session_id == session_id)
    )
    interaction = result.scalar_one_or_none()

    if interaction:
        history_messages = [
            InteractionMessage.model_validate(msg) for msg in interaction.messages
        ]
        # Get the last state from the list
        last_state = interaction.states[-1] if interaction.states else None
        current_state = (
            ChatflowState(last_state)
            if last_state
            else ChatflowState.CLASSIFYING_INTENT
        )
        interaction_data = interaction.interaction_data or {}
        user_data = interaction.user_data or {}
    else:
        # New interaction
        history_messages = []
        current_state = ChatflowState.CLASSIFYING_INTENT
        interaction_data = {}
        user_data = {}
        interaction = Interaction(
            session_id=session_id,
            messages=[],
            states=[current_state.value],
            interaction_data=interaction_data,
            user_data=user_data,
        )
        db.add(interaction)

    # Append new user message to history
    history_messages.append(user_message)

    if interaction_request.userData:
        user_data.update(interaction_request.userData)
        interaction.user_data = user_data

    client = request.app.state.genai_client
    sheets_service = request.app.state.sheets_service

    response_messages, new_state, tool_call, interaction_data = await handle_chatflow(
        session_id=session_id,
        history_messages=history_messages,
        current_state=current_state,
        interaction_data=interaction_data,
        client=client,
    )

    # Update history with new messages from the handler
    history_messages.extend(response_messages)

    # Persist changes
    interaction.messages = [
        msg.model_dump(mode="json", exclude_none=True) for msg in history_messages
    ]
    # Append the new state. To ensure SQLAlchemy detects the change, we re-assign the list.
    interaction.states = interaction.states + [new_state.value]
    interaction.interaction_data = interaction_data

    await db.commit()
    await db.refresh(interaction)

    return InteractionResponse(
        sessionId=session_id,
        messages=response_messages,
        toolCall=tool_call,
        state=new_state.value,
    )