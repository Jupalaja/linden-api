import logging
from typing import Optional, Any

import google.genai as genai

from src.shared.state import GlobalState
from .state import ChatflowState
from .workflows import run_chatflow_workflow
from src.services.google_sheets import GoogleSheetsService
from src.shared.schemas import InteractionMessage

logger = logging.getLogger(__name__)


async def handle_chatflow(
    session_id: str,
    history_messages: list[InteractionMessage],
    current_state: ChatflowState,
    interaction_data: Optional[dict],
    client: genai.Client,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], GlobalState, str | None, dict] | tuple[
    list[InteractionMessage], Any, str | None, dict
]:
    interaction_data = dict(interaction_data) if interaction_data else {}

    return await run_chatflow_workflow(
        session_id=session_id,
        history_messages=history_messages,
        current_state=current_state,
        interaction_data=interaction_data,
        client=client,
        sheets_service=sheets_service,
    )
