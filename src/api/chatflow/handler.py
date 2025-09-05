import logging
from typing import Optional, Any
import google.genai as genai

from .state import ChatflowState
from .workflows import *
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

    # TODO: Build Handler function
