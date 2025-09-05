import logging
from typing import Optional, Tuple, List, Callable, Any

import google.genai as genai

from .prompts import *
from .state import ChatflowState
from .tools import *
from src.config import settings
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.state import GlobalState
from src.services.google_sheets import GoogleSheetsService
from src.shared.utils.functions import execute_tool_calls_and_get_response

logger = logging.getLogger(__name__)


async def _write_chatflow_to_sheet(
    interaction_data: dict, sheets_service: Optional[GoogleSheetsService]
):
    if interaction_data.get("sheet_row_added"):
        logger.info(
            "Data for job candidate has already been written to Google Sheet. Skipping."
        )
        return

    if not settings.GOOGLE_SHEET_ID_EXPORT or not sheets_service:
        logger.warning(
            "Spreadsheet ID for export not configured or sheets service not available. Skipping write."
        )
        return

    try:
        worksheet = sheets_service.get_worksheet(
            spreadsheet_id=settings.GOOGLE_SHEET_ID_EXPORT,
            worksheet_name="data",
        )
        if not worksheet:
            logger.error("Could not find data worksheet.")
            return

        row_to_append = []

        sheets_service.append_row(worksheet, row_to_append)
        interaction_data["sheet_row_added"] = True
        logger.info(
            "Successfully wrote data for job candidate to Google Sheet and marked as added."
        )

    except Exception as e:
        logger.error(f"Failed to write to Google Sheet: {e}", exc_info=True)

# TODO: Build rest of the workflow functions