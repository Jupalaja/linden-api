import logging
from typing import Optional, Tuple, List, Callable, Any
from datetime import datetime

import google.genai as genai
from google.genai import types, errors

from .prompts import *
from .state import ChatflowState
from .tools import *
from src.config import settings
from src.shared.constants import GEMINI_MODEL
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.state import GlobalState
from src.shared.utils.history import get_genai_history
from src.services.google_sheets import GoogleSheetsService
from src.shared.utils.functions import execute_tool_calls_and_get_response
from src.shared.tools import obtener_ayuda_humana as shared_obtener_ayuda_humana

logger = logging.getLogger(__name__)

CHATFLOW_SYSTEM_PROMPT = """
Eres Linden, el asistente virtual de Aya Naturopathic Medicine.
Tu objetivo es ayudar a los usuarios respondiendo sus preguntas y guiándolos a través de las opciones de atención.
Sé amable, profesional y directo. Usa las herramientas disponibles cuando sea necesario para determinar la intención del usuario y proporcionar la información correcta.
NO menciones el nombre de las herramientas.
"""


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


async def _call_model_with_tools(
    history_messages: list[InteractionMessage],
    client: genai.Client,
    tools: List[Callable],
) -> Tuple[Optional[str], dict, List[str], dict]:
    return await execute_tool_calls_and_get_response(
        history_messages=history_messages,
        client=client,
        tools=tools,
        system_prompt=CHATFLOW_SYSTEM_PROMPT,
    )


async def run_chatflow_workflow(
    session_id: str,
    history_messages: list[InteractionMessage],
    current_state: ChatflowState,
    interaction_data: dict,
    client: genai.Client,
    sheets_service: Optional[GoogleSheetsService],
) -> Tuple[list[InteractionMessage], ChatflowState, Optional[str], dict]:
    assistant_messages: list[InteractionMessage] = []
    tool_call_name: Optional[str] = None
    next_state: Any = current_state
    
    # This loop allows for automatic state transitions without waiting for user input
    for _ in range(5):  # Max 5 internal transitions to prevent infinite loops
        current_state = next_state
        logger.info(f"Processing state: {current_state}")
        
        # --- Terminal/Message states ---
        if current_state == ChatflowState.IDLE:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=LINDEN_INTRODUCTION_MESSAGE))
            next_state = ChatflowState.AWAITING_MESSAGE
            break

        elif current_state == ChatflowState.AWAITING_MESSAGE:
            next_state = ChatflowState.ANALYZING_INTENT
            # Fall through to ANALYZING_INTENT in the same loop iteration

        elif current_state == ChatflowState.INVALID_REQUEST_EMERGENCY:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=OUTPUT_MESSAGE_EMERGENCY))
            next_state = ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE
            break

        elif current_state == ChatflowState.CONDITION_NOT_TREATED_SEND_CONTACT_INFO:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_CONDITION_NOT_TREATED))
            next_state = ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE
            break
            
        elif current_state == ChatflowState.CUSTOMER_IN_NON_VALID_STATE:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_INVALID_STATE))
            next_state = ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE
            break

        elif current_state == ChatflowState.BOOK_CALL_OFFER_DECLINED:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_PROVIDE_CONTACT_INFO))
            next_state = ChatflowState.CONVERSATION_FINISHED
            break

        elif current_state == ChatflowState.CONVERSATION_FINISHED:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_OFFER_NEWSLETTER))
            # Here we expect user to say yes/no. The state remains CONVERSATION_FINISHED
            # until we get a response to analyze for newsletter sign up.
            break

        # --- Tool-based states ---
        elif current_state == ChatflowState.ANALYZING_INTENT:
            tools_to_use = [is_emergency, is_potential_patient, is_question_about_condition, is_question_event, is_frequently_asked_question, is_out_of_scope_question, is_frustrated_needs_human, is_acknowledgment]
            _, tool_results, tool_names, _ = await _call_model_with_tools(history_messages, client, tools_to_use)
            tool_call_name = tool_names[0] if tool_names else None

            if tool_results.get("is_emergency"): next_state = ChatflowState.INVALID_REQUEST_EMERGENCY
            elif tool_results.get("is_potential_patient"): next_state = ChatflowState.INTENT_POTENTIAL_PATIENT
            elif tool_results.get("is_question_about_condition"): next_state = ChatflowState.INTENT_QUESTION_CONDITION
            elif tool_results.get("is_frequently_asked_question"): next_state = ChatflowState.INTENT_FAQ
            elif tool_results.get("is_frustrated_needs_human"): next_state = ChatflowState.INTENT_FRUSTRATED_CUSTOMER
            elif tool_results.get("is_acknowledgment"): next_state = ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE
            elif tool_results.get("is_question_event"): next_state = ChatflowState.INTENT_EVENT_QUESTION
            elif tool_results.get("is_out_of_scope_question"): next_state = ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION
            else:
                next_state = ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION # Fallback
            break

        elif current_state == ChatflowState.INTENT_POTENTIAL_PATIENT:
            _, tool_results, tool_names, _ = await _call_model_with_tools(history_messages, client, [is_condition_treated])
            tool_call_name = tool_names[0] if tool_names else None
            
            if tool_results.get("is_condition_treated"): next_state = ChatflowState.CONDITION_IS_TREATED
            else: next_state = ChatflowState.CONDITION_NOT_TREATED_SEND_CONTACT_INFO
            break

        elif current_state == ChatflowState.INTENT_FAQ:
            tools_to_use = [is_question_insurance, is_service_pricey, is_question_in_person, is_question_location]
            _, tool_results, tool_names, _ = await _call_model_with_tools(history_messages, client, tools_to_use)
            tool_call_name = tool_names[0] if tool_names else None
            
            if tool_results.get("is_question_insurance"): next_state = ChatflowState.ANSWER_QUESTION_INSURANCE
            elif tool_results.get("is_question_in_person"): next_state = ChatflowState.ANSWER_QUESTION_IN_PERSON
            elif tool_results.get("is_question_location"): next_state = ChatflowState.ANSWER_QUESTION_LOCATION
            elif tool_results.get("is_service_pricey"): next_state = ChatflowState.ANSWER_QUESTION_PRICEY_SERVICE
            else: 
                next_state = ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION # Fallback
            break

        elif current_state == ChatflowState.ASKED_STATE:
            _, tool_results, tool_names, _ = await _call_model_with_tools(history_messages, client, [is_valid_state])
            tool_call_name = tool_names[0] if tool_names else None

            if tool_results.get("is_valid_state"): next_state = ChatflowState.OFFER_BOOK_CALL
            else: next_state = ChatflowState.CUSTOMER_IN_NON_VALID_STATE
            break

        elif current_state == ChatflowState.SENT_BOOK_CALL_OFFER:
            _, tool_results, tool_names, _ = await _call_model_with_tools(history_messages, client, [user_accepts_book_call])
            tool_call_name = tool_names[0] if tool_names else None

            if tool_results.get("user_accepts_book_call"): next_state = ChatflowState.BOOK_CALL_LINK_SENT
            else: next_state = ChatflowState.BOOK_CALL_OFFER_DECLINED
            break

        # --- Decision states ---
        elif current_state in [ChatflowState.INTENT_FRUSTRATED_CUSTOMER, ChatflowState.CONDITION_IS_TREATED, ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE, ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION, ChatflowState.RECOMMENDED_DOCTOR]:
            if interaction_data.get("book_call_offered"):
                next_state = ChatflowState.CONVERSATION_FINISHED
            else:
                next_state = ChatflowState.BOOK_CALL_NOT_OFFERED_YET
            
            # Message for frustrated customer
            if current_state == ChatflowState.INTENT_FRUSTRATED_CUSTOMER:
                 assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_FRUSTRATED_CUSTOMER_OFFER_BOOK_CALL))
            elif current_state == ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE:
                 assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=ACKNOWLEDGMENT_MESSAGE))

            if next_state == ChatflowState.CONVERSATION_FINISHED:
                break
            # else, fall through to BOOK_CALL_NOT_OFFERED_YET

        # --- Message + Transition states ---
        elif current_state in [ChatflowState.ANSWER_QUESTION_INSURANCE, ChatflowState.ANSWER_QUESTION_IN_PERSON, ChatflowState.ANSWER_QUESTION_PRICEY_SERVICE]:
            if current_state == ChatflowState.ANSWER_QUESTION_INSURANCE:
                assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_QUESTION_INSURANCE))
            elif current_state == ChatflowState.ANSWER_QUESTION_IN_PERSON:
                assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_QUESTION_IN_PERSON))
            elif current_state == ChatflowState.ANSWER_QUESTION_PRICEY_SERVICE:
                assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_QUESTION_PRICEY_SERVICE))
            next_state = ChatflowState.PROVIDED_FAQ_INFORMATION
            break
        
        elif current_state == ChatflowState.ANSWER_QUESTION_LOCATION:
            next_state = ChatflowState.ASKED_STATE
            # No message here, just transition to ask for state. Fall through to BOOK_CALL_NOT_OFFERED_YET logic.
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_ASK_STATE))
            break

        elif current_state == ChatflowState.BOOK_CALL_NOT_OFFERED_YET:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_ASK_STATE))
            next_state = ChatflowState.ASKED_STATE
            break
        
        elif current_state == ChatflowState.OFFER_BOOK_CALL:
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=PROMPT_OFFER_BOOK_CALL))
            interaction_data["book_call_offered"] = True
            next_state = ChatflowState.SENT_BOOK_CALL_OFFER
            break

        # --- Special function states ---
        elif current_state in [ChatflowState.AWAITING_USER_DATA, ChatflowState.INTENT_QUESTION_CONDITION, ChatflowState.PROVIDED_CONDITION_INFORMATION, ChatflowState.RECOMMENDED_DOCTOR, ChatflowState.PROVIDED_FAQ_INFORMATION, ChatflowState.INTENT_EVENT_QUESTION, ChatflowState.BOOK_CALL_LINK_SENT, ChatflowState.MAILING_LIST_OFFER_ACCEPTED]:
            # These states correspond to calling a special function and then transitioning.
            # In this implementation, we simulate this by setting the tool call name and moving to the next state.
            if current_state == ChatflowState.AWAITING_USER_DATA: tool_call_name, next_state = send_user_data_form(), ChatflowState.ANALYZING_INTENT
            elif current_state == ChatflowState.INTENT_QUESTION_CONDITION: tool_call_name, next_state = send_information_about_condition(), ChatflowState.PROVIDED_CONDITION_INFORMATION
            elif current_state == ChatflowState.PROVIDED_CONDITION_INFORMATION: tool_call_name, next_state = send_information_about_condition(), ChatflowState.RECOMMENDED_DOCTOR
            elif current_state == ChatflowState.RECOMMENDED_DOCTOR: tool_call_name, next_state = send_doctor_information(), ChatflowState.RECOMMENDED_DOCTOR # This then goes to decision
            elif current_state == ChatflowState.PROVIDED_FAQ_INFORMATION: tool_call_name, next_state = send_faq_information(), ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE
            elif current_state == ChatflowState.INTENT_EVENT_QUESTION: tool_call_name, next_state = send_event_information(), ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE
            elif current_state == ChatflowState.BOOK_CALL_LINK_SENT: tool_call_name, next_state = send_book_call_link(), ChatflowState.CONVERSATION_FINISHED
            elif current_state == ChatflowState.MAILING_LIST_OFFER_ACCEPTED: tool_call_name, next_state = save_to_mailing_list(), ChatflowState.CONVERSATION_FINISHED # End of flow
            
            # The RECOMMENDED_DOCTOR state needs to make a decision after tool call
            if current_state == ChatflowState.RECOMMENDED_DOCTOR:
                if interaction_data.get("book_call_offered"):
                    next_state = ChatflowState.CONVERSATION_FINISHED
                else:
                    next_state = ChatflowState.BOOK_CALL_NOT_OFFERED_YET
            break

        elif current_state == ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE:
            # This is a waiting state. The conversation turn ends here.
            # The next user message will start the flow again from ANALYZING_INTENT.
            next_state = ChatflowState.ANALYZING_INTENT
            break

        else:
            logger.warning(f"Unhandled state: {current_state}. Escalating to human.")
            assistant_messages.append(InteractionMessage(role=InteractionType.MODEL, message=shared_obtener_ayuda_humana()))
            next_state = GlobalState.HUMAN_ESCALATION
            tool_call_name = "obtener_ayuda_humana"
            break

    return assistant_messages, next_state, tool_call_name, interaction_data