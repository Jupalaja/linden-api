import logging
from typing import Optional

import google.genai as genai

from .state import ChatflowState
from .knowledge_data import *
from .prompts import *
from .tools import *
from src.config import settings
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.utils.functions import execute_tool_calls_and_get_response

logger = logging.getLogger(__name__)

async def _get_response(
    history_messages: list[InteractionMessage],
    client: genai.Client,
    tools: list,
    system_prompt: str,
    context: str | None = None,
) -> tuple[str, dict, list[str], dict]:
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    return await execute_tool_calls_and_get_response(
        history_messages=history_messages,
        client=client,
        tools=tools,
        system_prompt=full_system_prompt,
    )


async def _send_message(
    message: str, next_state: ChatflowState, interaction_data: dict
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_messages = [InteractionMessage(role=InteractionType.MODEL, message=message)]
    return response_messages, next_state, None, interaction_data


async def intent_classification_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [classify_intent]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    intent = tool_results.get("classify_intent")

    state_map = {
        "is_emergency": ChatflowState.INVALID_REQUEST_EMERGENCY,
        "is_potential_patient": ChatflowState.INTENT_POTENTIAL_PATIENT,
        "is_question_about_condition": ChatflowState.INTENT_QUESTION_CONDITION,
        "is_question_event": ChatflowState.INTENT_EVENT_QUESTION,
        "is_frequently_asked_question": ChatflowState.INTENT_FAQ,
        "is_out_of_scope_question": ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION,
        "is_frustrated_needs_human": ChatflowState.INTENT_FRUSTRATED_CUSTOMER,
        "is_acknowledgment": ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE,
    }
    next_state = state_map.get(
        intent, ChatflowState.CLASSIFYING_INTENT
    )  # Fallback to re-classify
    return [], next_state, None, interaction_data


async def question_condition_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [is_condition_treated]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    treated = tool_results.get("is_condition_treated", False)
    next_state = (
        ChatflowState.PROVIDE_CONDITION_INFORMATION
        if treated
        else ChatflowState.CONDITION_NOT_TREATED_SEND_CONTACT_INFO
    )
    return [], next_state, None, interaction_data


async def provide_condition_information_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text, _, _, _ = await _get_response(
        history_messages, client, [], CHATFLOW_SYSTEM_PROMPT, context=CONDITIONS_DATA
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.RECOMMENDED_DOCTOR,
        None,
        interaction_data,
    )


async def potential_patient_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return [], ChatflowState.BOOK_CALL_NOT_OFFERED_YET, None, interaction_data


async def frustrated_customer_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_FRUSTRATED_CUSTOMER_OFFER_BOOK_CALL,
        ChatflowState.BOOK_CALL_NOT_OFFERED_YET,
        interaction_data,
    )


async def out_of_scope_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text, _, _, _ = await _get_response(
        history_messages, client, [], CHATFLOW_SYSTEM_PROMPT
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def recommended_doctor_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [send_doctor_information]
    response_text, _, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.BOOK_CALL_NOT_OFFERED_YET,
        None,
        interaction_data,
    )


async def customer_acknowledges_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        ACKNOWLEDGMENT_MESSAGE, ChatflowState.BOOK_CALL_NOT_OFFERED_YET, interaction_data
    )


async def faq_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [classify_faq]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT, context=FAQ_DATA
    )

    intent = tool_results.get("classify_faq")

    state_map = {
        "is_question_in_person": ChatflowState.ANSWER_QUESTION_IN_PERSON,
        "is_question_insurance": ChatflowState.ANSWER_QUESTION_INSURANCE,
        "is_service_pricey": ChatflowState.ANSWER_QUESTION_PRICEY_SERVICE,
        "is_general_faq_question": ChatflowState.PROVIDED_FAQ_INFORMATION,
    }
    next_state = state_map.get(
        intent, ChatflowState.PROVIDED_FAQ_INFORMATION
    )  # Fallback to faq-information
    return [], next_state, None, interaction_data


async def ask_state_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(PROMPT_ASK_STATE, ChatflowState.ASKED_STATE, interaction_data)


async def condition_not_treated_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_CONDITION_NOT_TREATED,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def event_question_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text, _, _, _ = await _get_response(
        history_messages,
        client,
        [],
        CHATFLOW_SYSTEM_PROMPT,
        context=EVENTS_DATA,
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def provided_faq_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text, _, _, _ = await _get_response(
        history_messages, client, [], CHATFLOW_SYSTEM_PROMPT, context=FAQ_DATA
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def emergency_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        OUTPUT_MESSAGE_EMERGENCY,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_insurance_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_QUESTION_INSURANCE,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_pricey_service_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_QUESTION_PRICEY_SERVICE,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_in_person_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_QUESTION_IN_PERSON,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def validate_state_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [is_valid_state]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    valid = tool_results.get("is_valid_state", False)
    if valid:
        next_state = ChatflowState.OFFER_BOOK_CALL
        return [], next_state, None, interaction_data
    else:
        return await _send_message(
            PROMPT_INVALID_STATE,
            ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
            interaction_data,
        )


async def offer_book_call_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_OFFER_BOOK_CALL, ChatflowState.SENT_BOOK_CALL_OFFER, interaction_data
    )


async def request_resolved_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    # This state loops back to intent classification for a new user query
    return [], ChatflowState.CLASSIFYING_INTENT, None, interaction_data


async def sent_book_call_offer_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [user_accepts_book_call]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    accepts = tool_results.get("user_accepts_book_call", False)
    next_state = (
        ChatflowState.BOOK_CALL_LINK_SENT
        if accepts
        else ChatflowState.BOOK_CALL_OFFER_DECLINED
    )
    return [], next_state, None, interaction_data


async def book_call_declined_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_PROVIDE_CONTACT_INFO, ChatflowState.CONVERSATION_FINISHED, interaction_data
    )


async def book_call_link_sent_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [send_book_call_link]
    response_text, _, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.CONVERSATION_FINISHED,
        None,
        interaction_data,
    )


async def conversation_finished_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        PROMPT_OFFER_NEWSLETTER,
        ChatflowState.AWAITING_NEWSLETTER_RESPONSE,
        interaction_data,
    )


async def await_newsletter_response_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [user_accepts_newsletter]
    _, tool_results, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    accepts = tool_results.get("user_accepts_newsletter", False)
    if accepts:
        return [], ChatflowState.MAILING_LIST_OFFER_ACCEPTED, None, interaction_data

    # If user declines, we can end the conversation.
    return [], ChatflowState.IDLE, None, interaction_data


async def mailing_list_accepted_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    client: genai.Client,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    tools_to_use = [save_to_mailing_list]
    response_text, _, _, _ = await _get_response(
        history_messages, client, tools_to_use, CHATFLOW_SYSTEM_PROMPT
    )
    # After saving, conversation can be considered idle/ended
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response_text)],
        ChatflowState.IDLE,
        None,
        interaction_data,
    )