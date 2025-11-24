import logging

from langchain_core.language_models import BaseChatModel

from .state import ChatflowState
from .knowledge_data import *
from .prompts import *
from .tools import *
from src.services.embeddings import retrieve_data
from src.services.google_sheets import GoogleSheetsService
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage
from src.shared.utils.functions import (
    call_single_tool,
    generate_response_text,
    write_candidato_a_empleo_to_sheet,
)
from src.shared.utils.history import get_langchain_history

logger = logging.getLogger(__name__)


async def _send_message(
    history_messages: list[InteractionMessage],
    model: BaseChatModel,
    message: str,
    next_state: ChatflowState,
    interaction_data: dict,
    add_acknowledgment: bool = True,
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    if add_acknowledgment:
        context = f"{INSTRUCTION_FOR_ACKNOWLEDGEMENT}\n\nUPCOMING SYSTEM MESSAGE: {message}"
        acknowledgment = await generate_response_text(
            history_messages,
            model,
            system_prompt=CHATFLOW_SYSTEM_PROMPT,
            context=context,
        )
        if acknowledgment and acknowledgment.strip():
            full_message = f"{acknowledgment}\n\n{message}"
        else:
            # Fallback if acknowledgment generation fails or is empty
            full_message = message
    else:
        full_message = message

    response_messages = [InteractionMessage(role=InteractionType.MODEL, message=full_message)]
    return response_messages, next_state, None, interaction_data


async def idle_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return (
        [],
        ChatflowState.CLASSIFYING_INTENT,
        None,
        interaction_data,
    )


async def ask_user_data_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    condition_info = interaction_data.pop("condition_info_response", "")
    doctor_recommendation = interaction_data.pop("doctor_recommendation_response", "")

    # Generate a cohesive response that combines condition info, doctor recommendation, and user data request
    context_parts = []
    context_parts.append("The user has asked about a condition. Create a natural, cohesive response that:")
    
    if condition_info:
        context_parts.append(f"- Incorporates this condition information: {condition_info}")
    
    if doctor_recommendation:
        context_parts.append(f"- Incorporates this doctor recommendation: {doctor_recommendation}")
    
    context_parts.append(f"- Asks for their information naturally: {PROMPT_ASK_USER_DATA}")
    context_parts.append("\nCreate a single, flowing response that feels natural and cohesive, not like separate pieces stitched together.")

    context = "\n".join(context_parts)

    full_message = await generate_response_text(
        history_messages,
        model,
        system_prompt=CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )

    if not full_message:
        # Fallback if generation fails
        message_parts = []
        if condition_info:
            message_parts.append(condition_info)
        if doctor_recommendation:
            message_parts.append(doctor_recommendation)
        message_parts.append(PROMPT_ASK_USER_DATA)
        full_message = "\n\n".join(message_parts)

    return await _send_message(
        history_messages,
        model,
        full_message,
        ChatflowState.GET_USER_DATA,
        interaction_data,
        add_acknowledgment=False,
    )


async def get_user_data_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, get_user_data, CHATFLOW_SYSTEM_PROMPT
    )

    user_data = tool_results.get("get_user_data")
    if user_data:
        interaction_data = interaction_data.copy()
        if "user_data" in interaction_data and isinstance(
            interaction_data["user_data"], dict
        ):
            interaction_data["user_data"] = interaction_data["user_data"].copy()
            interaction_data["user_data"].update(user_data)
        else:
            interaction_data["user_data"] = user_data

    return [], ChatflowState.ASKED_STATE, None, interaction_data


async def intent_classification_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    practice_id = interaction_data.get("practice_id")
    if practice_id and history_messages:
        query = history_messages[-1].message
        response, found = retrieve_data(query=query, practice_id=practice_id)
        if found:
            interaction_data["embeddings_response"] = response
            return [], ChatflowState.REPLY_FROM_EMBEDDINGS, None, interaction_data

    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, classify_intent, CHATFLOW_SYSTEM_PROMPT
    )
    intent = tool_results.get("classify_intent")

    state_map = {
        "is_emergency": ChatflowState.INVALID_REQUEST_EMERGENCY,
        "is_potential_patient": ChatflowState.ASK_USER_DATA,
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
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, is_condition_treated, CHATFLOW_SYSTEM_PROMPT
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
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    context = f"{INSTRUCTION_ANSWER_ABOUT_CONDITION}\n\n{CONDITIONS_DATA}"
    response_text = await generate_response_text(
        history_messages, model, CHATFLOW_SYSTEM_PROMPT, context=context
    )
    interaction_data["condition_info_response"] = response_text
    return (
        [],
        ChatflowState.RECOMMENDED_DOCTOR,
        None,
        interaction_data,
    )


async def frustrated_customer_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_FRUSTRATED_CUSTOMER_OFFER_BOOK_CALL,
        ChatflowState.OFFER_BOOK_CALL,
        interaction_data,
    )


async def out_of_scope_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_OUT_OF_SCOPE_QUESTION,
        ChatflowState.OFFER_BOOK_CALL,
        interaction_data,
    )


async def reply_from_embeddings_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response = interaction_data.pop("embeddings_response", "I am not sure how to answer that, can you rephrase?")
    return (
        [InteractionMessage(role=InteractionType.MODEL, message=response)],
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        None,
        interaction_data,
    )


async def recommended_doctor_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, send_doctor_information, CHATFLOW_SYSTEM_PROMPT
    )
    doctor_recommendation = tool_results.get(
        "send_doctor_information", "Our doctors would be happy to help with your condition."
    )

    context = f"{INSTRUCTION_RECOMMEND_DOCTOR}\n\nDoctor recommendation: {doctor_recommendation}\n\n{CONDITIONS_DATA}"
    response_text = await generate_response_text(
        history_messages,
        model,
        CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )
    interaction_data["doctor_recommendation_response"] = response_text

    return (
        [],
        ChatflowState.ASK_USER_DATA,
        None,
        interaction_data,
    )


async def customer_acknowledges_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        ACKNOWLEDGMENT_MESSAGE,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
        add_acknowledgment=False,
    )


async def faq_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, classify_faq, CHATFLOW_SYSTEM_PROMPT, context=FAQ_DATA
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



async def condition_not_treated_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_CONDITION_NOT_TREATED,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def event_question_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    response_text = await generate_response_text(
        history_messages,
        model,
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
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    context = f"{INSTRUCTION_GENERAL_FAQ_QUESTION}\n\n{FAQ_DATA}"
    response_text = await generate_response_text(
        history_messages, model, CHATFLOW_SYSTEM_PROMPT, context=context
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
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        OUTPUT_MESSAGE_EMERGENCY,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_insurance_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_QUESTION_INSURANCE,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_pricey_service_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_QUESTION_PRICEY_SERVICE,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def answer_in_person_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_QUESTION_IN_PERSON,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
        interaction_data,
    )


async def validate_state_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, is_valid_state, CHATFLOW_SYSTEM_PROMPT
    )
    valid = tool_results.get("is_valid_state", False)
    if valid:
        next_state = ChatflowState.BOOK_CALL_OFFER_ACCEPTED
        return [], next_state, None, interaction_data
    else:
        return await _send_message(
            history_messages,
            model,
            PROMPT_INVALID_STATE,
            ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE,
            interaction_data,
        )


async def offer_book_call_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    return await _send_message(
        history_messages,
        model,
        PROMPT_OFFER_BOOK_CALL,
        ChatflowState.AWAITING_BOOK_CALL_OFFER_RESPONSE,
        interaction_data,
    )


async def request_resolved_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    # This state loops back to intent classification for a new user query
    return [], ChatflowState.CLASSIFYING_INTENT, None, interaction_data


async def await_book_call_response_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, user_accepts_book_call, CHATFLOW_SYSTEM_PROMPT
    )
    accepts = tool_results.get("user_accepts_book_call", False)
    next_state = (
        ChatflowState.BOOK_CALL_OFFER_ACCEPTED
        if accepts
        else ChatflowState.BOOK_CALL_OFFER_DECLINED
    )
    return [], next_state, None, interaction_data


async def book_call_declined_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    full_message = f"{PROMPT_PROVIDE_CONTACT_INFO}\n\n{PROMPT_OFFER_NEWSLETTER}"
    return await _send_message(
        history_messages,
        model,
        full_message,
        ChatflowState.AWAITING_NEWSLETTER_RESPONSE,
        interaction_data,
    )


async def book_call_link_accepted_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, send_book_call_link, CHATFLOW_SYSTEM_PROMPT
    )
    # The send_book_call_link tool returns the message to send
    response_text = tool_results.get("send_book_call_link")

    # Generate a friendly response including the link and the newsletter offer
    context = (
        f"The user has agreed to book a call. Create a response that includes:\n"
        f"- A brief acknowledgment.\n"
        f"- This booking info: {response_text}\n"
        f"- This newsletter offer: {PROMPT_OFFER_NEWSLETTER}"
    )

    full_message = await generate_response_text(
        history_messages,
        model,
        system_prompt=CHATFLOW_SYSTEM_PROMPT,
        context=context,
    )

    if not full_message:
        full_message = f"{response_text}\n\n{PROMPT_OFFER_NEWSLETTER}"

    return await _send_message(
        history_messages,
        model,
        full_message,
        ChatflowState.AWAITING_NEWSLETTER_RESPONSE,
        interaction_data,
        add_acknowledgment=False,
    )


async def await_newsletter_response_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, user_accepts_newsletter, CHATFLOW_SYSTEM_PROMPT
    )
    accepts = tool_results.get("user_accepts_newsletter", False)
    next_state = (
        ChatflowState.MAILING_LIST_OFFER_ACCEPTED
        if accepts
        else ChatflowState.MAILING_LIST_OFFER_DECLINED
    )
    return [], next_state, None, interaction_data


async def mailing_list_accepted_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    langchain_messages = get_langchain_history(history_messages)
    tool_results = await call_single_tool(
        langchain_messages, model, save_to_mailing_list, CHATFLOW_SYSTEM_PROMPT
    )
    # The save_to_mailing_list tool returns a confirmation message
    response_text = tool_results.get(
        "save_to_mailing_list", "Thank you for subscribing to our mailing list!"
    )
    full_message = f"{response_text}\n\n{PROMPT_FAREWELL_MESSAGE}"

    response_message = InteractionMessage(
        role=InteractionType.MODEL, message=full_message
    )
    full_conversation = history_messages + [response_message]

    await write_candidato_a_empleo_to_sheet(
        interaction_data=interaction_data,
        conversation=full_conversation,
        sheets_service=sheets_service,
    )

    # After saving, conversation can be considered idle/ended
    return (
        [response_message],
        ChatflowState.FINAL,
        None,
        interaction_data,
    )


async def final_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    # This state is terminal, it does not produce any message and keeps the same state.
    return [], ChatflowState.FINAL, None, interaction_data


async def mailing_list_declined_workflow(
    history_messages: list[InteractionMessage],
    interaction_data: dict,
    model: BaseChatModel,
    sheets_service: Optional[GoogleSheetsService],
) -> tuple[list[InteractionMessage], ChatflowState, str | None, dict]:
    acknowledgment = await generate_response_text(
        history_messages,
        model,
        system_prompt=CHATFLOW_SYSTEM_PROMPT,
        context=INSTRUCTION_FOR_ACKNOWLEDGEMENT,
    )
    if acknowledgment and acknowledgment.strip():
        full_message = f"{acknowledgment}\n\n{PROMPT_FAREWELL_MESSAGE}"
    else:
        # Fallback if acknowledgment generation fails or is empty
        full_message = PROMPT_FAREWELL_MESSAGE

    response_message = InteractionMessage(
        role=InteractionType.MODEL, message=full_message
    )
    full_conversation = history_messages + [response_message]

    await write_candidato_a_empleo_to_sheet(
        interaction_data=interaction_data,
        conversation=full_conversation,
        sheets_service=sheets_service,
    )

    return (
        [response_message],
        ChatflowState.FINAL,
        None,
        interaction_data,
    )
