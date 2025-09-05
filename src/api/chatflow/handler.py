from .workflows import *
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage

logger = logging.getLogger(__name__)


async def handle_chatflow(
    session_id: str,
    history_messages: list[InteractionMessage],
    current_state: ChatflowState,
    interaction_data: Optional[dict],
    client: genai.Client,
) -> tuple[list[InteractionMessage], list[ChatflowState], str | None, dict]:
    interaction_data = dict(interaction_data) if interaction_data else {}

    workflow_map = {
        ChatflowState.CLASSIFYING_INTENT: intent_classification_workflow,
        ChatflowState.INTENT_QUESTION_CONDITION: question_condition_workflow,
        ChatflowState.PROVIDE_CONDITION_INFORMATION: provide_condition_information_workflow,
        ChatflowState.INTENT_POTENTIAL_PATIENT: potential_patient_workflow,
        ChatflowState.INTENT_FRUSTRATED_CUSTOMER: frustrated_customer_workflow,
        ChatflowState.INTENT_OUT_OF_SCOPE_QUESTION: out_of_scope_workflow,
        ChatflowState.RECOMMENDED_DOCTOR: recommended_doctor_workflow,
        ChatflowState.CUSTOMER_ACKNOWLEDGES_RESPONSE: customer_acknowledges_workflow,
        ChatflowState.INTENT_FAQ: faq_workflow,
        ChatflowState.BOOK_CALL_NOT_OFFERED_YET: ask_state_workflow,
        ChatflowState.CONDITION_NOT_TREATED_SEND_CONTACT_INFO: condition_not_treated_workflow,
        ChatflowState.INTENT_EVENT_QUESTION: event_question_workflow,
        ChatflowState.PROVIDED_FAQ_INFORMATION: provided_faq_workflow,
        ChatflowState.INVALID_REQUEST_EMERGENCY: emergency_workflow,
        ChatflowState.ANSWER_QUESTION_INSURANCE: answer_insurance_workflow,
        ChatflowState.ANSWER_QUESTION_PRICEY_SERVICE: answer_pricey_service_workflow,
        ChatflowState.ANSWER_QUESTION_IN_PERSON: answer_in_person_workflow,
        ChatflowState.ASKED_STATE: validate_state_workflow,
        ChatflowState.OFFER_BOOK_CALL: offer_book_call_workflow,
        ChatflowState.REQUEST_RESOLVED_AWAIT_NEW_MESSAGE: request_resolved_workflow,
        ChatflowState.SENT_BOOK_CALL_OFFER: sent_book_call_offer_workflow,
        ChatflowState.BOOK_CALL_OFFER_DECLINED: book_call_declined_workflow,
        ChatflowState.BOOK_CALL_LINK_SENT: book_call_link_sent_workflow,
        ChatflowState.CONVERSATION_FINISHED: conversation_finished_workflow,
        ChatflowState.AWAITING_NEWSLETTER_RESPONSE: await_newsletter_response_workflow,
        ChatflowState.MAILING_LIST_OFFER_ACCEPTED: mailing_list_accepted_workflow,
    }

    # Prepending introduction message for new conversations
    all_new_messages = []
    if len(history_messages) == 1 and history_messages[0].role == InteractionType.USER:
        all_new_messages.append(
            InteractionMessage(role=InteractionType.MODEL, message=LINDEN_INTRODUCTION_MESSAGE)
        )

    next_state = current_state
    final_tool_call = None
    new_states = []

    # Loop to handle state transitions within a single turn
    for _ in range(5):  # Safety break to prevent infinite loops
        workflow_func = workflow_map.get(next_state)
        if not workflow_func:
            logger.warning(
                f"No workflow for state: {next_state}. Defaulting to intent classification."
            )
            workflow_func = intent_classification_workflow

        # The history for the tool call should include messages generated so far in this turn
        current_turn_history = history_messages + all_new_messages

        new_messages, new_state, tool_call, interaction_data = await workflow_func(
            current_turn_history, interaction_data, client
        )

        if new_messages:
            all_new_messages.extend(new_messages)
        if tool_call:
            final_tool_call = tool_call

        if new_state == next_state:
            # State is stable, break loop
            break

        new_states.append(new_state)
        next_state = new_state

        if new_messages or tool_call:
            # If workflow produced output for the user, stop for this turn
            break

    return all_new_messages, new_states, final_tool_call, interaction_data
