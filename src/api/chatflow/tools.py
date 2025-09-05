from typing import Literal, List
from .prompts import *
ConversationType = Literal[
    "is_emergency",
    "is_potential_patient",
    "is_question_about_condition",
    "is_question_event",
    "is_frequently_asked_question",
    "is_out_of_scope_question",
    "is_frustrated_needs_human",
    "is_acknowledgment",
]

def classify_intent(
    conversation_history: List[str]
) -> ConversationType:
    """
    Analyzes the user's conversation and returns the single most relevant classification for the most recent message.

    Args:
        conversation_history: An ordered list of messages in the conversation, starting with the oldest.

    Returns:
        The single most appropriate category chosen from the ConversationType options.
    """
    pass


def is_valid_state(is_valid: bool):
    """Use this tool to validate if the state where user resides in is a valid state to provide the service

    Aya Naturopathic Medicine offers services to NH, ME, MA, CT and CA
    """
    return is_valid


def is_question_in_person():
    """Use this tool when the user asks about in-person visits"""
    return PROMPT_QUESTION_IN_PERSON


def is_question_insurance():
    """Use this tool when the user asks about consultations with insurance"""
    return PROMPT_QUESTION_INSURANCE


def is_service_pricey():
    """Use this tool when the answer expresses that the service is expensive or pricey, also if they ask for 'discounts' or reduced prices"""
    return PROMPT_QUESTION_PRICEY_SERVICE


def is_general_faq_question(is_question_in_person: bool, is_question_insurance: bool, is_service_pricey: bool):
    """This tool should return True if all the input values are false"""
    return not (is_question_in_person or is_question_insurance or is_service_pricey)


# Boolean toggle tools
def is_condition_treated() -> bool:
    """Use this tool to identify if the condition provided by the user is treated or not, if the condition is treated return True, otherwise return False.

    At Aya Naturopathic Medicine, we support a wide range of health concerns using a root-cause, systems-based approach. Our doctors work with patients of all ages (6+) and specialize in the following areas:
    - Women’s Health
    - Mental Health & Brain Function
    - Digestive Health
    - Metabolic & Endocrine Health
    - Immune & Inflammatory Concerns
    - Prevention, Aging & Optimization

    We do not provide:
    - Emergency care
    - Primary care services
    - Vaccinations
    - Pregnancy and birth care
    - Cancer as a primary diagnosis
    - Pediatrics under age 6
    - Personality disorders, eating disorders, or unmanaged substance use
    - Severe psychiatric conditions
    - Primary immunodeficiency disorders
    """
    pass


def user_accepts_book_call(user_accepts: bool):
    """Use this tool to determine if the user accepts a book call, if the user accepts return True, otherwise return False"""
    return user_accepts


def user_accepts_newsletter(user_accepts: bool):
    """Use this tool to determine if the user accepts to be included in the newsletter mailing list, if the user accepts return True, otherwise return False"""
    return user_accepts


# Special functions (mocked for now)
def save_to_mailing_list() -> str:
    return "save_to_mailing_list"


def send_book_call_link() -> str:
    return "send_book_call_link"


def send_doctor_information() -> str:
    return "send_doctor_information"


def send_event_information() -> str:
    return "send_event_information"


def send_faq_information() -> str:
    return "send_faq_information"


def send_information_about_condition() -> str:
    return "send_information_about_condition"


def send_user_data_form() -> str:
    return "send_user_data_form"
