from typing import Literal

from .knowledge_data import FAQ_DATA, EVENTS_DATA, CONDITIONS_DATA
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


QuestionType = Literal[
    "is_question_in_person",
    "is_question_insurance",
    "is_service_pricey",
    "is_general_faq_question",
]


def classify_intent(intent: ConversationType) -> ConversationType:
    """Classifies the user's intent. Call this function with the most relevant classification.

    Args:
        intent: The user's intent.
    """
    return intent


def is_valid_state(is_valid: bool) -> bool:
    """Use this tool to validate if the state where user resides in is a valid state to provide the service

    Aya Naturopathic Medicine offers services to NH, ME, MA, CT and CA
    """
    return is_valid


def classify_faq(questionType: QuestionType) -> QuestionType:
    """Classifies the user's question. Call this function with the most relevant classification.

    Args:
        questionType: The user's type of question.
    """
    return questionType


# Boolean toggle tools
def is_condition_treated(is_treated: bool) -> bool:
    """Use this tool to identify if the condition provided by the user is treated or not. Set `is_treated` to True if the condition is treated, otherwise set to False.

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
    return is_treated


def user_accepts_book_call(user_accepts: bool) -> bool:
    """Use this tool to determine if the user accepts a book call, if the user accepts return True, otherwise return False"""
    return user_accepts


def user_accepts_newsletter(user_accepts: bool) -> bool:
    """Use this tool to determine if the user accepts to be included in the newsletter mailing list, if the user accepts return True, otherwise return False"""
    return user_accepts


# Special functions (mocked for now)
def save_to_mailing_list() -> str:
    return "save_to_mailing_list"


def send_book_call_link() -> str:
    """Use this tool to provide the user with the booking link for a free 15-minute discovery call.
    The link is: https://bookinglink.com/
    """
    return "Here's the bool call link: https://bookinglink.com/"


def send_user_data_form() -> str:
    """Use this tool to send a form to the user to collect their data."""
    return "send_user_data_form"


def send_doctor_information(best_doctor_for_client:str) -> str:
    """Use this tool to let the customer know what doctor is better for their necessity, their availability, and location.
    
    Doctors:
    - Dr. Jeffrey
    - Dr. Silva

    Availability:
    - In-Person (Whole Life Healthcare – 100 Shattuck Way, Newington, NH): Starting August 1st, Tuesday–Friday, 8:00 AM – 5:00 PM. Dr. Jeffrey and Dr. Silva will each be available for in-person visits two days per week, or in alternating shifts (AM/PM).
    - Telehealth: Tuesday–Friday, 9:00 AM – 5:00 PM (with evening appointments available Thursdays until 7:00 PM with Dr. Jeffrey).

    Service Area:
    - Both doctors see patients residing in NH, ME, MA, or CT.
    - Dr. Silva also sees California residents via telehealth.
    """
    return best_doctor_for_client



