# Boolean flag tools
def is_emergency():
    """Use this tool when the user mentions that this is an emergency or a death-live situation"""
    return True


def is_potential_patient():
    """Use this tool when the user expresses interest about the clinic and it's services"""
    return True


def is_question_about_condition():
    """Use this tool when the user asks a question about a condition they have"""
    return True


def is_question_event():
    """Use this tool when the user asks a question about an event"""
    return True


def is_frequently_asked_question():
    """Use this tool when the user asks a question about one of the following topics:
    - Questions about insurance
    - Questions about the price of the services
    - Questions about the location of the clinic and where does it provide services
    - Questions about in-person visits
    - Questions about insurance and insurance and insurance
    """
    return True


def is_out_of_scope_question():
    """Use this tool when the user asks a question un-related to anything from the clinic or health services"""
    return True


def is_frustrated_needs_human():
    """Use this tool when the user expresses frustration or intent co communicate with a human"""
    return True


def is_acknowledgment():
    """Use this tool when the user acknowledges a response and says something like
    - Thank you
    - That's all I wanted to know
    """
    return True


def is_question_in_person():
    """Use this tool when the user asks about in-person visits"""
    pass


def is_question_insurance():
    """Use this tool when the user asks about consultations with insurance"""
    pass


def is_question_location():
    """Use this tool when the user asks a question about where the services are provided"""
    pass


def is_service_pricey():
    """Use this tool when the answer expresses that the service is expensive or pricey, also if they ask for 'discounts' or reduced prices"""



def is_valid_state(is_valid: bool):
    """Use this tool to validate if the state where user resides in is a valid state to provide the service

    Aya Naturopathic Medicine offers services to NH, ME, MA, CT and CA
    """
    return is_valid



# Boolean toggle tools
def is_condition_treated(is_treated: bool) -> bool:
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
    return is_treated


def user_accepts_book_call(user_accepts: bool):
    """Use this tool to determine if the user accepts a book call, if the user accepts return True, otherwise return False"""
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
