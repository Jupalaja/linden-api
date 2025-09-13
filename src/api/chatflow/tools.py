from typing import Literal, Optional
from langchain_core.tools import tool


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


@tool
def classify_intent(intent: ConversationType) -> ConversationType:
    """Classifies the user's intent based on their message. Call this function with the most relevant classification.

    Classifications:
    - is_emergency: User mentions urgent medical situations, severe symptoms, or emergency care needs
    - is_potential_patient: User explicitly asks to become a patient or book an appointment. Use this for direct requests like "I want to book an appointment," "How do I become a patient?," or "I'd like to work with one of your doctors." Do NOT use for general service inquiries.
    - is_question_about_condition: User asks if specific health conditions are treated or mentions symptoms
    - is_question_event: User asks about scheduling, availability, hours, or appointment logistics
    - is_frequently_asked_question: User asks a general question about services, pricing, insurance, location, or other common topics. Use this for inquiries like "What do you offer?," "Tell me about your services," or "Do you take insurance?".
    - is_out_of_scope_question: User asks about services not offered (pregnancy care, pediatrics under 6, severe psychiatric conditions, etc.)
    - is_frustrated_needs_human: User expresses frustration, wants to speak to a person, or is dissatisfied with bot responses
    - is_acknowledgment: User says thanks, goodbye, or acknowledges information provided

    Args:
        intent: The user's intent classification.
    """
    return intent


@tool
def is_valid_state(is_valid: bool) -> bool:
    """Use this tool to validate if the state where user resides is eligible for services.

    Aya Naturopathic Medicine provides services to residents of:
    - New Hampshire (NH) - Both doctors available via telehealth and in-person
    - Maine (ME) - Both doctors available via telehealth only
    - Massachusetts (MA) - Both doctors available via telehealth only
    - Connecticut (CT) - Both doctors available via telehealth only
    - California (CA) - Dr. Silva ONLY via telehealth

    Set is_valid to True if user's state is in the above list, False otherwise.
    For California residents, they can only see Dr. Silva.

    Args:
        is_valid: True if state is served, False if not served
    """
    return is_valid


@tool
def classify_faq(questionType: QuestionType) -> QuestionType:
    """Classifies the user's frequently asked question type. Call this function with the most relevant classification.

    Question Types:
    - is_question_in_person: User asks about in-person visits, location, or physical office availability
    - is_question_insurance: User asks about insurance coverage, superbills, HSA/FSA, or payment methods
    - is_service_pricey: User expresses concern about cost, asks about affordability, payment plans, or pricing
    - is_general_faq_question: User asks other common questions about services, doctors, telehealth, or general practice information

    Args:
        questionType: The user's FAQ question type classification.
    """
    return questionType


@tool
def is_condition_treated(is_treated: bool) -> bool:
    """Use this tool to identify if the condition provided by the user is treated. Set `is_treated` to True if the condition is treated, otherwise set to False.

    CONDITIONS WE TREAT (set is_treated=True):

    Women's Health: hormone imbalances, perimenopause/menopause, PCOS, endometriosis, fibroids, PMS, PMDD, urinary incontinence, infertility, iron deficiency anemia, thyroid conditions, sexual health concerns, bone density issues, mood changes, skin issues from hormones

    Mental Health & Brain Function: anxiety, depression, OCD, PTSD, ADHD, bipolar disorder, insomnia, autism (age 6+), cognitive decline, MCI, Parkinson's/Alzheimer's support, brain fog, memory lapses, emotional regulation, executive dysfunction

    Digestive Health: IBS, SIBO, GERD, constipation, diarrhea, Celiac, Crohn's, ulcerative colitis, gastritis, gallbladder issues, pancreatitis, bloating, abdominal pain, dyspepsia, sluggish digestion

    Metabolic & Endocrine Health: hypothyroidism, Hashimoto's, Graves', prediabetes, type 2 diabetes, PCOS, metabolic syndrome, high cholesterol, gestational diabetes, weight regulation, type 1 diabetes (with endocrinologist co-management)

    Immune & Inflammatory: lupus, RA, Sjögren's, MS, scleroderma, psoriasis, eczema, asthma, chronic fatigue syndrome, fibromyalgia, post-viral syndromes, histamine intolerance, long COVID, chronic inflammation, acute colds/flus

    Prevention & Optimization: energy optimization, cognitive enhancement, mood support, immune resilience, detox pathways, sleep issues, oxidative stress, early cognitive decline, bone health, family history risk management, healthy aging, longevity, high performance

    CONDITIONS WE DON'T TREAT (set is_treated=False):
    - Emergency care situations
    - Primary care services (24/7 coverage, routine screenings, vaccinations)
    - Pregnancy and birth care (prenatal, labor, postpartum)
    - Cancer as primary diagnosis
    - Pediatrics under age 6
    - Personality disorders, eating disorders, unmanaged substance use
    - Severe psychiatric conditions (schizophrenia, psychosis)
    - Primary immunodeficiency disorders (SCID, CVID)

    Args:
        is_treated: True if condition is treated by Aya, False if not treated
    """
    return is_treated


@tool
def user_accepts_book_call(user_accepts: bool) -> bool:
    """Use this tool to determine if the user accepts to book a free 15-minute discovery call.

    Look for responses like:
    - "Yes, I'd like to book a call"
    - "That sounds good"
    - "Let's do it"
    - "I'm interested"
    - "How do I book?"

    Set user_accepts=True if they agree or show interest in booking.
    Set user_accepts=False if they decline, say "not right now", "maybe later", or express hesitation.

    Args:
        user_accepts: True if user wants to book discovery call, False if declining
    """
    return user_accepts


@tool
def user_accepts_newsletter(user_accepts: bool) -> bool:
    """Use this tool to determine if the user accepts to join the newsletter mailing list.

    The newsletter includes:
    - Monthly health tips and seasonal wellness advice
    - Nourishing recipes
    - Hormone and mood support information
    - Updates on retreats and new services (like upcoming biofeedback)
    - Users can unsubscribe anytime

    Look for responses like:
    - "Yes, sign me up"
    - "That sounds helpful"
    - "I'd like that"
    - "Sure"

    Set user_accepts=True if they agree to join.
    Set user_accepts=False if they decline or say "no thanks".

    Args:
        user_accepts: True if user wants to join newsletter, False if declining
    """
    return user_accepts


@tool
def save_to_mailing_list() -> str:
    """Use this tool to save the user to the newsletter mailing list after they have accepted.

    Only call this tool AFTER user_accepts_newsletter() returns True.
    This will add them to receive monthly health tips, recipes, hormone/mood support info,
    retreat updates, and news about new services like biofeedback.
    """
    return "Perfect! I already have your email, so you have been added to our mailing list"


@tool
def send_book_call_link() -> str:
    """Use this tool to provide the user with the booking link for a free 15-minute discovery call.

    Only call this tool AFTER user_accepts_book_call() returns True.

    The discovery call includes:
    - 15 minutes with a doctor to discuss health goals
    - Explanation of our approach to their specific concerns
    - Determination of next steps and fit assessment
    - Discovery call intake forms sent automatically after booking
    - Completely free with no obligation

    The booking link is: https://bookinglink.com/
    """
    return "Here's the **[Link to Book a Discovery Call](https://ayanaturopathicmedicine.practicebetter.io/#/66fb4a41904772d2c40fe3fc/bookings?r=6706dd2ad30b811dc03d2644&step=services)**"


@tool
def get_user_data(name: Optional[str] = None, email: Optional[str] = None) -> dict:
    """Extracts user's name and email from their message.
    If the user provides their name and/or email, use this tool to capture it.
    If they decline or don't provide the information, you can call this tool with no arguments.
    """
    user_data = {}
    if name:
        user_data["name"] = name
    if email:
        user_data["email"] = email
    return user_data


@tool
def send_doctor_information(best_doctor_for_client: str) -> str:
    """Use this tool to recommend the most appropriate doctor based on user's health concerns and location.

    DOCTOR MATCHING LOGIC:
    - Women's health or endocrine issues → Dr. Silva
    - Mental health, neurology, or Alzheimer's prevention → Dr. Jeffrey
    - General or complex cases → Alternate between Dr. Silva and Dr. Jeffrey
    - California residents → MUST be Dr. Silva (only doctor licensed in CA)

    AVAILABILITY SCHEDULE:

    In-Person (Whole Life Healthcare, 100 Shattuck Way, Newington, NH):
    - Current through July: Fridays 8AM-5PM (Dr. Jeffrey only)
    - Starting August 1st: Tuesday-Friday 8AM-5PM (both doctors, alternating days/shifts)

    Telehealth:
    - Tuesday-Friday 9AM-5PM
    - Thursday evenings until 7PM (Dr. Jeffrey only)

    SERVICE AREAS:
    - Dr. Jeffrey: NH, ME, MA, CT (telehealth + in-person in NH)
    - Dr. Silva: NH, ME, MA, CT, CA (telehealth + in-person in NH starting August)

    Args:
        best_doctor_for_client: Name of recommended doctor with brief reasoning
    """
    return best_doctor_for_client
