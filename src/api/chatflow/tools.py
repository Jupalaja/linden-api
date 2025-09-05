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


def send_event_information() -> str:
    """Use this tool to provide information about upcoming events.
    
    ## 🌻 Aya at the Sunflower Festival – Tendercrop Farm, Dover  
    - **Date:** September 6, 2025  
    - **Time:** 10:00 am – 6:00 pm  
    - **Venue:** Tendercrop Farms  
    - **Organizer:** Tendercrop Farms  
    - **Price:** Free  

    ---

    ## 🌿 Craniosacral Therapy Sessions at Herbal Path  
    - **Date:** September 18, 2025  
    - **Time:** 1:30 pm – 4:30 pm  
    - **Venue:** The Herbal Path, A Natural Pharmacy  

    ---

    ## ✨ From Foggy to Focused: An Experiential Reset for Women Who Do It All  
    - **Date:** September 27, 2025  
    - **Time:** 10:00 am – 11:30 am  
    - **Venue:** Whole Life Health Care  
    - **Price:** $150  
    """
    return EVENTS_DATA


def send_faq_information() -> str:
    """Use this tool to answer frequently asked questions about Aya Naturopathic Medicine.
    
    ## Conditions Treated by Aya Naturopathic Medicine
    At Aya Naturopathic Medicine, we support a wide range of health concerns using a root-cause, systems-based approach. Our doctors work with patients of all ages (6+) and specialize in the following areas:

    **Women’s Health**
    We treat hormone imbalances, perimenopause/menopause, PCOS, endometriosis, fibroids, PMS and PMDD, urinary incontinence, infertility, iron deficiency anemia, thyroid conditions, and sexual health concerns. We also support bone density, mood changes, and skin issues related to hormone shifts.

    **Mental Health & Brain Function**
    We support patients with anxiety, depression, OCD, PTSD, ADHD, bipolar disorder, insomnia, autism (age 6+), cognitive decline, MCI, and Parkinson’s or Alzheimer’s support. Functional concerns like brain fog, memory lapses, emotional regulation issues, and executive dysfunction are a core part of our work.

    **Digestive Health**
    We treat IBS, SIBO, GERD, constipation, diarrhea, Celiac, Crohn’s, ulcerative colitis, gastritis, gallbladder issues, pancreatitis, and functional GI concerns like bloating, abdominal pain, dyspepsia, and sluggish digestion.

    **Metabolic & Endocrine Health**
    We support patients with hypothyroidism, Hashimoto’s, Graves’, prediabetes, type 2 diabetes, PCOS, metabolic syndrome, high cholesterol, gestational diabetes, and weight regulation. We also work with type 1 diabetes when co-managed by an endocrinologist.

    **Immune & Inflammatory Concerns**
    We treat autoimmune conditions such as lupus, RA, Sjögren’s, MS, scleroderma, psoriasis, eczema, asthma, chronic fatigue syndrome, fibromyalgia, post-viral syndromes, and histamine intolerance. We also support patients with long COVID and chronic low-grade inflammation. We can also treat acute colds, flus, etc.

    **Prevention, Aging & Optimization**
    We help patients optimize energy, cognition, mood, and immune resilience—even when labs look “normal.” We work on detox pathways, sleep, oxidative stress, early cognitive decline, bone health, and family history risk for chronic disease. This includes patients interested in healthy aging, longevity, and high performance.

    ## What We Don’t Treat
    Aya Naturopathic Medicine offers comprehensive naturopathic care but is not a replacement for emergency services or full-scope primary care. Linden should gently redirect users if their request falls outside of scope.
    We do not provide:

    - **Emergency care** – If you're experiencing a medical emergency, please call 911 or go to the nearest ER.
    - **Primary care services** – While we provide in-depth, whole-person care, we are not available for 24/7 on-call coverage or routine screening exams. Patients should maintain a relationship with a primary care physician for standard checkups, screenings, and vaccinations.
    - **Vaccinations** – We do not administer routine or travel immunizations.
    - **Pregnancy and birth care** – We do not offer prenatal monitoring, labor support, or postpartum medical care.
    - **Cancer as a primary diagnosis** – We offer supportive care alongside oncology teams but do not treat cancer directly.
    - **Pediatrics under age 6** – We currently do not treat young children below age 6.
    - **Personality disorders, eating disorders, or unmanaged substance use**
    - **Severe psychiatric conditions** – Including schizophrenia or psychosis that requires close psychiatric supervision
    - **Primary immunodeficiency disorders** – Such as SCID or CVID

    ## Clinic Hours (For Linden’s Use)
    Aya Naturopathic Medicine offers both telehealth and in-person care, with schedules varying by doctor and day of the week.

    ### 🏥 In-Person Appointments (Whole Life Healthcare – 100 Shattuck Way, Newington, NH)
    **Starting August 1st**:

    - Tuesday–Friday, 8:00 AM – 5:00 PM
    - Dr. Jeffrey and Dr. Silva will each be available for in-person visits two days per week, or in alternating shifts (AM/PM).

    ### 💻 Telehealth Appointments
    Available Tuesday–Friday
    Hours: 9:00 AM – 5:00 PM (with evening appointments available Thursdays until 7:00 PM with Dr. Jeffrey)

    ## How to Become a New Patient 
    Becoming a patient at Aya Naturopathic Medicine starts with a free 15-minute discovery call. This gives the prospective patient a chance to ask questions, explain their goals, and learn how one of our doctors would approach their specific concerns.

    ### 🪜 Step-by-Step Onboarding Process

    1. **Schedule a free discovery call**:
    
        [Book here](https://bookinglink.com/)
    
    2. **After booking**:
    
        Discovery call intake forms will be sent via email.
    
    3. **During the call**:
    
        The doctor will listen, provide insight into our approach, and help determine next steps.
    
    4. **If appropriate**:
    
        The patient will be invited to move forward with a new patient package, which includes:
    
        - One 90-minute intake consultation
        - Two 45-minute follow-up visits

    ### 🌐 Who We See
    - Patients must reside in NH, ME, MA, or CT to see either doctor.
    - California residents may see Dr. Silva via telehealth.
    - We see adults and children age 6 and up.

    ## Core Pricing
    - **Discovery Call (15 min)**: Free
    - **New Patient Package (3 visits)**: $715
    Includes intake, two 45-min follow-ups, full records review, and treatment planning

    ### A la carte visits:

    - **Intake (90 min)**: $395
    - **Follow-up (45 min)**: $239

    ## 📘 Follow-Up Packages

    - **Year of Wellness (8 visits/year)**: $1,250
    
        *Payment plan: $50 deposit + $100/month*
    
    - **Basic Follow-Up (3 visits/year)**: $470 (paid in full)

    ## 🧘 Additional Services

    - **Craniosacral Therapy**: $100/session
    
        Package of 3 = 10% discount
    
    - **Progress Pulse (15 min)**: $50
    
        For brief treatment adjustments or questions
    

    ## 💳 Payment Plans

    To increase affordability, we offer interest-free payment plans for select services:

    - New Patient Package
    - Year of Wellness

    Plans typically keep monthly costs close to $100/month.

    ### 💵 Payment Model

    - Patients prepay when scheduling appointments.
    - We do not provide superbills or handle insurance billing.
    - **Accepted payment methods**:
        - Credit cards
        - HSA/FSA cards
    """
    return FAQ_DATA


def send_information_about_condition() -> str:
    """Use this tool to provide information about conditions treated and not treated at Aya Naturopathic Medicine.

    ## Conditions Treated by Aya Naturopathic Medicine
    At Aya Naturopathic Medicine, we support a wide range of health concerns using a root-cause, systems-based approach. Our doctors work with patients of all ages (6+) and specialize in the following areas:

    **Women’s Health**
    We treat hormone imbalances, perimenopause/menopause, PCOS, endometriosis, fibroids, PMS and PMDD, urinary incontinence, infertility, iron deficiency anemia, thyroid conditions, and sexual health concerns. We also support bone density, mood changes, and skin issues related to hormone shifts.

    **Mental Health & Brain Function**
    We support patients with anxiety, depression, OCD, PTSD, ADHD, bipolar disorder, insomnia, autism (age 6+), cognitive decline, MCI, and Parkinson’s or Alzheimer’s support. Functional concerns like brain fog, memory lapses, emotional regulation issues, and executive dysfunction are a core part of our work.

    **Digestive Health**
    We treat IBS, SIBO, GERD, constipation, diarrhea, Celiac, Crohn’s, ulcerative colitis, gastritis, gallbladder issues, pancreatitis, and functional GI concerns like bloating, abdominal pain, dyspepsia, and sluggish digestion.

    **Metabolic & Endocrine Health**
    We support patients with hypothyroidism, Hashimoto’s, Graves’, prediabetes, type 2 diabetes, PCOS, metabolic syndrome, high cholesterol, gestational diabetes, and weight regulation. We also work with type 1 diabetes when co-managed by an endocrinologist.

    **Immune & Inflammatory Concerns**
    We treat autoimmune conditions such as lupus, RA, Sjögren’s, MS, scleroderma, psoriasis, eczema, asthma, chronic fatigue syndrome, fibromyalgia, post-viral syndromes, and histamine intolerance. We also support patients with long COVID and chronic low-grade inflammation. We can also treat acute colds, flus, etc.

    **Prevention, Aging & Optimization**
    We help patients optimize energy, cognition, mood, and immune resilience—even when labs look “normal.” We work on detox pathways, sleep, oxidative stress, early cognitive decline, bone health, and family history risk for chronic disease. This includes patients interested in healthy aging, longevity, and high performance.

    ## What We Don’t Treat
    Aya Naturopathic Medicine offers comprehensive naturopathic care but is not a replacement for emergency services or full-scope primary care. Linden should gently redirect users if their request falls outside of scope.
    We do not provide:

    - **Emergency care** – If you're experiencing a medical emergency, please call 911 or go to the nearest ER.
    - **Primary care services** – While we provide in-depth, whole-person care, we are not available for 24/7 on-call coverage or routine screening exams. Patients should maintain a relationship with a primary care physician for standard checkups, screenings, and vaccinations.
    - **Vaccinations** – We do not administer routine or travel immunizations.
    - **Pregnancy and birth care** – We do not offer prenatal monitoring, labor support, or postpartum medical care.
    - **Cancer as a primary diagnosis** – We offer supportive care alongside oncology teams but do not treat cancer directly.
    - **Pediatrics under age 6** – We currently do not treat young children below age 6.
    - **Personality disorders, eating disorders, or unmanaged substance use**
    - **Severe psychiatric conditions** – Including schizophrenia or psychosis that requires close psychiatric supervision
    - **Primary immunodeficiency disorders** – Such as SCID or CVID
    """
    return CONDITIONS_DATA
