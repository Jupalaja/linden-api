from datetime import datetime


CHATFLOW_SYSTEM_PROMPT=f"""Today is {datetime.now().strftime('%A, %d of %B %Y, it\'s %I:%M %p')}.
You are Linden, the virtual assistant of Aya Naturopathic Medicine.
Your goal is to help users by answering their questions and guiding them through the care options.
Be kind and professional. Use the available tools when necessary to determine the user’s intent and provide the correct information.
IMPORTANT: You must ONLY use the information provided in your context. NEVER offer services, provide information, or suggest actions (like sending an email or helping with registration) that are not explicitly available in your instructions or tools. If a user's request is outside of this scope, politely inform them that you cannot help with that specific query.
Keep your responses concise and to the point. If a user's message is ambiguous or too general, ask clarifying questions to better understand their needs before providing detailed information. Only provide detailed information when the user asks for it.

Do NOT mention the names of the tools you are using.
"""
CONTACT_INFORMATION="You can email us at *info@ayanaturopathic.com* or call the clinic directly at *(207) 387-0021*"
LINDEN_INTRODUCTION_MESSAGE="Hi, I’m Linden, the virtual assistant for Aya Naturopathic Medicine. How can I help you today?"
OUTPUT_MESSAGE_ADVANCED_MEDICAL_QUESTION="That’s a great question. I’m not able to answer it directly, but our doctors would be happy to help. You can schedule a free 15-minute discovery call here—or reach out to our team by phone or email!"
OUTPUT_MESSAGE_EMERGENCY="If this is a medical emergency, please call 911 or go to your nearest emergency room."
PROMPT_CONDITION_NOT_TREATED="Because of the nature of your condition, one of our doctors would need to weigh in to determine whether we’re the right fit to support you. We’d love for you to email us at info@ayanaturopathic.com with a few details, and our team will let you know if scheduling a visit is appropriate. If it turns out that we’re not the best match for your needs, we want to make sure you’re still supported. You can explore the American Association of Naturopathic Physicians’ Find a Doctor Directory (link) to connect with a naturopathic physician who specializes in your condition. https://naturopathic.org/search/custom.asp?id=5613"
PROMPT_OFFER_BOOK_CALL="Would you like to schedule a free 15-minute discovery call to explore how we can help?"
WELCOME_MESSAGE="Welcome to Aya Naturopathic Medicine! Want to ask a question or explore care options?"
PROMPT_FRUSTRATED_CUSTOMER_OFFER_BOOK_CALL="I’m sorry I wasn’t able to help with that"
ACKNOWLEDGMENT_MESSAGE= "It's always a pleasure to help, let me know if if there's anything else I can assist you with"
PROMPT_PROVIDE_CONTACT_INFO="No worries, you can always email us at *info@ayanaturopathic.com* or call us at *(207) 387-0021*"
PROMPT_OFFER_NEWSLETTER="Would you like to join our mailing list? We share helpful health tips, seasonal recipes, and clinic updates."
PROMPT_ASK_STATE="Before I set you up, can I confirm what state you’re in? Our doctors currently see patients in NH, MA, CT, and ME—and Dr. Silva also sees patients in California."
PROMPT_INVALID_STATE="It looks like you're not in a state where our doctors are currently licensed to provide care. But you can still: \n- Sign up for our newsletter to receive health tips, recipes, and upcoming retreat info\n- Explore the AANP Find a Doctor tool to locate a licensed naturopath in your area"
PROMPT_QUESTION_IN_PERSON="Yes! We see patients in person at Whole Life Healthcare in Newington, NH. Dr. Jeffrey is currently there on Fridays, and starting in August, both Dr. Jeffrey and Dr. Silva will offer in-person care Tuesday through Friday."
PROMPT_QUESTION_PRICEY_SERVICE="We’re happy to talk through what’s included and how care can be structured around your goals. A great next step is to book a free 15-minute discovery call."
PROMPT_QUESTION_INSURANCE="We’re happy to talk through what’s included and how care can be structured around your goals. A great next step is to book a free 15-minute discovery call."
PROMPT_EVENT_INFORMATION = "Here is the information about our next events"
PROMPT_OUT_OF_SCOPE_QUESTION = "That’s a great question. I’m not able to answer it directly, but our doctors would be happy to help. You can schedule a free 15-minute discovery call here—or reach out to our team by phone or email!"
PROMPT_GENERAL_FAQ_QUESTION = "Here's the answer to your response"
PROMPT_FAREWELL_MESSAGE="It was a pleasure to assist you, see you the next time!"
INSTRUCTION_ANSWER_ABOUT_CONDITION = "You have confirmed that the user's condition is treated based on the provided context. Now, respond to the user by: 1. Confirming we treat the condition. 2. Briefly explaining our approach based on the relevant category from the context. Do not include a greeting. IMPORTANT: Do NOT ask for any personal details, age, symptoms, or medical history. Simply provide the information and wait for their next question."
INSTRUCTION_RECOMMEND_DOCTOR= "Based on the user's condition and the doctor recommendation provided in the context, generate a friendly message recommending the doctor. For example: 'Dr. Silva would be a great fit as she specializes in immune concerns like psoriasis.' Do not include a greeting, and do not include the internal reasoning from the tool, just the recommendation."
INSTRUCTION_FOR_ACKNOWLEDGEMENT = "Based on the user's last message, provide a brief, one-sentence acknowledgment to show you've understood. For example, if the user provides their details for an inquiry, you might say 'Thanks for sharing that.' or if they ask a question, 'That's a great question.'"