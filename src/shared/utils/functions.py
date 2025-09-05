import asyncio
import logging
from typing import Any, Callable, Awaitable

import google.genai as genai
from google.genai import types, errors

from src.shared.constants import GEMINI_MODEL, GEMINI_FALLBACK_MODEL
from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage

logger = logging.getLogger(__name__)

async def invoke_model_with_retries(
    generate_content_func: Callable[..., Awaitable[types.GenerateContentResponse]],
    *args: Any,
    **kwargs: Any,
) -> types.GenerateContentResponse:
    """
    Invokes a Gemini model's generate_content method with retries for server-side errors.
    If the primary model fails, it attempts to use a fallback model.
    """
    max_retries_per_model: int = 2  # 3 attempts per model
    initial_delay: float = 1.0
    backoff_factor: float = 2.0

    primary_model = kwargs.get("model", GEMINI_MODEL)
    models_to_try = [primary_model]
    if primary_model != GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL:
        models_to_try.append(GEMINI_FALLBACK_MODEL)

    last_exception = None

    for model_name in models_to_try:
        kwargs["model"] = model_name
        delay = initial_delay
        logger.info(f"Attempting to use model: {model_name}")

        for attempt in range(max_retries_per_model + 1):
            try:
                return await generate_content_func(*args, **kwargs)
            except errors.ServerError as e:
                last_exception = e
                logger.warning(
                    f"Server error on attempt {attempt + 1}/{max_retries_per_model + 1} with model {model_name}: {e}. Retrying in {delay}s..."
                )
                if attempt < max_retries_per_model:
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                else:
                    logger.error(f"All retries failed for model {model_name}.")
                    break  # Go to the next model

    logger.error("All retries for all models failed for model invocation.")
    if last_exception:
        raise last_exception

    raise RuntimeError("This line should not be reachable.")  # For mypy

async def call_single_tool(
    history_messages: list[InteractionMessage],
    client: genai.Client,
    tool_function: Callable,
    system_prompt: str,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Call a single tool function and return the tool results.

    Args:
        history_messages: The conversation history
        client: The genai client
        tool_function: The single tool function to call
        system_prompt: The system prompt
        context: Optional context to append to system prompt

    Returns:
        Dictionary containing the tool results
    """
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    # Convert messages to genai format
    genai_messages = []
    for msg in history_messages:
        genai_messages.append({
            "role": "user" if msg.role == InteractionType.USER else "model",
            "parts": [{"text": msg.message}]
        })

    try:
        config = types.GenerateContentConfig(
            tools=[tool_function],
            system_instruction=full_system_prompt,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = await invoke_model_with_retries(
            client.aio.models.generate_content,
            model=GEMINI_MODEL,
            contents=genai_messages,
            config=config
        )

        tool_results = {}

        # Extract tool calls and results
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    func_name = part.function_call.name
                    func_args = dict(part.function_call.args)

                    logger.info(f"Calling tool: {func_name} with args: {func_args}")

                    # Execute the tool function
                    try:
                        result = tool_function(**func_args)
                        tool_results[func_name] = result
                    except Exception as e:
                        logger.error(f"Error executing tool {func_name}: {e}")
                        tool_results[func_name] = None

        return tool_results

    except Exception as e:
        logger.error(f"Error in call_single_tool: {e}", exc_info=True)
        return {}


async def generate_response_text(
    history_messages: list[InteractionMessage],
    client: genai.Client,
    system_prompt: str,
    context: str | None = None,
) -> str:
    """
    Generate a response text without any tool calls.

    Args:
        history_messages: The conversation history
        client: The genai client
        system_prompt: The system prompt
        context: Optional context to append to system prompt

    Returns:
        The generated response text
    """
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    # Convert messages to genai format
    genai_messages = []
    for msg in history_messages:
        genai_messages.append({
            "role": "user" if msg.role == InteractionType.USER else "model",
            "parts": [{"text": msg.message}]
        })

    try:

        config = types.GenerateContentConfig(
            system_instruction=full_system_prompt,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = await invoke_model_with_retries(
            client.aio.models.generate_content,
            model=GEMINI_MODEL,
            contents=genai_messages,
            config=config
        )

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # Extract text from the response
            text_parts = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

            return " ".join(text_parts).strip()

        return ""

    except Exception as e:
        logger.error(f"Error in generate_response_text: {e}")
        return ""

