import logging
from typing import Any, Callable, Optional

import google.genai as genai
from google.genai.types import Tool

from src.shared.enums import InteractionType
from src.shared.schemas import InteractionMessage

logger = logging.getLogger(__name__)


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
        # Create tool from function
        tool = Tool.from_function(tool_function)
        
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=genai_messages,
            system_instruction=full_system_prompt,
            tools=[tool],
        )

        tool_results = {}
        
        # Extract tool calls and results
        if response.candidates and response.candidates[0].content.parts:
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
        logger.error(f"Error in call_single_tool: {e}")
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
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=genai_messages,
            system_instruction=full_system_prompt,
        )

        if response.candidates and response.candidates[0].content.parts:
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

