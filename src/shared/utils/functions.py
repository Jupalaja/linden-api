import logging
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from src.shared.schemas import InteractionMessage
from src.shared.utils.history import get_langchain_history

logger = logging.getLogger(__name__)


async def call_single_tool(
    history_messages: list[InteractionMessage],
    model: BaseChatModel,
    tool_function: Callable,
    system_prompt: str,
    context: str | None = None,
) -> dict[str, Any]:
    """
    Call a single tool function and return the tool results.

    Args:
        history_messages: The conversation history
        model: The LangChain chat model
        tool_function: The single tool function to call
        system_prompt: The system prompt
        context: Optional context to append to system prompt

    Returns:
        Dictionary containing the tool results
    """
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    langchain_messages = [
        SystemMessage(content=full_system_prompt)
    ] + get_langchain_history(history_messages)

    try:
        model_with_tools = model.bind_tools([tool_function])

        response = await model_with_tools.ainvoke(langchain_messages)

        tool_results = {}

        if isinstance(response, AIMessage) and response.tool_calls:
            for tool_call in response.tool_calls:
                func_name = tool_call["name"]
                func_args = tool_call["args"]

                logger.info(f"Calling tool: {func_name} with args: {func_args}")

                # Execute the tool function
                if func_name == tool_function.__name__:
                    try:
                        result = tool_function(**func_args)
                        tool_results[func_name] = result
                    except Exception as e:
                        logger.error(f"Error executing tool {func_name}: {e}")
                        tool_results[func_name] = None
                else:
                    logger.warning(
                        f"Model requested to call tool '{func_name}', but only '{tool_function.__name__}' is available."
                    )

        return tool_results

    except Exception as e:
        logger.error(f"Error in call_single_tool: {e}", exc_info=True)
        return {}


async def generate_response_text(
    history_messages: list[InteractionMessage],
    model: BaseChatModel,
    system_prompt: str,
    context: str | None = None,
) -> str:
    """
    Generate a response text without any tool calls.

    Args:
        history_messages: The conversation history
        model: The LangChain chat model
        system_prompt: The system prompt
        context: Optional context to append to system prompt

    Returns:
        The generated response text
    """
    full_system_prompt = system_prompt
    if context:
        full_system_prompt += f"\n\n## Context\n{context}"

    langchain_messages = [
        SystemMessage(content=full_system_prompt)
    ] + get_langchain_history(history_messages)

    try:
        response = await model.ainvoke(langchain_messages)
        return str(response.content)
    except Exception as e:
        logger.error(f"Error in generate_response_text: {e}")
        return ""
