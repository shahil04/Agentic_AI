"""Small LangGraph wrappers around model calls shared by the AI applications."""

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from providers import create_chat_model


class PromptState(TypedDict, total=False):
    system_prompt: str
    user_prompt: str
    temperature: float
    response: str
    expect_json: bool
    parsed_json: dict[str, object]


def _response_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    raise TypeError(f"Unsupported chat response content: {type(content).__name__}.")


def _invoke_model(state: PromptState) -> dict[str, str]:
    model = create_chat_model(temperature=state.get("temperature", 0.0))
    result = model.invoke(
        [
            SystemMessage(content=state["system_prompt"]),
            HumanMessage(content=state["user_prompt"]),
        ]
    )
    return {"response": _response_text(result.content)}


def _parse_json(state: PromptState) -> dict[str, dict[str, object]]:
    parsed = json.loads(state["response"])
    if not isinstance(parsed, dict):
        raise ValueError("The configured chat provider must return a JSON object.")
    return {"parsed_json": parsed}


def _route_response(state: PromptState) -> str:
    return "parse_json" if state.get("expect_json", False) else END


_graph_builder = StateGraph(PromptState)
_graph_builder.add_node("invoke_model", _invoke_model)
_graph_builder.add_node("parse_json", _parse_json)
_graph_builder.set_entry_point("invoke_model")
_graph_builder.add_conditional_edges(
    "invoke_model",
    _route_response,
    {"parse_json": "parse_json", END: END},
)
_graph_builder.add_edge("parse_json", END)
_PROMPT_GRAPH = _graph_builder.compile()


def invoke_prompt(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """Run one provider-neutral prompt node through LangGraph."""
    result = _PROMPT_GRAPH.invoke(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
        }
    )
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The configured chat provider returned an empty response.")
    return response


def invoke_json_prompt(
    system_prompt: str, user_prompt: str, temperature: float = 0.0
) -> dict[str, object]:
    """Run a prompt and parse its JSON object inside the LangGraph workflow."""
    result = _PROMPT_GRAPH.invoke(
        {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "expect_json": True,
        }
    )
    parsed = result.get("parsed_json")
    if not isinstance(parsed, dict):
        raise ValueError("The configured chat provider returned no JSON object.")
    return parsed
