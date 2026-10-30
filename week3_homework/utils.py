from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ToolCallPart


@dataclass
class ToolCall:
    name: str
    args: Any


def collect_tools(messages):
    """Collect application tool calls in message order, excluding final output."""
    tool_calls = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                # PydanticAI's default structured-output tool (including variants).
                if part.tool_name == "final_result" or part.tool_name.startswith("final_result_"):
                    continue
                tool_calls.append(ToolCall(name=part.tool_name, args=part.args_as_dict()))
    return tool_calls
