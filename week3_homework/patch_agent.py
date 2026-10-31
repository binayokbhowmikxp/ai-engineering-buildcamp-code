"""Track completed Agent.run calls using the course's token-price formula."""

import inspect
from collections import Counter
from functools import wraps

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse


# USD per million tokens, preserved from 01-foundation/02-rag/openai.ipynb.
MODEL_PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-nano": {"input": 0.075, "output": 0.30},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
}

model_usage = {}
agent_runs = Counter()
_original_run = None


def calculate_cost(model_name, input_tokens, output_tokens):
    # Provider responses can include the dated snapshot rather than the alias.
    name = next(
        (name for name in MODEL_PRICES
         if model_name.lower() == name or model_name.lower().startswith(name + "-20")),
        None,
    )
    if name is None:
        raise ValueError(f"No course price configured for {model_name!r}")
    prices = MODEL_PRICES[name]
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def reset_usage():
    model_usage.clear()
    agent_runs.clear()


def patch_agent():
    """Patch the Agent class once, covering SQL-agent and judge instances."""
    global _original_run
    if _original_run is not None:
        return
    _original_run = Agent.run
    original_run = _original_run

    @wraps(original_run)
    async def tracked_run(self, *args, **kwargs):
        # Preserve 2.63's caller-based name inference despite this extra frame.
        if kwargs.get("infer_name", True) and self.name is None:
            self._infer_name(inspect.currentframe())
        result = await original_run(self, *args, **kwargs)
        # New ModelResponses include every turn of the tool loop, without
        # counting prior conversation history again.
        for message in result.new_messages():
            if not isinstance(message, ModelResponse):
                continue
            name = message.model_name
            usage = message.usage
            totals = model_usage.setdefault(name, Counter())
            totals["requests"] += 1
            totals["input_tokens"] += usage.input_tokens
            totals["output_tokens"] += usage.output_tokens
            totals["total_tokens"] += usage.input_tokens + usage.output_tokens
            totals["cost"] += calculate_cost(name, usage.input_tokens, usage.output_tokens)
        agent_runs[self.name or "unnamed"] += 1
        return result

    Agent.run = tracked_run


def unpatch_agent():
    global _original_run
    if _original_run is not None:
        Agent.run = _original_run
        _original_run = None


def usage_summary():
    lines = [
        "Pricing: course MODEL_PRICES; USD; standard input/output token rates.",
        "Cached-input discounts are not applied (course approximation).",
        "Completed agent runs: " + ", ".join(
            f"{name}={count}" for name, count in sorted(agent_runs.items())
        ),
    ]
    total = Counter()
    for name, usage in sorted(model_usage.items()):
        total.update(usage)
        lines.append(
            f"{name}: requests={usage['requests']}, input_tokens={usage['input_tokens']}, "
            f"output_tokens={usage['output_tokens']}, total_tokens={usage['total_tokens']}, "
            f"estimated_cost=${usage['cost']:.8f}"
        )
    lines.append(
        f"TOTAL: requests={total['requests']}, input_tokens={total['input_tokens']}, "
        f"output_tokens={total['output_tokens']}, total_tokens={total['total_tokens']}"
    )
    lines.append(f"Approximate total cost: ${total['cost']:.8f}")
    return "\n".join(lines)
