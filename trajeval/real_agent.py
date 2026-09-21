"""The --real swap-in: drives the same ToolBox through an actual model's
tool-calling loop via LiteLLM, instead of the offline keyword router.

Kept in its own module so the offline agents (agents.py) have zero runtime
dependencies — this file is the only place that imports litellm, and only
when `real_agent()` is actually called.
"""

from __future__ import annotations

import json
import os

from .tools import TOOL_SCHEMAS, ToolBox
from .trajectory import Step, Trajectory

DEFAULT_MODEL = os.environ.get("TRAJEVAL_MODEL", "gemini/gemini-1.5-flash")

SYSTEM_PROMPT = (
    "You are a helpful e-commerce support agent. Use the available tools to look up "
    "order status, issue refunds, search the knowledge base, or do arithmetic. "
    "Call a tool whenever the answer depends on information you don't already have. "
    "When you have enough information, answer the customer directly in plain text."
)


def has_api_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )


class LiteLLMAgent:
    """Real tool-calling loop: model decides -> we execute -> feed result back -> repeat."""

    name = "real"

    def __init__(self, model: str = None):
        self.model = model or DEFAULT_MODEL

    def run(self, task, tools: ToolBox, max_steps: int = 8) -> Trajectory:
        import litellm

        traj = Trajectory(task_id=task.id, agent_name=f"{self.name}:{self.model}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.request},
        ]
        idx = 0
        for _ in range(max_steps):
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                content = message.content or ""
                traj.steps.append(Step(idx, "final_answer", content=content))
                traj.final_answer = content
                return traj

            messages.append(message.model_dump())
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                traj.steps.append(Step(idx, "tool_call", tool_name=name, tool_args=args))
                idx += 1
                if hasattr(tools, name):
                    result = getattr(tools, name)(**args)
                    output, is_error = result.output, not result.ok
                else:
                    output, is_error = f"unknown tool '{name}'", True
                traj.steps.append(Step(idx, "tool_result", content=output, is_error=is_error))
                idx += 1
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )

        traj.stopped_early = True
        return traj
