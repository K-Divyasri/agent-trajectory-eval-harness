"""The shape of an agent run: a Trajectory is a list of Steps.

This is the object every metric and every judge reads. Keeping it a plain,
backend-agnostic record (not tied to OpenAI's or Anthropic's wire format) is
what lets the same harness score a fake offline agent and a real LiteLLM
agent identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

StepKind = Literal["thought", "tool_call", "tool_result", "final_answer"]


@dataclass
class Step:
    index: int
    kind: StepKind
    content: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    is_error: bool = False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "kind": self.kind,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "is_error": self.is_error,
        }


@dataclass
class Trajectory:
    task_id: str
    agent_name: str
    steps: list = field(default_factory=list)
    final_answer: str = ""
    stopped_early: bool = False

    def tool_calls(self) -> list:
        return [s for s in self.steps if s.kind == "tool_call"]

    def tool_call_names(self) -> list:
        return [s.tool_name for s in self.tool_calls()]

    def had_tool_error(self) -> bool:
        return any(s.kind == "tool_result" and s.is_error for s in self.steps)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "stopped_early": self.stopped_early,
        }


def render_transcript(traj: Trajectory) -> str:
    """Human-readable transcript, the thing an LLM judge actually reads."""
    lines = [f"Task: {traj.task_id}", f"Agent: {traj.agent_name}", ""]
    for step in traj.steps:
        if step.kind == "thought":
            lines.append(f"[{step.index}] THOUGHT: {step.content}")
        elif step.kind == "tool_call":
            lines.append(f"[{step.index}] TOOL_CALL: {step.tool_name}({step.tool_args})")
        elif step.kind == "tool_result":
            tag = "ERROR" if step.is_error else "RESULT"
            lines.append(f"[{step.index}] TOOL_{tag}: {step.content}")
        else:
            lines.append(f"[{step.index}] FINAL_ANSWER: {step.content}")
    if traj.stopped_early:
        lines.append("(agent stopped early: hit max_steps without a final answer)")
    return "\n".join(lines)
