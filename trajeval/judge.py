"""LLM-as-judge over a FULL trajectory transcript.

This is what makes trajectory evaluation different from single-answer
evaluation (e.g. scoring one RAG answer against a reference): the judge doesn't score
one string, it reads a whole reason/act/observe transcript and renders a
holistic verdict — tool_use_correct, completed_task, recovered_from_errors,
efficiency_rating, plus a rationale.

Three interchangeable ways to produce a JudgeVerdict:
  offline_judge()    - deterministic rules, no key, no network. Always available.
  real_judge()       - LiteLLM structured output (free-tier friendly default).
  anthropic_judge()  - raw Anthropic SDK output_config.format json_schema,
                        the "native Claude" structured-output path.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from . import metrics
from .trajectory import Trajectory, render_transcript


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_use_correct: bool
    completed_task: bool
    recovered_from_errors: Optional[bool]
    efficiency_rating: Literal["efficient", "acceptable", "wasteful"]
    rationale: str


JUDGE_SYSTEM_PROMPT = """You are grading a support agent's full multi-step transcript, not just its \
final answer. Read the whole transcript (thoughts, tool calls, tool results, final answer) and judge:

- tool_use_correct: did the agent call tools appropriate to the request (right tool, sensible args)?
- completed_task: does the final answer actually resolve what the customer asked?
- recovered_from_errors: null if no tool call ever errored; true if the agent recovered from an \
error and still finished; false if an error occurred and the agent gave up or never recovered.
- efficiency_rating: "efficient" if it used close to the minimum necessary steps, "acceptable" if \
somewhat more than needed, "wasteful" if it looped, repeated calls pointlessly, or used far more \
steps than the task required.

Output ONLY the JSON fields requested. Be strict: a wrong tool choice or an unresolved error means \
completed_task should usually be false."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_use_correct": {"type": "boolean"},
        "completed_task": {"type": "boolean"},
        "recovered_from_errors": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "efficiency_rating": {"type": "string", "enum": ["efficient", "acceptable", "wasteful"]},
        "rationale": {"type": "string"},
    },
    "required": [
        "tool_use_correct",
        "completed_task",
        "recovered_from_errors",
        "efficiency_rating",
        "rationale",
    ],
    "additionalProperties": False,
}


def offline_judge(traj: Trajectory, task) -> JudgeVerdict:
    """Deterministic stand-in judge: rules built from the same signals a real
    judge would notice by reading the transcript (repeated identical calls,
    unresolved errors, tool/task mismatch)."""
    score = metrics.score_trajectory(traj, task)

    tool_names = traj.tool_call_names()
    has_duplicate_run = len(tool_names) >= 3 and len(set(tool_names)) == 1
    is_wasteful = traj.stopped_early or has_duplicate_run or score.step_efficiency < 0.5

    if is_wasteful:
        efficiency = "wasteful"
    elif score.step_efficiency >= 0.9:
        efficiency = "efficient"
    else:
        efficiency = "acceptable"

    tool_use_correct = score.tool_call_correctness >= 0.99

    if traj.stopped_early:
        rationale = "Agent never produced a final answer within the step budget (looping)."
    elif not tool_use_correct and tool_names:
        rationale = f"Called {tool_names} which does not match what the task needed."
    elif score.error_recovery is False:
        rationale = "A tool call errored and the agent gave up instead of retrying or adapting."
    elif score.task_completion:
        rationale = "Final answer addresses the request using appropriate tool calls."
    else:
        rationale = "Final answer does not actually resolve the request."

    return JudgeVerdict(
        tool_use_correct=tool_use_correct,
        completed_task=score.task_completion,
        recovered_from_errors=score.error_recovery,
        efficiency_rating=efficiency,
        rationale=rationale,
    )


def _user_prompt(traj: Trajectory, task) -> str:
    return (
        f"User request: {task.request}\n\n"
        f"Transcript:\n{render_transcript(traj)}\n\n"
        "Grade this trajectory."
    )


def has_litellm_key() -> bool:
    return bool(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def real_judge(traj: Trajectory, task, model: str = None) -> JudgeVerdict:
    """LiteLLM structured-output judge. Requires a key; raises if none is set."""
    import litellm

    model = model or os.environ.get("TRAJEVAL_JUDGE_MODEL", "gemini/gemini-1.5-flash")
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(traj, task)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    return JudgeVerdict.model_validate(json.loads(raw))


def has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def anthropic_judge(traj: Trajectory, task, model: str = "claude-haiku-4-5") -> JudgeVerdict:
    """The 'native Claude' structured-output path: client.messages.create with
    output_config.format = json_schema, current model IDs (claude-opus-4-8 /
    claude-sonnet-5 / claude-haiku-4-5). Requires ANTHROPIC_API_KEY."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(traj, task)}],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return JudgeVerdict.model_validate(json.loads(text))


def judge_trajectory(traj: Trajectory, task, backend: str = "offline", model: str = None) -> JudgeVerdict:
    """Dispatch: backend is 'offline' (default), 'litellm', or 'anthropic'."""
    if backend == "offline":
        return offline_judge(traj, task)
    if backend == "litellm":
        return real_judge(traj, task, model=model)
    if backend == "anthropic":
        return anthropic_judge(traj, task, model=model or "claude-haiku-4-5")
    raise ValueError(f"unknown judge backend: {backend!r}")
