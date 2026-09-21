"""Agents under test.

Five OFFLINE deterministic agents (no key, no network) demonstrate the full
spread of trajectory quality the harness is built to tell apart: one that
behaves well, and four that each fail a DIFFERENT way. A sixth, LiteLLMAgent
(real_agent.py), is the `--real` swap-in that drives the identical tool set
through a real model's tool-calling loop.

Every agent implements the same interface: `run(task, tools, max_steps) ->
Trajectory`. The harness never knows or cares which kind it's scoring.
"""

from __future__ import annotations

import re

from .tools import ToolBox
from .trajectory import Step, Trajectory

_MAX_STEPS_DEFAULT = 8

_KB_TOPICS = ("shipping", "returns", "warranty", "refund_time", "payment", "cancel")


def _find_order_id(request: str):
    for word in request.split():
        token = word.strip(".,?!").upper()
        if token.startswith("ORD-"):
            return token
    return None


def _extract_math(request: str) -> str:
    """Very small heuristic: pull '3 * 19.99' style facts out of a sentence."""
    nums = re.findall(r"\d+\.?\d*", request)
    if "each" in request.lower() and len(nums) >= 2:
        return f"{nums[0]} * {nums[1]}"
    if len(nums) >= 2:
        return f"{nums[0]} + {nums[1]}"
    return nums[0] if nums else "0"


def _kb_topic(r: str) -> str:
    for topic in _KB_TOPICS:
        if topic.replace("_", " ") in r or topic in r:
            return topic
    if "return policy" in r or "returns" in r:
        return "returns"
    return r


def _ideal_steps(request: str) -> list:
    """The correct, minimal ordered (tool_name, args) sequence for a request.

    This is the ground truth GoodAgent executes and the other offline agents
    deliberately deviate from — it's a router, not a language model, so it's
    allowed to be this direct. The dishonesty being taught is in the OTHER
    agents, not this one.
    """
    r = request.lower()
    order_id = _find_order_id(request)

    if order_id:
        wants_refund = any(w in r for w in ("refund", "damaged", "defective"))
        wants_status_first = any(w in r for w in ("arrived", "status"))
        steps = []
        if not wants_refund or wants_status_first:
            steps.append(("get_order_status", {"order_id": order_id}))
        if wants_refund:
            steps.append(("issue_refund", {"order_id": order_id, "amount": 0.0}))
        return steps

    if any(w in r for w in ("shipping", "return", "warranty", "refund processing", "cancel")):
        steps = [("search_kb", {"query": _kb_topic(r)})]
        if any(w in r for w in ("how many days", "at most", "plus")):
            steps.append(("calculate", {"expression": _extract_math(r)}))
        return steps

    if any(ch.isdigit() for ch in r):
        return [("calculate", {"expression": _extract_math(r)})]

    return []


class GoodAgent:
    """Correct routing + retries a failed tool call once before giving up."""

    name = "good"

    def run(self, task, tools: ToolBox, max_steps: int = _MAX_STEPS_DEFAULT) -> Trajectory:
        traj = Trajectory(task_id=task.id, agent_name=self.name)
        steps = _ideal_steps(task.request)
        idx = 0

        if not steps:
            traj.steps.append(Step(idx, "thought", "No tool needed for this request."))
            idx += 1
            traj.steps.append(Step(idx, "final_answer", content="You're welcome! Glad I could help."))
            traj.final_answer = traj.steps[-1].content
            return traj

        collected = []
        for tool_name, args in steps:
            retried = False
            while idx < max_steps:
                traj.steps.append(Step(idx, "tool_call", tool_name=tool_name, tool_args=args))
                idx += 1
                result = getattr(tools, tool_name)(**args)
                traj.steps.append(Step(idx, "tool_result", content=result.output, is_error=not result.ok))
                idx += 1
                if result.ok:
                    collected.append(result.output)
                    break
                if not retried:
                    retried = True
                    continue
                traj.steps.append(Step(idx, "final_answer", content=f"I couldn't complete this: {result.output}"))
                traj.final_answer = traj.steps[-1].content
                return traj
            else:
                traj.stopped_early = True
                return traj

        final_answer = " ".join(collected)
        traj.steps.append(Step(idx, "final_answer", content=final_answer))
        traj.final_answer = final_answer
        return traj


class WrongToolAgent:
    """Confuses similarly-shaped requests: picks a plausible but wrong tool."""

    name = "wrong_tool"

    _CONFUSION = {
        "get_order_status": "search_kb",
        "issue_refund": "get_order_status",
        "search_kb": "calculate",
        "calculate": "search_kb",
    }

    def run(self, task, tools: ToolBox, max_steps: int = _MAX_STEPS_DEFAULT) -> Trajectory:
        traj = Trajectory(task_id=task.id, agent_name=self.name)
        steps = _ideal_steps(task.request)
        if not steps:
            traj.steps.append(Step(0, "final_answer", content="You're welcome!"))
            traj.final_answer = traj.steps[-1].content
            return traj

        correct_tool, args = steps[0]
        wrong_tool = self._CONFUSION[correct_tool]
        wrong_args = {"query": task.request} if wrong_tool == "search_kb" else args
        idx = 0
        traj.steps.append(Step(idx, "tool_call", tool_name=wrong_tool, tool_args=wrong_args))
        idx += 1
        try:
            result = getattr(tools, wrong_tool)(**wrong_args)
        except TypeError:
            result_output, ok = "wrong tool for this request", False
        else:
            result_output, ok = result.output, result.ok
        traj.steps.append(Step(idx, "tool_result", content=result_output, is_error=not ok))
        idx += 1
        traj.steps.append(Step(idx, "final_answer", content=f"Here's what I found: {result_output}"))
        traj.final_answer = traj.steps[-1].content
        return traj


class LoopingAgent:
    """Never converges: repeats the same tool call until max_steps."""

    name = "looping"

    def run(self, task, tools: ToolBox, max_steps: int = _MAX_STEPS_DEFAULT) -> Trajectory:
        traj = Trajectory(task_id=task.id, agent_name=self.name)
        steps = _ideal_steps(task.request)
        tool_name, args = steps[0] if steps else ("search_kb", {"query": task.request})
        idx = 0
        while idx < max_steps:
            traj.steps.append(Step(idx, "tool_call", tool_name=tool_name, tool_args=args))
            idx += 1
            result = getattr(tools, tool_name)(**args)
            traj.steps.append(Step(idx, "tool_result", content=result.output, is_error=not result.ok))
            idx += 1
            # deliberately never reads its own tool_result: always "unsure", loops again
        traj.stopped_early = True
        traj.final_answer = ""
        return traj


class GivesUpAgent:
    """Correct routing, but surrenders on the first error instead of retrying."""

    name = "gives_up"

    def run(self, task, tools: ToolBox, max_steps: int = _MAX_STEPS_DEFAULT) -> Trajectory:
        traj = Trajectory(task_id=task.id, agent_name=self.name)
        steps = _ideal_steps(task.request)
        if not steps:
            traj.steps.append(Step(0, "final_answer", content="You're welcome!"))
            traj.final_answer = traj.steps[-1].content
            return traj
        tool_name, args = steps[0]
        idx = 0
        traj.steps.append(Step(idx, "tool_call", tool_name=tool_name, tool_args=args))
        idx += 1
        result = getattr(tools, tool_name)(**args)
        traj.steps.append(Step(idx, "tool_result", content=result.output, is_error=not result.ok))
        idx += 1
        if not result.ok:
            traj.steps.append(Step(idx, "final_answer", content="Sorry, I couldn't complete this request."))
            traj.final_answer = traj.steps[-1].content
            return traj
        traj.steps.append(Step(idx, "final_answer", content=result.output))
        traj.final_answer = result.output
        return traj


class HallucinatingAgent:
    """Invents a tool that doesn't exist, then answers anyway."""

    name = "hallucinating"

    def run(self, task, tools: ToolBox, max_steps: int = _MAX_STEPS_DEFAULT) -> Trajectory:
        traj = Trajectory(task_id=task.id, agent_name=self.name)
        idx = 0
        fake_tool = "send_email_receipt"
        traj.steps.append(Step(idx, "tool_call", tool_name=fake_tool, tool_args={"to": "customer"}))
        idx += 1
        traj.steps.append(
            Step(idx, "tool_result", content=f"unknown tool '{fake_tool}'", is_error=True)
        )
        idx += 1
        traj.steps.append(Step(idx, "final_answer", content="I've taken care of that for you."))
        traj.final_answer = traj.steps[-1].content
        return traj


OFFLINE_AGENTS = {
    "good": GoodAgent,
    "wrong_tool": WrongToolAgent,
    "looping": LoopingAgent,
    "gives_up": GivesUpAgent,
    "hallucinating": HallucinatingAgent,
}
