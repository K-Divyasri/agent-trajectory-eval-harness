"""Programmatic trajectory metrics — the non-LLM half of the harness.

These are cheap, deterministic, and always available; `judge.py` adds an
LLM-as-judge layer on top that reads the same trajectory holistically.
Keeping both is the point: programmatic metrics catch what they're built to
catch, a judge can catch what you didn't think to hand-code a rule for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .trajectory import Trajectory


def tool_call_correctness(traj: Trajectory, task) -> float:
    """F1 over the multiset of tool names called vs. the expected minimal set.

    Order-blind on purpose: retries after a recoverable error legitimately
    repeat a tool name, and shouldn't be punished the way a wrong tool would
    be. Returns 1.0 for a no-tool task that correctly calls no tools.
    """
    called = traj.tool_call_names()
    expected = task.expected_tools

    if not expected and not called:
        return 1.0
    if not called:
        return 0.0

    called_set = set(called)
    expected_set = set(expected)
    if not expected_set:
        # expected no tools but agent called some -> 0 precision, undefined recall
        return 0.0

    true_positives = len(called_set & expected_set)
    precision = true_positives / len(called_set) if called_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def task_completion(traj: Trajectory, task) -> bool:
    if traj.stopped_early or not traj.final_answer:
        return False
    return bool(task.check(traj.final_answer))


def error_recovery(traj: Trajectory) -> Optional[bool]:
    """None if no tool error occurred (recovery wasn't applicable).

    True if, after a tool call errors, the SAME tool is called again and
    later succeeds. False if it errors and the trajectory ends (or the same
    tool is never retried) without a subsequent success.
    """
    # pair each tool_call with the tool_result that immediately follows it
    pairs = []
    for step in traj.steps:
        if step.kind == "tool_call":
            result = next(
                (s for s in traj.steps if s.kind == "tool_result" and s.index == step.index + 1),
                None,
            )
            if result is not None:
                pairs.append((step.tool_name, not result.is_error))

    if not any(not ok for _, ok in pairs):
        return None  # no error ever occurred

    first_error_tool = next(name for name, ok in pairs if not ok)
    seen_first_error = False
    for name, ok in pairs:
        if name != first_error_tool:
            continue
        if not seen_first_error:
            seen_first_error = True  # this pair IS the first error, not a retry
            continue
        if ok:
            return True
    return False


def step_efficiency(traj: Trajectory, task) -> float:
    """min_steps / actual tool-call count, capped at 1.0. 1.0 for a 0-tool task done in 0 calls."""
    actual = len(traj.tool_calls())
    if task.min_steps == 0:
        return 1.0 if actual == 0 else 0.0
    if actual == 0:
        return 0.0
    return min(1.0, task.min_steps / actual)


@dataclass
class TrajectoryScore:
    tool_call_correctness: float
    task_completion: bool
    error_recovery: Optional[bool]
    step_efficiency: float
    stopped_early: bool

    def to_dict(self) -> dict:
        return {
            "tool_call_correctness": self.tool_call_correctness,
            "task_completion": self.task_completion,
            "error_recovery": self.error_recovery,
            "step_efficiency": self.step_efficiency,
            "stopped_early": self.stopped_early,
        }


def score_trajectory(traj: Trajectory, task) -> TrajectoryScore:
    return TrajectoryScore(
        tool_call_correctness=tool_call_correctness(traj, task),
        task_completion=task_completion(traj, task),
        error_recovery=error_recovery(traj),
        step_efficiency=step_efficiency(traj, task),
        stopped_early=traj.stopped_early,
    )
