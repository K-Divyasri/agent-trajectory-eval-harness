"""The harness: run an agent over the whole task suite, score every
trajectory, and roll it up into a Scorecard — plus a JSONL history log and a
CI ship gate whose exit code IS the pass/fail signal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import metrics
from .judge import JudgeVerdict, judge_trajectory
from .tasks import TASKS
from .tools import ToolBox
from .trajectory import Trajectory


@dataclass
class TaskResult:
    task_id: str
    trajectory: Trajectory
    score: metrics.TrajectoryScore
    verdict: Optional[JudgeVerdict]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "trajectory": self.trajectory.to_dict(),
            "score": self.score.to_dict(),
            "verdict": self.verdict.model_dump() if self.verdict else None,
        }


@dataclass
class Scorecard:
    agent_name: str
    results: list = field(default_factory=list)

    @property
    def avg_tool_call_correctness(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score.tool_call_correctness for r in self.results) / len(self.results)

    @property
    def task_completion_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.score.task_completion) / len(self.results)

    @property
    def avg_step_efficiency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score.step_efficiency for r in self.results) / len(self.results)

    @property
    def error_recovery_rate(self) -> Optional[float]:
        applicable = [r for r in self.results if r.score.error_recovery is not None]
        if not applicable:
            return None
        return sum(1 for r in applicable if r.score.error_recovery) / len(applicable)

    @property
    def judge_agreement_rate(self) -> Optional[float]:
        with_verdict = [r for r in self.results if r.verdict is not None]
        if not with_verdict:
            return None
        agree = sum(1 for r in with_verdict if r.verdict.completed_task == r.score.task_completion)
        return agree / len(with_verdict)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "avg_tool_call_correctness": self.avg_tool_call_correctness,
            "task_completion_rate": self.task_completion_rate,
            "avg_step_efficiency": self.avg_step_efficiency,
            "error_recovery_rate": self.error_recovery_rate,
            "judge_agreement_rate": self.judge_agreement_rate,
            "n_tasks": len(self.results),
        }


def run_harness(
    agent,
    tasks=None,
    judge_backend: str = "offline",
    judge_model: str = None,
) -> Scorecard:
    tasks = tasks if tasks is not None else TASKS
    scorecard = Scorecard(agent_name=getattr(agent, "name", agent.__class__.__name__))
    for task in tasks:
        tools = ToolBox()
        traj = agent.run(task, tools)
        score = metrics.score_trajectory(traj, task)
        verdict = judge_trajectory(traj, task, backend=judge_backend, model=judge_model)
        scorecard.results.append(TaskResult(task.id, traj, score, verdict))
    return scorecard


def save_history(scorecard: Scorecard, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(scorecard.to_dict()) + "\n")


def load_history(path: str) -> list:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gate(
    scorecard: Scorecard,
    min_completion: float = 0.7,
    min_tool_correctness: float = 0.6,
    min_efficiency: float = 0.5,
) -> bool:
    """Ship gate: True (pass) only if every threshold is met."""
    if scorecard.task_completion_rate < min_completion:
        return False
    if scorecard.avg_tool_call_correctness < min_tool_correctness:
        return False
    if scorecard.avg_step_efficiency < min_efficiency:
        return False
    return True
