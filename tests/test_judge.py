from trajeval.agents import GoodAgent, HallucinatingAgent, LoopingAgent, WrongToolAgent
from trajeval.judge import JudgeVerdict, offline_judge
from trajeval.tasks import TASKS_BY_ID
from trajeval.tools import ToolBox


def test_offline_judge_returns_a_valid_verdict_shape():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    verdict = offline_judge(traj, task)
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.efficiency_rating in ("efficient", "acceptable", "wasteful")


def test_offline_judge_approves_good_agent():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    verdict = offline_judge(traj, task)
    assert verdict.tool_use_correct is True
    assert verdict.completed_task is True
    assert verdict.recovered_from_errors is None


def test_offline_judge_flags_wrong_tool_use():
    task = TASKS_BY_ID["order_status_simple"]
    traj = WrongToolAgent().run(task, ToolBox())
    verdict = offline_judge(traj, task)
    assert verdict.tool_use_correct is False


def test_offline_judge_calls_looping_agent_wasteful():
    task = TASKS_BY_ID["kb_shipping"]
    traj = LoopingAgent().run(task, ToolBox(), max_steps=6)
    verdict = offline_judge(traj, task)
    assert verdict.efficiency_rating == "wasteful"
    assert verdict.completed_task is False


def test_offline_judge_flags_hallucinated_tool():
    task = TASKS_BY_ID["order_status_simple"]
    traj = HallucinatingAgent().run(task, ToolBox())
    verdict = offline_judge(traj, task)
    assert verdict.tool_use_correct is False
    assert verdict.completed_task is False


def test_judge_verdict_rejects_unknown_fields():
    import pytest

    with pytest.raises(Exception):
        JudgeVerdict(
            tool_use_correct=True,
            completed_task=True,
            recovered_from_errors=None,
            efficiency_rating="efficient",
            rationale="ok",
            extra_field="not allowed",
        )
