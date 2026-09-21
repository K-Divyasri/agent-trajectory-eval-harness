from trajeval.agents import GivesUpAgent, GoodAgent, LoopingAgent
from trajeval.metrics import (
    error_recovery,
    score_trajectory,
    step_efficiency,
    task_completion,
    tool_call_correctness,
)
from trajeval.tasks import TASKS_BY_ID
from trajeval.tools import ToolBox


def test_tool_call_correctness_perfect_match():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    assert tool_call_correctness(traj, task) == 1.0


def test_tool_call_correctness_zero_when_no_tools_called_but_expected():
    task = TASKS_BY_ID["kb_shipping"]
    traj = GivesUpAgent().run(TASKS_BY_ID["greeting_no_tool"], ToolBox())
    # greeting task expects zero tools; scoring it against kb_shipping's
    # expectations (which wants a tool) must read as a correctness failure
    assert tool_call_correctness(traj, task) == 0.0


def test_task_completion_false_when_stopped_early():
    task = TASKS_BY_ID["kb_shipping"]
    traj = LoopingAgent().run(task, ToolBox(), max_steps=4)
    assert task_completion(traj, task) is False


def test_error_recovery_none_when_no_error_occurred():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    assert error_recovery(traj) is None


def test_error_recovery_true_after_successful_retry():
    task = TASKS_BY_ID["refund_with_retry"]
    traj = GoodAgent().run(task, ToolBox())
    assert error_recovery(traj) is True


def test_error_recovery_false_when_agent_gives_up():
    task = TASKS_BY_ID["refund_with_retry"]
    traj = GivesUpAgent().run(task, ToolBox())
    assert error_recovery(traj) is False


def test_step_efficiency_full_score_at_minimum_steps():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    assert step_efficiency(traj, task) == 1.0


def test_step_efficiency_zero_tool_task():
    task = TASKS_BY_ID["greeting_no_tool"]
    traj = GoodAgent().run(task, ToolBox())
    assert step_efficiency(traj, task) == 1.0


def test_score_trajectory_bundles_all_four_metrics():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    score = score_trajectory(traj, task)
    assert score.task_completion is True
    assert score.tool_call_correctness == 1.0
    assert score.error_recovery is None
    assert score.step_efficiency == 1.0
    assert score.stopped_early is False
