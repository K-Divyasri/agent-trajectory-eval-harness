from trajeval.agents import (
    GivesUpAgent,
    GoodAgent,
    HallucinatingAgent,
    LoopingAgent,
    OFFLINE_AGENTS,
    WrongToolAgent,
)
from trajeval.tasks import TASKS_BY_ID
from trajeval.tools import ToolBox


def test_offline_agents_registry_has_five_entries():
    assert set(OFFLINE_AGENTS) == {"good", "wrong_tool", "looping", "gives_up", "hallucinating"}


def test_good_agent_completes_simple_lookup():
    task = TASKS_BY_ID["order_status_simple"]
    traj = GoodAgent().run(task, ToolBox())
    assert traj.final_answer
    assert task.check(traj.final_answer)
    assert traj.tool_call_names() == ["get_order_status"]


def test_good_agent_recovers_from_transient_error():
    task = TASKS_BY_ID["refund_with_retry"]
    traj = GoodAgent().run(task, ToolBox())
    assert task.check(traj.final_answer)
    # two attempts at the same tool: the transient failure, then the retry
    assert traj.tool_call_names() == ["issue_refund", "issue_refund"]
    assert traj.had_tool_error()


def test_good_agent_does_not_retry_forever_on_permanent_error():
    task = TASKS_BY_ID["order_status_unknown"]
    traj = GoodAgent().run(task, ToolBox())
    assert not traj.stopped_early
    assert len(traj.tool_call_names()) == 2  # one try, one retry, then it gives up


def test_good_agent_chains_two_tools():
    task = TASKS_BY_ID["refund_after_status"]
    traj = GoodAgent().run(task, ToolBox())
    assert traj.tool_call_names() == ["get_order_status", "issue_refund"]
    assert task.check(traj.final_answer)


def test_good_agent_no_tool_needed():
    task = TASKS_BY_ID["greeting_no_tool"]
    traj = GoodAgent().run(task, ToolBox())
    assert traj.tool_call_names() == []
    assert task.check(traj.final_answer)


def test_wrong_tool_agent_never_calls_the_expected_tool():
    task = TASKS_BY_ID["order_status_simple"]
    traj = WrongToolAgent().run(task, ToolBox())
    assert "get_order_status" not in traj.tool_call_names()


def test_looping_agent_always_stops_early():
    task = TASKS_BY_ID["kb_shipping"]
    traj = LoopingAgent().run(task, ToolBox(), max_steps=6)
    assert traj.stopped_early
    assert traj.final_answer == ""
    assert len(traj.tool_call_names()) == 3  # 6 steps / 2 per iteration


def test_gives_up_agent_fails_on_first_error():
    task = TASKS_BY_ID["refund_with_retry"]
    traj = GivesUpAgent().run(task, ToolBox())
    assert traj.tool_call_names() == ["issue_refund"]  # never retries
    assert not task.check(traj.final_answer)


def test_hallucinating_agent_calls_a_tool_that_does_not_exist():
    task = TASKS_BY_ID["order_status_simple"]
    traj = HallucinatingAgent().run(task, ToolBox())
    assert traj.tool_call_names() == ["send_email_receipt"]
    assert traj.had_tool_error()
