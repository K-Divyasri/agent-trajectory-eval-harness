import json

from trajeval.agents import GoodAgent, WrongToolAgent
from trajeval.harness import gate, load_history, run_harness, save_history
from trajeval.tasks import TASKS


def test_run_harness_good_agent_passes_gate():
    scorecard = run_harness(GoodAgent(), tasks=TASKS)
    assert scorecard.task_completion_rate == 1.0
    assert gate(scorecard) is True


def test_run_harness_wrong_tool_agent_fails_gate():
    scorecard = run_harness(WrongToolAgent(), tasks=TASKS)
    assert gate(scorecard) is False


def test_scorecard_error_recovery_rate_only_over_applicable_tasks():
    scorecard = run_harness(GoodAgent(), tasks=TASKS)
    # exactly 2 of the 9 tasks ever hit a tool error
    applicable = [r for r in scorecard.results if r.score.error_recovery is not None]
    assert len(applicable) == 2
    assert scorecard.error_recovery_rate == 0.5


def test_scorecard_judge_agreement_rate_is_high_for_good_agent():
    scorecard = run_harness(GoodAgent(), tasks=TASKS)
    assert scorecard.judge_agreement_rate == 1.0


def test_save_and_load_history_roundtrip(tmp_path):
    history_path = tmp_path / "history.jsonl"
    scorecard = run_harness(GoodAgent(), tasks=TASKS)
    save_history(scorecard, str(history_path))
    save_history(scorecard, str(history_path))

    rows = load_history(str(history_path))
    assert len(rows) == 2
    assert rows[0]["agent_name"] == "good"
    # every row is valid, independently-parseable JSON
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            json.loads(line)


def test_load_history_empty_when_file_missing(tmp_path):
    assert load_history(str(tmp_path / "nope.jsonl")) == []
