from trajeval.cli import main


def test_cli_run_good_agent(capsys):
    exit_code = main(["run", "--agent", "good"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Task completion rate: 1.00" in out


def test_cli_run_json_output(capsys):
    import json

    exit_code = main(["run", "--agent", "good", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["agent_name"] == "good"


def test_cli_gate_good_passes():
    assert main(["gate", "--config", "good"]) == 0


def test_cli_gate_bad_fails():
    assert main(["gate", "--config", "bad"]) == 1


def test_cli_compare_prints_all_agents(capsys):
    exit_code = main(["compare"])
    assert exit_code == 0
    out = capsys.readouterr().out
    for name in ("good", "wrong_tool", "looping", "gives_up", "hallucinating"):
        assert name in out


def test_cli_run_writes_history(tmp_path):
    history_path = tmp_path / "hist.jsonl"
    main(["run", "--agent", "good", "--history", str(history_path)])
    assert history_path.exists()
    assert len(history_path.read_text(encoding="utf-8").splitlines()) == 1


def test_cli_unknown_agent_exits_with_error():
    import pytest

    with pytest.raises(SystemExit):
        main(["run", "--agent", "not_a_real_agent"])
