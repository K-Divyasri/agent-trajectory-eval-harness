# trajeval

A multi-turn agent trajectory evaluation harness. It scores a whole agent
*transcript* (reasoning, tool calls, tool results, final answer), not just the
final answer. Single-answer evaluation, such as scoring one generated RAG answer
against one reference, cannot see how an agent got there. trajeval scores the
multi-step *process*: tool-call correctness, task completion, error recovery, and
step efficiency, plus an LLM-as-judge that reads the full transcript holistically.

Everything runs offline by default: five deterministic simulated agents, a
rule-based judge, no API key. A real tool-calling agent and real judge backends
are optional.

## Install

```bash
pip install -r requirements.txt
# or, for just the core (no streamlit/matplotlib/litellm/anthropic):
pip install -e .
```

## Try it

```bash
python -m trajeval run --agent good
python -m trajeval compare
python -m trajeval gate --config good     # exit 0
python -m trajeval gate --config bad      # exit 1
streamlit run web_app.py
```

- `run` scores one agent over the task suite and prints a markdown scorecard
  (`--json` for machine-readable output, `--history FILE` to append the result to
  a JSONL log).
- `compare` runs every offline agent and prints a leaderboard.
- `gate` is a CI ship gate: the exit code is the verdict (0 pass, 1 fail). It
  passes only if task completion, tool-call correctness and step efficiency all
  meet minimum thresholds.
- `--agent real` runs a real tool-calling loop through LiteLLM. `--judge litellm`
  or `--judge anthropic` swaps the offline judge for a real one. These need API
  keys and the `real` extra: `pip install -e ".[real]"`.

## How it works

The harness runs an agent over a labelled task suite against a simulated
helpdesk toolbox (`search_kb`, `get_order_status`, `issue_refund`, `calculate`).
Each run is recorded as a trajectory, then scored by four metrics and one judge.

The five offline agents are `good` (the reference) plus four that fail in
different ways so the metrics have something to catch: `wrong_tool`, `looping`,
`gives_up`, and `hallucinating`.

## Package layout

- `trajeval/tools.py`: the simulated helpdesk toolbox (search_kb, get_order_status, issue_refund, calculate)
- `trajeval/tasks.py`: the labelled task suite (ground truth: expected tools, min steps, answer check)
- `trajeval/trajectory.py`: `Step` / `Trajectory`, the transcript record everything else reads
- `trajeval/agents.py`: 5 offline deterministic agents: `good`, `wrong_tool`, `looping`, `gives_up`, `hallucinating`
- `trajeval/real_agent.py`: `LiteLLMAgent`, the real tool-calling loop (`--agent real`)
- `trajeval/metrics.py`: tool_call_correctness, task_completion, error_recovery, step_efficiency
- `trajeval/judge.py`: `JudgeVerdict` (pydantic) + `offline_judge` / `real_judge` (LiteLLM) / `anthropic_judge` (raw Anthropic SDK structured output)
- `trajeval/harness.py`: `run_harness`, `Scorecard`, JSONL history, `gate()` for CI
- `trajeval/report.py`: markdown scorecard + leaderboard + matplotlib chart
- `trajeval/cli.py`: `python -m trajeval run|compare|gate`
- `web_app.py`: Streamlit transcript viewer
- `tests/`: offline pytest suite
- `hosting/`: deployment guide, checklist and a CI workflow template

## Tests

```bash
python -m pytest -q
```

The 44 tests run offline and need no API key.
