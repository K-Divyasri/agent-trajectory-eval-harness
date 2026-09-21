"""Command-line entry point: python -m trajeval <command> ..."""

from __future__ import annotations

import argparse
import json
import sys

from .agents import OFFLINE_AGENTS
from .harness import gate, run_harness, save_history
from .report import format_scorecard, leaderboard_table
from .tasks import TASKS

GATE_CONFIGS = {"good": "good", "bad": "wrong_tool"}


def _build_agent(name: str, model: str = None):
    if name == "real":
        from .real_agent import LiteLLMAgent

        return LiteLLMAgent(model=model)
    if name in OFFLINE_AGENTS:
        return OFFLINE_AGENTS[name]()
    raise SystemExit(f"unknown agent {name!r}; choose from {list(OFFLINE_AGENTS) + ['real']}")


def cmd_run(args):
    agent = _build_agent(args.agent, model=args.model)
    scorecard = run_harness(agent, tasks=TASKS, judge_backend=args.judge, judge_model=args.model)
    if args.json:
        print(json.dumps(scorecard.to_dict(), indent=2))
    else:
        print(format_scorecard(scorecard))
    if args.history:
        save_history(scorecard, args.history)
    return 0


def cmd_compare(args):
    scorecards = []
    for name in OFFLINE_AGENTS:
        agent = OFFLINE_AGENTS[name]()
        scorecards.append(run_harness(agent, tasks=TASKS, judge_backend=args.judge))
    print(leaderboard_table(scorecards))
    return 0


def cmd_gate(args):
    agent_name = GATE_CONFIGS.get(args.config, args.config)
    agent = _build_agent(agent_name)
    scorecard = run_harness(agent, tasks=TASKS, judge_backend="offline")
    print(format_scorecard(scorecard))
    passed = gate(scorecard)
    print(f"\nGATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="trajeval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one agent over the task suite")
    p_run.add_argument("--agent", default="good", choices=list(OFFLINE_AGENTS) + ["real"])
    p_run.add_argument("--judge", default="offline", choices=["offline", "litellm", "anthropic"])
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--history", default=None)
    p_run.set_defaults(func=cmd_run)

    p_compare = sub.add_parser("compare", help="run every offline agent and print a leaderboard")
    p_compare.add_argument("--judge", default="offline", choices=["offline", "litellm", "anthropic"])
    p_compare.set_defaults(func=cmd_compare)

    p_gate = sub.add_parser("gate", help="run the CI ship gate; exit code is the verdict")
    p_gate.add_argument("--config", default="good", choices=["good", "bad", *OFFLINE_AGENTS])
    p_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
