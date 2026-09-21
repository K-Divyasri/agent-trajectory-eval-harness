"""The labelled task suite: each Task has ground truth to score a trajectory against.

`expected_tools` is the ideal, minimal ordered tool-call sequence a competent
agent would use. `check` grades the final answer text. `involves_error` flags
tasks that exercise the ORD-500 transient-failure fixture in tools.py, so the
harness can compute an error-recovery rate over exactly the tasks where
recovery was actually possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    id: str
    request: str
    expected_tools: list
    check: Callable[[str], bool]
    involves_error: bool = False
    min_steps: int = 1


def _contains_all(*words):
    def check(answer: str) -> bool:
        a = answer.lower()
        return all(w.lower() in a for w in words)

    return check


def _contains_any(*words):
    def check(answer: str) -> bool:
        a = answer.lower()
        return any(w.lower() in a for w in words)

    return check


TASKS = [
    Task(
        id="order_status_simple",
        request="What's the status of order ORD-100?",
        expected_tools=["get_order_status"],
        check=_contains_all("delivered"),
        min_steps=1,
    ),
    Task(
        id="order_status_unknown",
        request="What's the status of order ORD-999?",
        expected_tools=["get_order_status"],
        check=_contains_any("not found", "no order", "doesn't exist", "don't have"),
        min_steps=1,
    ),
    Task(
        id="refund_after_status",
        request="Order ORD-200 arrived damaged, please refund the full amount.",
        expected_tools=["get_order_status", "issue_refund"],
        check=_contains_all("refund"),
        min_steps=2,
    ),
    Task(
        id="refund_with_retry",
        request="Please refund order ORD-500, it was defective.",
        expected_tools=["issue_refund"],
        check=_contains_all("refund"),
        involves_error=True,
        min_steps=1,
    ),
    Task(
        id="kb_shipping",
        request="How long does shipping usually take?",
        expected_tools=["search_kb"],
        check=_contains_any("business day", "days"),
        min_steps=1,
    ),
    Task(
        id="kb_returns",
        request="What is your return policy?",
        expected_tools=["search_kb"],
        check=_contains_any("30 days", "return"),
        min_steps=1,
    ),
    Task(
        id="calc_total",
        request="If I order 3 items at $19.99 each, what's the total?",
        expected_tools=["calculate"],
        check=_contains_any("59.97"),
        min_steps=1,
    ),
    Task(
        id="kb_then_calc",
        request="What's the refund processing time, and how many days is that at most if it starts today plus 5?",
        expected_tools=["search_kb", "calculate"],
        check=_contains_any("3-5", "business day", "5 day", "10"),
        min_steps=2,
    ),
    Task(
        id="greeting_no_tool",
        request="Hi, thanks for your help earlier!",
        expected_tools=[],
        check=_contains_any("welcome", "help", "glad", "hi", "hello"),
        min_steps=0,
    ),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
