"""A tiny simulated helpdesk toolbox the agents under test can call.

Nothing here talks to a network. `issue_refund` is rigged to fail exactly
once for order ORD-500 so tasks can test whether an agent recovers from a
transient tool error instead of giving up.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    ok: bool
    output: str


# name -> {"status": str, "total": float, "refunded": bool}
_ORDERS = {
    "ORD-100": {"status": "delivered", "total": 42.50, "refunded": False},
    "ORD-200": {"status": "shipped", "total": 19.99, "refunded": False},
    "ORD-300": {"status": "processing", "total": 120.00, "refunded": False},
    "ORD-500": {"status": "delivered", "total": 75.00, "refunded": False},
}

_KB = {
    "shipping": "Standard shipping takes 5-7 business days. Express shipping takes 2 business days.",
    "returns": "Items can be returned within 30 days of delivery for a full refund.",
    "warranty": "All electronics carry a 1-year manufacturer warranty covering defects.",
    "refund_time": "Refunds are processed within 3-5 business days after approval.",
    "payment": "We accept credit card, debit card, and PayPal. No cash on delivery.",
    "cancel": "Orders can be cancelled for free before they enter 'processing' status.",
}

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str) -> float:
    """Evaluate a numeric expression using ast, never eval()."""
    node = ast.parse(expr, mode="eval").body
    return _eval_node(node)


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


@dataclass
class ToolBox:
    """Stateful simulated backend. One instance per trajectory run."""

    orders: dict = field(default_factory=lambda: {k: dict(v) for k, v in _ORDERS.items()})
    _refund_attempts: dict = field(default_factory=dict)

    def search_kb(self, query: str) -> ToolResult:
        query_l = query.lower()
        for key, text in _KB.items():
            if key in query_l or any(word in query_l for word in key.split("_")):
                return ToolResult(True, text)
        return ToolResult(False, f"no knowledge base article matches '{query}'")

    def get_order_status(self, order_id: str) -> ToolResult:
        order = self.orders.get(order_id)
        if order is None:
            return ToolResult(False, f"order {order_id} not found")
        return ToolResult(True, f"order {order_id} is {order['status']}, total ${order['total']:.2f}")

    def issue_refund(self, order_id: str, amount: float) -> ToolResult:
        order = self.orders.get(order_id)
        if order is None:
            return ToolResult(False, f"order {order_id} not found")
        if order["refunded"]:
            return ToolResult(False, f"order {order_id} was already refunded")

        attempts = self._refund_attempts.get(order_id, 0)
        self._refund_attempts[order_id] = attempts + 1

        # ORD-500 simulates a transient payment-gateway failure on the FIRST
        # attempt only — this is the fixture for testing error recovery.
        if order_id == "ORD-500" and attempts == 0:
            return ToolResult(False, "payment gateway timeout, please retry")

        order["refunded"] = True
        return ToolResult(True, f"refunded ${amount:.2f} to order {order_id}")

    def calculate(self, expression: str) -> ToolResult:
        try:
            value = _safe_eval(expression)
        except Exception as exc:  # noqa: BLE001 - surface any parse/eval failure as a tool error
            return ToolResult(False, f"could not evaluate '{expression}': {exc}")
        return ToolResult(True, f"{expression} = {value}")


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the help center knowledge base for a policy topic (shipping, returns, warranty, refund_time, payment, cancel).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status and total for an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund for an order. Requires the order id and the dollar amount to refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a numeric arithmetic expression, e.g. '42.50 * 2'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
