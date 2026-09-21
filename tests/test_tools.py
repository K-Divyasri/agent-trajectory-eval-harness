from trajeval.tools import ToolBox


def test_search_kb_hits_and_misses():
    tools = ToolBox()
    assert tools.search_kb("shipping").ok
    assert "business day" in tools.search_kb("shipping").output
    assert not tools.search_kb("gibberish topic xyz").ok


def test_get_order_status_known_and_unknown():
    tools = ToolBox()
    result = tools.get_order_status("ORD-100")
    assert result.ok
    assert "delivered" in result.output

    missing = tools.get_order_status("ORD-999")
    assert not missing.ok


def test_issue_refund_ord_500_fails_once_then_succeeds():
    tools = ToolBox()
    first = tools.issue_refund("ORD-500", 75.0)
    assert not first.ok
    second = tools.issue_refund("ORD-500", 75.0)
    assert second.ok


def test_issue_refund_rejects_double_refund():
    tools = ToolBox()
    tools.issue_refund("ORD-100", 42.50)
    second = tools.issue_refund("ORD-100", 42.50)
    assert not second.ok
    assert "already refunded" in second.output


def test_calculate_safe_eval():
    tools = ToolBox()
    result = tools.calculate("3 * 19.99")
    assert result.ok
    assert "59.97" in result.output


def test_calculate_rejects_unsafe_expression():
    tools = ToolBox()
    result = tools.calculate("__import__('os').system('echo hi')")
    assert not result.ok
