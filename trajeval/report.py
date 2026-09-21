"""Rendering: a markdown scorecard table and a matplotlib comparison chart."""

from __future__ import annotations

from .harness import Scorecard


def format_scorecard(scorecard: Scorecard) -> str:
    lines = [
        f"# Trajectory Scorecard: {scorecard.agent_name}",
        "",
        f"- Task completion rate: {scorecard.task_completion_rate:.2f}",
        f"- Avg tool-call correctness (F1): {scorecard.avg_tool_call_correctness:.2f}",
        f"- Avg step efficiency: {scorecard.avg_step_efficiency:.2f}",
    ]
    if scorecard.error_recovery_rate is not None:
        lines.append(f"- Error recovery rate: {scorecard.error_recovery_rate:.2f}")
    else:
        lines.append("- Error recovery rate: n/a (no tasks triggered an error)")
    if scorecard.judge_agreement_rate is not None:
        lines.append(f"- Judge/metric agreement: {scorecard.judge_agreement_rate:.2f}")
    lines.append("")
    lines.append("| Task | Completed | Tool F1 | Efficiency | Recovery | Judge verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in scorecard.results:
        recovery = "n/a" if r.score.error_recovery is None else str(r.score.error_recovery)
        verdict = r.verdict.efficiency_rating if r.verdict else "n/a"
        lines.append(
            f"| {r.task_id} | {r.score.task_completion} | {r.score.tool_call_correctness:.2f} | "
            f"{r.score.step_efficiency:.2f} | {recovery} | {verdict} |"
        )
    return "\n".join(lines)


def leaderboard_table(scorecards: list) -> str:
    lines = ["| Agent | Completion | Tool F1 | Efficiency | Recovery |", "|---|---|---|---|---|"]
    for sc in scorecards:
        recovery = "n/a" if sc.error_recovery_rate is None else f"{sc.error_recovery_rate:.2f}"
        lines.append(
            f"| {sc.agent_name} | {sc.task_completion_rate:.2f} | "
            f"{sc.avg_tool_call_correctness:.2f} | {sc.avg_step_efficiency:.2f} | {recovery} |"
        )
    return "\n".join(lines)


def plot_leaderboard(scorecards: list, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [sc.agent_name for sc in scorecards]
    completion = [sc.task_completion_rate for sc in scorecards]
    tool_f1 = [sc.avg_tool_call_correctness for sc in scorecards]
    efficiency = [sc.avg_step_efficiency for sc in scorecards]

    x = range(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width for i in x], completion, width, label="Task completion")
    ax.bar(list(x), tool_f1, width, label="Tool-call F1")
    ax.bar([i + width for i in x], efficiency, width, label="Step efficiency")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Agent trajectory quality by metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
