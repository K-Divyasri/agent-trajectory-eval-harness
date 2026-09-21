"""Streamlit demo: pick an agent, watch its trajectories, see the harness score them.

Defaults to the offline judge so a freshly cloned/hosted copy works with no
API key. Run with: streamlit run web_app.py
"""

import streamlit as st

from trajeval.agents import OFFLINE_AGENTS
from trajeval.harness import run_harness
from trajeval.judge import has_anthropic_key, has_litellm_key
from trajeval.tasks import TASKS
from trajeval.tools import ToolBox
from trajeval.trajectory import render_transcript

st.set_page_config(page_title="Agent Trajectory Eval Harness", layout="wide")
st.title("Multi-Turn Agent Trajectory Evaluation Harness")
st.caption(
    "Score whole agent transcripts — tool-call correctness, task completion, "
    "error recovery, and step efficiency — not just the final answer."
)

with st.sidebar:
    st.header("Settings")
    agent_name = st.selectbox("Agent under test", list(OFFLINE_AGENTS), index=0)

    judge_options = ["offline"]
    if has_litellm_key():
        judge_options.append("litellm")
    if has_anthropic_key():
        judge_options.append("anthropic")
    judge_backend = st.selectbox("Judge backend", judge_options, index=0)
    if len(judge_options) == 1:
        st.caption("Set GEMINI_API_KEY/OPENAI_API_KEY or ANTHROPIC_API_KEY to unlock a real judge.")

agent = OFFLINE_AGENTS[agent_name]()
scorecard = run_harness(agent, tasks=TASKS, judge_backend=judge_backend)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Task completion", f"{scorecard.task_completion_rate:.0%}")
col2.metric("Tool-call correctness", f"{scorecard.avg_tool_call_correctness:.2f}")
col3.metric("Step efficiency", f"{scorecard.avg_step_efficiency:.2f}")
recovery_display = "n/a" if scorecard.error_recovery_rate is None else f"{scorecard.error_recovery_rate:.0%}"
col4.metric("Error recovery", recovery_display)

st.divider()
st.subheader("Per-task transcripts")

for result in scorecard.results:
    completed_icon = "✅" if result.score.task_completion else "❌"
    with st.expander(f"{completed_icon} {result.task_id}"):
        st.code(render_transcript(result.trajectory), language=None)
        st.write(
            {
                "task_completion": result.score.task_completion,
                "tool_call_correctness": round(result.score.tool_call_correctness, 2),
                "step_efficiency": round(result.score.step_efficiency, 2),
                "error_recovery": result.score.error_recovery,
                "stopped_early": result.score.stopped_early,
            }
        )
        if result.verdict is not None:
            st.markdown(f"**Judge verdict** ({judge_backend}): {result.verdict.efficiency_rating}")
            st.write(result.verdict.model_dump())

st.divider()
st.caption(
    "Run `python -m trajeval compare` from the CLI for a leaderboard across "
    "all five offline agents, or `python -m trajeval gate --config good` to "
    "see the CI ship-gate in action."
)
