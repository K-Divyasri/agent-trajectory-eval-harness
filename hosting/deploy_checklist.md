# Deploy checklist — Agent Trajectory Evaluation Harness

This is the project's "definition of done." Walk it top to bottom. Don't tick a box you
haven't actually verified by running the command — "should work" isn't the same as "works."
Commands assume you're at the repo root unless noted.

## Runs locally

- [ ] Fresh virtual environment, dependencies install cleanly:
      `python -m venv .venv ; .\.venv\Scripts\Activate.ps1` then
      `pip install -r requirements.txt`
- [ ] A single agent run prints a scorecard:
      `python -m trajeval run --agent good` (9 tasks, offline judge, no key needed)
- [ ] The leaderboard looks right:
      `python -m trajeval compare` (expect `good` at 1.00 completion / 1.00 tool-F1 / 0.89
      efficiency; `wrong_tool`, `looping`, `gives_up`, `hallucinating` all visibly worse)
- [ ] The Streamlit app runs and renders (from the **repo root**):
      `streamlit run web_app.py` - sidebar shows the 5 offline agents and a
      judge selector locked to `offline` (no key set); main panel shows 4 metric tiles and 9
      expandable transcripts.

## The ship gate works both ways

- [ ] `python -m trajeval gate --config good` prints `GATE: PASS` and exits 0.
      Verify the code: `python -m trajeval gate --config good ; echo "exit = $LASTEXITCODE"`
      → `exit = 0`.
- [ ] `python -m trajeval gate --config bad` (maps to `WrongToolAgent`) prints `GATE: FAIL` and
      exits 1. Verify: `python -m trajeval gate --config bad ; echo "exit = $LASTEXITCODE"`
      → `exit = 1`. This non-zero exit is what fails a build in CI.

## Tests pass

- [ ] `pytest` run from the repo root is all green - 44 tests, all offline.
- [ ] You ran it in the fresh venv, not just your everyday one, so you know
      `requirements.txt` is complete on its own.

## Secrets are clean

- [ ] No API key is hardcoded anywhere in the source — `trajeval/judge.py` and
      `trajeval/real_agent.py` only ever read keys via `os.environ.get(...)`.
- [ ] If you added a local `.env` for convenience, a project-root `.gitignore` lists it (plus
      `.venv/`, `__pycache__/`, `.pytest_cache/`), and `git status` shows it's not tracked.
- [ ] `git ls-files | Select-String ".env"` shows nothing (this project ships no
      `.env.example` — there's no dotenv step baked into the package, so there's simply no
      secrets file to leak in the first place).
- [ ] No real judge/agent env var (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) is
      set anywhere the app's default behavior depends on it — confirm `git clone` to a clean
      folder and running with no env vars set still gives you the offline judge only.

## README is recruiter-ready

- [ ] Root `README.md` covers: what you built, why it's the sequel to project 10 (whole
      transcript vs. one answer), the four metrics, the folder map, the path to follow, and
      copy-pasteable run commands.
- [ ] The verified leaderboard numbers table is in the README, so a reader sees proof this runs
      without having to run it themselves first.
- [ ] The CI status badge is at the top, once Actions has run once.
- [ ] The **live dashboard URL** (Streamlit Cloud or the Space) is added near the top, so a
      recruiter can click straight through to a working demo.

## Pushed to GitHub with CI

- [ ] Repo created empty on github.com (no auto README/license), named
      `agent-trajectory-eval-harness`, public.
- [ ] `git init` → `git add .` → `git commit` → `git branch -M main` →
      `git remote add origin ...` → `git push -u origin main`, all from the **repo root**
      (the folder containing `hosting/`).
- [ ] `.github/workflows/ci.yml` (copied from `hosting/github_actions/ci.yml`) is committed and
      pushed.
- [ ] The Actions tab shows a completed run with a green checkmark. It's keyless — every step
      runs offline. The run includes both the `pytest` step **and** the ship-gate step
      (`trajeval gate --config good`), so agent quality is checked on every push, not just
      syntax.
- [ ] If it went red, you read the log bottom-up and fixed the cause (a missing dep, or the
      gate catching a real regression in agent behavior), then re-ran to green.

## Live dashboard is up

- [ ] Deployed via Path A (Streamlit Community Cloud, main file path
      `web_app.py`) **or** Path B (Hugging Face Spaces, Docker SDK +
      Streamlit template, `app.py` + `trajeval/` + `requirements.txt` at the Space root).
- [ ] Open the live URL — sidebar lets you switch between the 5 offline agents, main panel
      updates the 4 metric tiles and the 9 transcripts. Judge selector shows only `offline`
      (no key is set on the host) — that's correct, not a bug.
- [ ] (Optional) If you want a real judge backend online, `GEMINI_API_KEY` /
      `OPENAI_API_KEY` (litellm) or `ANTHROPIC_API_KEY` (native Claude structured output) is set
      in the host's secrets vault (NOT in the repo), and the judge dropdown grows a second
      option after the app restarts.

When every box is ticked, the project is done and presentable. Send the repo link and the live
dashboard link together — a working demo, plus a CI that fails the build when an agent's
trajectory quality regresses, is what gets you the follow-up conversation.
