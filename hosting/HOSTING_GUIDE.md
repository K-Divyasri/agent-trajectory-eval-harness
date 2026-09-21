# Publishing the Agent Trajectory Evaluation Harness

This project has the same two things worth showing off as its predecessor, project 10 (the
RAG Evaluation Harness):

1. **The repo on GitHub with CI that runs the ship gate on every push.** Most people's CI runs
   unit tests. Yours runs the tests *and* `python -m trajeval gate --config good` — so a commit
   that breaks the agent's tool routing, sends it into a retry loop, or drops its task-completion
   rate below the bar turns the build red. That's the whole point of this project existing: you
   can *prove* an agent's trajectory quality the same way you'd prove test coverage.
2. **The Streamlit dashboard on a free live URL.** `build_from_scratch/web_app.py` — pick an
   agent, watch its transcripts, see the four metrics and the judge verdict per task. A live
   link a recruiter can click is what makes the project land.

Same as project 10, this is cheap and easy to host for one reason: everything runs **fully
offline**. The five offline agents and the offline judge are pure Python — no API key, no
network call, no cost. The hosted app works for anyone the moment it builds. A real judge
backend (LiteLLM or native Anthropic) is an optional toggle you turn on later with a key placed
in the host's secrets vault — **never** in the repo.

---

## The layout you're working with

```
38-agent-trajectory-eval-harness/     <- this whole folder becomes your GitHub repo
├── build_from_scratch/               <- the real package lives here
│   ├── trajeval/                     <- tools, agents, metrics, judge, harness, cli
│   ├── web_app.py                    <- the Streamlit app -- this is what you deploy
│   ├── requirements.txt              <- pydantic, matplotlib, pytest, litellm, streamlit, anthropic
│   ├── tests/                        <- 44 offline tests
│   └── pyproject.toml
├── hosting/                          <- you are here
│   ├── HOSTING_GUIDE.md
│   ├── deploy_checklist.md
│   └── github_actions/ci.yml         <- copy this to .github/workflows/ci.yml
├── knowledge/, notebooks/, labs/     <- the teaching material, not needed by the deployed app
└── README.md
```

One thing follows from this, and it's simpler than project 10: **`web_app.py` already lives
right next to the `trajeval/` package it imports**, both inside `build_from_scratch/`. Project
10's dashboard lived in a separate `hosting/streamlit_app/` folder and had to walk up the
directory tree (or have the package copied alongside it) to find its package. This app doesn't
need any of that — point a host at `build_from_scratch/web_app.py` and Python's own import
rules do the rest, because `trajeval/` is a plain sibling directory of the script being run.

---

## Step 0 — Get the code on GitHub

If you did project 10 (or any earlier project on this roadmap), this is the same dance — skim
it, the new bits are flagged.

### Install Git and tell it who you are (once per machine)

If Git already works for you, skip this. Otherwise install from
<https://git-scm.com/download/win>, open a **new** PowerShell window, and check:

```powershell
git --version
```

Then stamp your identity onto commits (use the same email as your GitHub account):

```powershell
git config --global user.name "Your Name"
git config --global user.email "mathuransada@gmail.com"
```

### Know what must NOT go in the repo

This package reads API keys straight from environment variables (`os.environ.get(...)` in
`trajeval/judge.py` and `trajeval/real_agent.py`) — there's no `.env` file or `python-dotenv`
step baked into `build_from_scratch/`. That's actually the simplest possible secrets story:
**there is no secrets file to accidentally commit**, because there is no secrets file at all.
Locally you export a key into your shell session for the session's lifetime; on a host, you set
it in that host's secrets vault, which materializes it as an environment variable at runtime.
Either way, nothing sensitive ever needs to touch a tracked file.

If you *do* add a local `.env` for convenience later (some people like `python-dotenv` for this),
add a project-root `.gitignore` first with at least:

```
.env
.venv/
__pycache__/
.pytest_cache/
```

Rule of thumb, same as every project on this roadmap: **source code, config, and docs go in.
Secrets and machine junk stay out.**

### Make the repo and push

Run these from the **project root** — the `38-agent-trajectory-eval-harness/` folder, the one
with `build_from_scratch/` and this `hosting/` folder inside it:

```powershell
cd ai\38-agent-trajectory-eval-harness
git init
git add .
git commit -m "Initial commit: Agent Trajectory Evaluation Harness"
```

Check the state:

```powershell
git status
```

You want `nothing to commit, working tree clean`. If you added a `.gitignore` for a local
`.env`, double-check it's honored:

```powershell
git ls-files | Select-String ".env"
```

This should print nothing (there's no `.env.example` shipped in this project, unlike project 10
— see above for why). If a real `.env` shows up here, you committed a secret — jump to
*Committed a secret by accident* at the bottom.

Then make an **empty** repo on github.com (the **+** menu, top-right → **New repository**), name
it `agent-trajectory-eval-harness`, leave it **Public**, and do **not** tick "Add a README /
.gitignore / license" (an empty repo avoids a first-push collision). Copy the URL it shows you,
then:

```powershell
git branch -M main
git remote add origin https://github.com/YOURNAME/agent-trajectory-eval-harness.git
git push -u origin main
```

If the first push asks for a password typed into the terminal, that won't work — GitHub turned
off password auth years ago. Use the browser sign-in it offers, or install the GitHub CLI
(<https://cli.github.com>) and run `gh auth login` once.

---

## Step 1 — Add CI that runs the tests AND the ship gate

This is the part that makes the project stand out. CI proves your 44 tests pass on a clean
machine every push — and, uniquely here, it also runs the trajectory-quality gate, so an
agent-routing regression fails the build, not just a syntax error. This `hosting/` folder ships
a ready workflow at `github_actions/ci.yml`. GitHub only runs workflows that live under
`.github/workflows/`, so copy it there. From the **project root**:

```powershell
mkdir .github\workflows
copy hosting\github_actions\ci.yml .github\workflows\ci.yml
git add .github\workflows\ci.yml
git commit -m "Add CI: run the offline tests and the ship gate on every push"
git push
```

Open the repo's **Actions** tab to watch it run. The steps are: checkout → install Python →
install `build_from_scratch/requirements.txt` → `pytest` (44 tests, run from inside
`build_from_scratch/`) → **`python -m trajeval gate --config good`**. It's **entirely keyless**
— every step runs offline, so nothing needs a secret. Green means all tests passed *and* the
gate passed on GitHub's machine.

The gate step is the differentiator. `gate --config good` builds `GoodAgent`, runs it over all
9 tasks with the offline judge, and exits 0 only when `task_completion_rate`,
`avg_tool_call_correctness`, and `avg_step_efficiency` all clear their thresholds — non-zero
otherwise. To *see* it work, point the gate at a broken agent on a branch:

```powershell
python -m trajeval gate --config bad     # maps to WrongToolAgent -> exit 1
```

push that change, and watch the Actions run go red on the gate step even though the unit tests
still pass — a quality regression caught before merge, not after a demo goes wrong. Read the
failed step's log; `format_scorecard` prints the per-task breakdown right above the `GATE: FAIL`
line, so you can see exactly which metric tripped the wire.

Once it's green, grab the status badge (Actions page → the workflow → the `...` menu → **Create
status badge**) and paste it at the top of your root `README.md`.

---

## Step 2 — Deploy the dashboard

`build_from_scratch/web_app.py` is the whole deployed app — no separate copy, no second file to
maintain. Streamlit Community Cloud is the simplest path because it can point directly at a file
deep in your repo with zero copying; Hugging Face Spaces works too but needs one small
adjustment because of a recent platform change (covered below).

### Path A (recommended) — Streamlit Community Cloud

Streamlit's own free host. It connects straight to the GitHub repo you just pushed and redeploys
automatically on every push. Docs: <https://docs.streamlit.io/deploy/streamlit-community-cloud>.

1. Go to <https://share.streamlit.io> and sign in with GitHub, granting it read access to your
   repos.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. **Repository:** `YOURNAME/agent-trajectory-eval-harness`. **Branch:** `main`.
4. **Main file path:** `build_from_scratch/web_app.py`.
5. Click **Deploy**.

That's the entire configuration. Two things make this work with no extra fiddling:

- **Dependencies.** Community Cloud looks for a `requirements.txt` first in the same directory
  as your main file, then at the repo root. `build_from_scratch/requirements.txt` is sitting
  right there next to `web_app.py`, so it's picked up automatically — you don't need to point
  at it explicitly or duplicate it at the repo root.
- **Imports.** The app's working directory on Community Cloud is always the repo root, but
  Python still puts the *script's own directory* on `sys.path` when it runs `web_app.py` — and
  `trajeval/` is a plain sibling folder of `web_app.py` inside `build_from_scratch/`. So
  `from trajeval.agents import OFFLINE_AGENTS` (and the rest of `web_app.py`'s imports) resolve
  with no path hacking at all.

A minute or two later you have a public `*.streamlit.app` URL. Open it: the sidebar shows the
five offline agents (`good`, `wrong_tool`, `looping`, `gives_up`, `hallucinating`) and a judge
selector locked to `offline` (no key is set, so that's the only option — expected). The main
panel shows the four metric tiles and nine expandable transcripts, exactly what you saw running
it locally.

### Path B — Hugging Face Spaces

A "Space" is a free, always-on web app hosted by Hugging Face. One heads-up before you start:
as of this writing, Hugging Face has **deprecated selecting "Streamlit" directly as a Space SDK**
for new Spaces — the create-Space form no longer offers it as a first-class option. The current
path is: pick the **Docker** SDK, then choose the **Streamlit** template from Docker's template
gallery, which hands you a working `Dockerfile` that runs `streamlit run app.py` on port 8501.
(General Spaces docs: <https://huggingface.co/docs/hub/spaces-overview>.)

1. Make a free account at <https://huggingface.co>.
2. Top-right, your avatar → **New Space**.
3. **Space name:** `agent-trajectory-eval-harness`. **License:** whatever you like.
4. **Space SDK:** click **Docker** first to open the template list, then pick the **Streamlit**
   template from it. Leave hardware on the free **CPU basic** tier — this app is light.
   Click **Create Space**.

The generated `Dockerfile`'s launch step hardcodes `streamlit run app.py`, so the file you
upload as the entry point must be named `app.py` at the Space's **root**, with the `trajeval/`
package sitting right next to it (same sibling-import logic as Path A — `app.py` needs
`trajeval/` next door to import it). Using the **Files** tab → **Add file** → **Upload files**:

- Upload `build_from_scratch/web_app.py`'s contents as a file named **`app.py`** at the Space
  root (open it locally, copy the contents, paste into a new `app.py` in the Space's web
  editor — or upload the file and rename it after).
- Upload the whole `build_from_scratch/trajeval/` folder to the Space root, so it sits next to
  `app.py`.
- Upload `build_from_scratch/requirements.txt` as `requirements.txt` at the Space root.

> You don't need `tests/`, `pyproject.toml`, the notebooks, or the labs on the Space — the
> dashboard only imports the `trajeval` package.

Commit, then watch the **Logs** tab: it builds the Docker image, installs `requirements.txt`,
and launches Streamlit. A minute or two later you get a live URL like
`https://huggingface.co/spaces/YOURNAME/agent-trajectory-eval-harness`.

If a future Space creation screen brings back a plain "Streamlit" SDK option (platforms change
these flows over time), it's even simpler: pick it directly, and the same file layout above
still applies — `app.py` + `trajeval/` + `requirements.txt` at the Space root.

---

## Turning on a real judge or real agent online (optional, and it costs money)

The deployed app is offline and free by default — leave it that way and it never calls a model.
But `web_app.py`'s sidebar is written to notice a key the moment one exists:

```python
judge_options = ["offline"]
if has_litellm_key():
    judge_options.append("litellm")
if has_anthropic_key():
    judge_options.append("anthropic")
```

`has_litellm_key()` (in `trajeval/judge.py`) checks for `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or
`OPENAI_API_KEY`; `has_anthropic_key()` checks for `ANTHROPIC_API_KEY`. Set the matching
environment variable as a host secret and the corresponding option quietly appears in the judge
dropdown next time the app restarts — no code change, no redeploy needed beyond the restart the
secret change triggers. The key **never goes in the repo**. Both hosts have a proper vault:

- **Streamlit Community Cloud:** open your app, click the **⋮** menu → **Settings** → **Secrets**
  — a small TOML editor. Add:

  ```toml
  GEMINI_API_KEY = "your-key-here"
  ```

  Save; the app restarts with the key available as an environment variable.

- **Hugging Face Spaces:** Space **Settings** tab → **Variables and secrets** → **New secret**.
  Name it `GEMINI_API_KEY` (or `ANTHROPIC_API_KEY` for the native Claude judge path), paste the
  value, save. The Space restarts and the key is available to the running app — HF never shows
  the value again once saved.

A secret set either way is stored encrypted and never appears in your repo or your build logs.
That's the whole design: the key lives in the vault, the code only ever asks "is one set?"

Get a free Gemini key at <https://aistudio.google.com/apikey> for the `litellm` judge/agent
path, or an Anthropic key at <https://console.anthropic.com> for the native `anthropic_judge`
structured-output path (see `trajeval/judge.py`'s `anthropic_judge` for the exact
`client.messages.create(..., output_config={"format": {"type": "json_schema", ...}})` call it
makes). Both paths are guarded by `has_litellm_key()` / `has_anthropic_key()` everywhere in this
project — notebooks, labs, and the CLI — so nothing breaks with no key set; you just get the
offline backend instead.

---

## Common Git mistakes (troubleshooting)

**Committed a secret by accident.** First, treat the key as compromised — go to the provider and
**delete/rotate it**; if you pushed, it's already public. Then remove the file from Git while
keeping it on disk:

```powershell
git rm --cached .env
git commit -m "Remove committed secret"
git push
```

`git rm --cached` only stops tracking it going forward — the key still sits in Git *history*,
which is exactly why you rotate it rather than trusting the delete.

**`error: failed to push` / push rejected.** The remote has commits your local repo doesn't —
usually because GitHub added a README or license when you created the repo. Pull and replay:

```powershell
git pull origin main --rebase
git push
```

Next time, create the repo completely empty.

**CI is red on the gate step but the tests passed.** That's the gate doing its job — one of
`task_completion_rate` / `avg_tool_call_correctness` / `avg_step_efficiency` fell below its
threshold. Read the failed step's log; `format_scorecard`'s table is printed right above the
`GATE: FAIL` line, so you can see which task and which metric moved. If it's an intended change
(you deliberately touched a threshold in `trajeval/harness.py`'s `gate()`), that's fine. If it's
a real regression in agent behavior, that's the build correctly refusing to merge a worse agent.

**The Streamlit Cloud build fails on an import of `trajeval`.** Double-check the main file path
is exactly `build_from_scratch/web_app.py` (not a copy elsewhere) — the whole point of this
app's layout is that it needs nothing extra as long as that path is right.

**The Space build fails on an import of `trajeval`.** On Path B, confirm `trajeval/` (the
package folder, not the whole `build_from_scratch/` folder) is sitting at the Space root, right
next to `app.py`. That sibling relationship is what makes the plain `import trajeval` work.

**Authentication fails on push.** GitHub no longer accepts your account password in the
terminal. Install the GitHub CLI (<https://cli.github.com>) and run `gh auth login`, following
the browser prompts. It handles auth for all future Git commands.
