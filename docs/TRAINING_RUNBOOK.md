# NetWeaver SRE — Training Runbook

This document captures **every step we executed** for both training runs:

1. **Run #1 — Heuristic baseline training (no GPU)** — produces honest noisy curves locally on any machine.
2. **Run #2 — Real GRPO training on HF Cloud GPU** — the headline submission run.

Both runs share the same live OpenEnv environment at
`https://shasidharyadavr-netweaver-sre.hf.space`.

---

## Pre-requisites (one-time setup)

### 1. HF account + token

You need a Hugging Face account at <https://huggingface.co> (the `Shasidharyadavr` account in our case).

Create or edit a token at <https://huggingface.co/settings/tokens>:

| Permission | Why we need it |
|---|---|
| `Repos → write` | Push code/results to the Space `Shasidharyadavr/netweaver_sre` |
| `Inference → Serverless` (optional) | If you call the inference router from `inference.py` |
| **`Jobs → write`** | **Required** for `hf jobs uv run` (cloud GPU runs) |

> ⚠️ The token used during the hackathon (`hf_yyhxog…ztkG`) was at one point only `repo:write`. We had to add `Jobs → write` mid-run before the cloud GPU launch.

### 2. Local CLI

```powershell
# Install/upgrade huggingface_hub which provides the `hf` CLI
pip install --upgrade huggingface_hub

# The `hf` binary lands in ...\Python313\Scripts; ensure that's on PATH
$env:Path += ";C:\Users\shash\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"

# Login (interactively, OR via env var as shown below)
$env:HF_TOKEN = "<your hf_… token>"
hf auth login --token $env:HF_TOKEN --add-to-git-credential
hf auth whoami      # should print: user=Shasidharyadavr
```

### 3. Verify the live env is reachable

```powershell
curl https://shasidharyadavr-netweaver-sre.hf.space/health
# {"status":"ok","version":"3.0.0"}
```

---

## Run #1 — Heuristic Baseline Training (no GPU, ~10s)

This is the **first run we did** to produce honest noisy curves on this Windows
machine where torch/CUDA aren't available. It uses an epsilon-decay heuristic
policy that calls the live HF Space env via `fastapi.testclient.TestClient`.

### What it does

1. Boot the FastAPI env in-process (no external server)
2. Run **30 evaluation episodes** with a uniform-random policy → `before_rewards`
3. Run **50 training steps** with epsilon-decay (1.0 → 0.05) → `training_rewards`
4. Run **30 evaluation episodes** with the trained low-epsilon policy → `after_rewards`
5. Save `training_results.json` + 5 PNG plots into `server/assets/`

### Exact command

```powershell
cd c:\Users\shash\Desktop\Shasi_codes\OPEN_env\netweaver_sre
python scripts\run_training_demo.py
```

### Knobs (env vars)

| Variable | Default | What it does |
|---|---:|---|
| `DEMO_STEPS` | 50 | Number of training steps |
| `DEMO_EVAL` | 30 | Number of eval episodes (per before/after phase) |

### Result we got

```
[EVAL] BEFORE training (epsilon=1.0)... before_avg=0.560 (1.5s)
[step 50/50] task=t22 eps=0.07 reward=0.999
[EVAL] AFTER training (epsilon=0.05)...  after_avg=0.989 (1.0s)
=== Done. before_avg=0.560 after_avg=0.989 delta=+0.429 ===
```

### Files produced

```
training_results.json                     # source: "heuristic_training_demo"
server/assets/reward_curve.png            # noisy ascending curve
server/assets/loss_curve.png              # 1 - reward, descending
server/assets/baseline_vs_trained.png     # both runs on same axes (hack.md §278)
server/assets/before_after.png            # aggregate bar chart
server/assets/difficulty_breakdown.png    # easy/medium/hard split
```

### Failure modes we hit & fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `ImportError: DLL load failed while importing _path` | Windows Application Control policy blocks `matplotlib` native DLLs | Pillow fallback renderer in `_plot_with_pillow()` (no native deps) |
| `ImportError: cannot import name 'jobs' from 'huggingface_hub'` | The Python API has the CLI separately | Use `hf` CLI binary, not the Python module |

---

## Run #2 — Real GRPO Training on HF Cloud GPU (T4, ~15-25 min, ~$0.20)

This is the **headline run** — actual policy-gradient training of
`Qwen/Qwen2.5-0.5B-Instruct` against the live env, using the OpenEnv Rubric
system as reward signal.

### What it does

1. **Launch a job** on HF infrastructure with `hf jobs uv run` against
   `scripts/hf_job_train.py`.
2. The job container (T4-small, 16 GB GPU, $0.40/hr) does:
   - `git clone` the netweaver_sre HF Space → has all our code (`train_grpo.py`, `rubrics.py`, etc.)
   - Probe `/health` on the live env to confirm reachability
   - Eval Qwen2.5-0.5B on 6 random tasks → `before_rewards`
   - GRPO training for 30 steps using **3 composed reward functions**
     (`reward_action_parses` + `reward_correct_command` + `reward_episode_resolution`),
     each multi-step rollout hitting the live env via HTTP.
   - Eval again → `after_rewards`
   - **Upload the new `training_results.json` + 5 plots back to the Space**,
     replacing the heuristic placeholder data.

### Exact command we ran

```powershell
cd c:\Users\shash\Desktop\Shasi_codes\OPEN_env\netweaver_sre

hf jobs uv run `
    --flavor t4-small `
    -s HF_TOKEN `
    -e "ENV_URL=https://shasidharyadavr-netweaver-sre.hf.space" `
    -e "MAX_TRAIN_STEPS=30" `
    -e "EVAL_EPISODES=6" `
    -e "MAX_STEPS_PER_EPISODE=5" `
    --timeout 45m `
    -d `
    scripts\hf_job_train.py
```

### What `-d` and `-s` do

| Flag | Effect |
|---|---|
| `--flavor t4-small` | 1× T4 GPU, 16 GB VRAM, $0.40/hr |
| `-s HF_TOKEN` | Inject `$env:HF_TOKEN` as a **secret** env var inside the container (not echoed to logs) |
| `-e "KEY=VALUE"` | Plain env var (visible in `hf jobs inspect`) |
| `--timeout 45m` | Hard cap so a hang doesn't burn credits |
| `-d` | Detached: returns the Job ID immediately, runs in the background |

### Output

```
Job started with ID: 69ecce86d2c8bd8662bcdc4e
View at: https://huggingface.co/jobs/Shasidharyadavr/69ecce86d2c8bd8662bcdc4e
```

### Following the job

```powershell
# Status (one-shot)
hf jobs inspect 69ecce86d2c8bd8662bcdc4e

# Live logs (streams until job finishes)
hf jobs logs 69ecce86d2c8bd8662bcdc4e

# List recent jobs
hf jobs ps

# Cancel if needed
hf jobs cancel 69ecce86d2c8bd8662bcdc4e
```

### Job lifecycle stages

| Stage | What's happening | Duration |
|---|---|---|
| `SCHEDULING` | Picking a GPU node | 30 s – 2 min |
| `INITIALIZING` | Pulling Docker image + installing PEP-723 deps with `uv` | 2 – 5 min |
| `RUNNING` | Actual training loop | 10 – 20 min |
| `COMPLETED` / `FAILED` | Final state | – |

### Failure modes we hit & fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `403 Forbidden … missing permissions: job.write` | The HF token only had `repo:write`, not Jobs scope | Edit the token at <https://huggingface.co/settings/tokens> and tick **Jobs → write** |
| `Token is valid (permission: fineGrained)` warning | First-time CLI login | Normal; the warning just notes the token type |
| Job stuck in `SCHEDULING` for >5 min | T4 capacity surge | Re-launch with `--flavor a10g-small` (slightly higher cost, near-instant scheduling) |

### Result we got

After the job finished:

- The HF Space's `training_results.json` flipped from `source: "heuristic_training_demo"` to `source: "grpo_real_run"`.
- The 5 plots in `server/assets/` were replaced with the real GRPO curves (commit message: *"Real GRPO training run (30 steps, Qwen/Qwen2.5-0.5B-Instruct)"*).

To pull the fresh artifacts back to the local repo:

```powershell
git pull origin main
```

---

## Verifying everything still works after a training run

```powershell
# All three should exit 0 with no traceback
python scratch\smoke_test.py            # 10 invariants
python scratch\verify_all_tasks.py      # 22/22 tasks resolve
python scratch\test_reward_funcs.py     # composed reward funcs

# And the live HF Space:
curl https://shasidharyadavr-netweaver-sre.hf.space/health     # {"status":"ok","version":"3.0.0"}
curl https://shasidharyadavr-netweaver-sre.hf.space/tasks      # 22 tasks
curl https://shasidharyadavr-netweaver-sre.hf.space/grader     # {"resolved":...,"total":...,"breakdown":{...}}
```

---

## Cost ledger (this hackathon)

| Item | Cost |
|---|---:|
| HF Space hosting (cpu-basic, always-on) | Free |
| Run #1 heuristic training (local CPU) | Free |
| Run #2 GRPO training (t4-small × 25 min) | ~$0.17 |
| **Total** | **~$0.17 of $30 hackathon credit** |

---

## How to redo a fresh training run later

```powershell
cd c:\Users\shash\Desktop\Shasi_codes\OPEN_env\netweaver_sre

# Optional: bump the step count for a longer run
hf jobs uv run --flavor t4-small -s HF_TOKEN `
    -e "ENV_URL=https://shasidharyadavr-netweaver-sre.hf.space" `
    -e "MAX_TRAIN_STEPS=100" `
    -e "EVAL_EPISODES=10" `
    -e "MAX_STEPS_PER_EPISODE=6" `
    --timeout 90m -d `
    scripts\hf_job_train.py

# Then follow the logs
hf jobs logs <returned-job-id>
```
