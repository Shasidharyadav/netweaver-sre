---
title: Netweaver SRE
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - site-reliability-engineering
  - reinforcement-learning
  - autonomous-agents
---

# NetWeaver SRE — Autonomous Site Reliability Engineering 🛡️

[![OpenEnv](https://img.shields.io/badge/Spec-OpenEnv--0.2.0-indigo)](https://github.com/meta-pytorch/OpenEnv)
[![Demo](https://img.shields.io/badge/Playground-Live%20Demo-emerald)](https://Shasidharyadavr-netweaver-sre.hf.space)
[![Missions](https://img.shields.io/badge/Missions-22%20Tasks-rose)](#-mission-gallery-22-specialized-tasks)
[![Built for](https://img.shields.io/badge/Hackathon-Scaler%20%C3%97%20OpenEnv-blueviolet)](https://github.com/meta-pytorch/OpenEnv)

**Live HF Space:** [Shasidharyadavr/netweaver-sre](https://Shasidharyadavr-netweaver-sre.hf.space)

NetWeaver SRE is a high-fidelity OpenEnv reinforcement-learning environment for **autonomous site-reliability engineering on a 100-node GPU cluster**. The agent ingests structured telemetry — `hardware_logs`, `queue_depths`, `gradient_variances`, `gpu_memory_usage`, `system_health` — and must issue the correct remediation command (with the correct target and value) within a 15-step SLA budget.

---

## 📖 The Story

### 🎯 Problem
Frontier LLMs can read a single error message, but they are **not trained on closed-loop SRE work**: read structured telemetry → locate the faulty entity → pick the right tool → parameterise it correctly → recover from wrong moves within an SLA. Today's "AI on-call" assistants degrade rapidly past the first surface-level fix because there's no environment that actually *teaches* this skill via RL.

### 🛠️ Environment
NetWeaver SRE simulates a 100-node GPU cluster experiencing **22 different failure modes**: node offline, DNS cache poisoning, OOM crashes, TLS expiry, disk full, unhealthy pods, zombie processes, PFC congestion, power throttling, BGP flap, MTU mismatch, DDoS, connection pool exhaustion, CPU storms, NaN contagion in gradients, broadcast storms, GPU memory leaks, cluster deadlocks, network partitions, corrupt DB blocks, **cascading failure chains**, and **gradient poisoning with broadcast amplification**. Every reset randomises the affected entity (which node, which switch, which DB, which cluster) so the agent **must read the alert** — it cannot memorise names. Three of the Hard tasks (T15, T21, T22) require **2–3 commands in the correct order** for full credit.

The agent receives a rich `NetweaverSreObservation` (`alert`, `hardware_logs`, `queue_depths`, `gradient_variances`, `gpu_memory_usage`, `system_health`) and emits a typed `NetweaverSreAction` (`command`, `target`, optional integer `value`). Reward comes from a **3-section composable rubric** built on `openenv.core.rubrics.base.Rubric` (RFC 004): diagnosis (40%) + resolution (40%) + best-practice (20%).

### 📊 Results
A heuristic epsilon-decay policy improved from a random baseline of **0.540** average rubric reward to **0.989** after 50 training steps against the live environment — Δ = **+0.449**. Per-difficulty gains: Easy +0.50, Medium +0.47, Hard (incl. multi-step) +0.39. Plots and reproduction commands are in [📈 Training Evidence](#-training-evidence) below. Full GRPO with `Qwen/Qwen2.5-0.5B-Instruct` is one click away in `notebooks/train_unsloth_netweaver.ipynb` on a free Colab T4.

### 💡 Why It Matters
A model trained on NetWeaver SRE would be measurably better at the kind of structured-telemetry reasoning that real on-call SREs perform every day. No existing benchmark measures this: HumanEval / MMLU / MATH all test single-turn knowledge, not the close-the-loop "look at logs → pick a tool → parameterise → recover" pattern. That makes this environment a candidate building block for genuinely useful **AI SREs** — the kind of agent that could safely sit in a production rotation. A researcher could write a paper on training/eval methodology for closed-loop infrastructure agents using this environment as the benchmark.

---

---

## ✨ What's New (v3)

- **Single source of truth.** `server/app.py` now delegates `/reset` and `/step` directly to `NetweaverSreEnvironment`, removing a parallel hard-coded `FAULT_SCENARIOS` dict and making sure every alert is **freshly randomised** per episode.
- **22 task scenarios** (T01–T22), with three new multi-step Hard tasks: T15 (NaN contagion), T21 (cascading failure chain), T22 (gradient poisoning + amplification).
- **Order-enforcing grader.** T15 / T21 / T22 require the right commands **in the right order** for full resolution score; out-of-order issuances get partial credit.
- **Strict score clamping** to `(0.001, 0.999)` everywhere a score touches HTTP — verified by `scratch/smoke_test.py`.
- **Real training evidence** generated with `scripts/run_training_demo.py` against the live env (not synthetic data).
- **OpenEnv Rubric system (RFC 004).** `rubrics.py` defines `DiagnosisRubric`, `ResolutionRubric`, `BestPracticeRubric` — each subclassing `openenv.core.rubrics.base.Rubric` — and a top-level `NetWeaverSREComposedRubric` that auto-registers them as children. The `/grader` endpoint and `compute_grader_score()` both delegate here.
- **Composed reward functions** (`reward_action_parses` + `reward_correct_command` + `reward_episode_resolution`) — passed as a list to GRPOTrainer for a rich learning signal, matching the official OpenEnv × Unsloth tutorial pattern.
- **Two Colab notebooks**: a TRL-only path and an Unsloth + LoRA path with `Qwen/Qwen2.5-0.5B-Instruct` (free T4 friendly).
- **Real loss + reward + same-axes baseline-vs-trained plots** committed to `server/assets/`.

---

## 🎯 Mission Gallery (22 Specialized Tasks)

| Difficulty | Tasks | Key Remediation |
|---|---|---|
| **Easy** (T01–T07) | Node offline, DNS cache, OOM crash, TLS expiry, disk full, unhealthy pod, zombie process | `DRAIN_TRAFFIC`, `CLEAR_DNS_CACHE`, `RESTART_SERVICE`, `RENEW_CERTIFICATE`, `CLEAR_TEMP_FILES`, `RESTART_POD`, `KILL_ZOMBIE_PROCESS` |
| **Medium** (T08–T14) | PFC congestion, power throttle, BGP flap, MTU mismatch, DDoS, conn pool exhaustion, CPU storm | `TUNE_PFC_THRESHOLD`, `ADJUST_POWER_CAP`, `MITIGATE_ROUTE_FLAP`, `INCREASE_MTU=9000`, `SET_RATE_LIMIT`, `SCALE_CONN_POOL`, `PIN_CPU_THREADS` |
| **Hard** (T15–T22) | NaN contagion, broadcast storm, GPU leak, cluster deadlock, network partition, corrupt DB, **cascading failure**, **gradient poisoning + amplification** | Multi-step ordered chains: e.g. T21 = power → CPU → DB; T22 = triage → isolate → drain |

Each task **randomises the target entity** on every `/reset`. The agent must read the alert and identify the specific node/cluster/switch/DB — there are no fixed names.

---

## 🛰️ Action & Observation Space

### Actions (`NetweaverSreAction`)
20 specialised SRE commands, e.g.:
- **Triage**: `DRAIN_TRAFFIC`, `RESTART_POD`, `KILL_ZOMBIE_PROCESS`
- **Network**: `TUNE_PFC_THRESHOLD`, `MITIGATE_ROUTE_FLAP`, `INCREASE_MTU`, `ISOLATE_BROADCAST_STORM`, `REBOOT_LEAF_SWITCHES`
- **Compute**: `ADJUST_POWER_CAP`, `PIN_CPU_THREADS`, `RESTART_GPU_DAEMON`
- **Deep diagnostics**: `RUN_MINI_ITERATION`, `ISSUE_GLOBAL_ROLLBACK`, `PURGE_CORRUPT_BLOCK`, `SCALE_CONN_POOL`

### Observations (`NetweaverSreObservation`)
- `alert` — short freeform incident header
- `hardware_logs` — recent log lines (contains the randomised entity name)
- `queue_depths` — per-switch buffer utilisation
- `gradient_variances` — per-rank gradient stats (NaN → -1.0, poison → 999.9)
- `gpu_memory_usage` — per-sub-cluster memory utilisation
- `system_health` — global SLA score
- `step_count`, `reward`, `done`, `error_rate`, `active_connections`

---

## 🧮 Rubric Grader (`/grader`)

Every episode is scored by a 3-section deterministic rubric:

| Section | Weight | What it rewards |
|---|---|---|
| **Diagnosis** | 40% | Reading the right obs field (20%) + targeting the right entity class (20%) |
| **Resolution** | 40% | Issuing all required commands (with order enforcement on T15/T21/T22) and a value in the valid range; multiplied by an efficiency factor (-5% per step over `ideal_steps`, floor 50%) |
| **Best Practice** | 20% | No destructive commands, error rate < 30% |

The endpoint returns a structured payload:

```json
{
  "resolved": true,
  "total": 0.94,
  "breakdown": {"diagnosis": 0.40, "resolution": 0.36, "best_practice": 0.20}
}
```

All sub-scores and the total are clamped to `(0.001, 0.999)` per the OpenEnv spec.

---

## 📈 Training Evidence

We trained an **epsilon-decay heuristic policy** for 50 steps against the live environment, with 30-episode evaluations before and after. This is honest, real RL evidence (no GPU required). Full GRPO with `Qwen/Qwen2.5-0.5B-Instruct` is available via `python train_grpo.py` on a Colab T4.

| Metric | Before (random) | After (trained) | Δ |
|:---|:---:|:---:|:---:|
| Average reward (30 eps) | **0.540** | **0.989** | **+0.449** |
| Easy tasks | 0.48 | 0.98 | +0.50 |
| Medium tasks | 0.51 | 0.98 | +0.47 |
| Hard tasks (incl. multi-step) | 0.61 | 1.00 | +0.39 |

![Reward Curve](server/assets/reward_curve.png)
*Per-episode rubric reward across 50 training steps, with a 5-step moving average overlay.*

![Loss Curve](server/assets/loss_curve.png)
*Training loss (1 − rubric reward) over the same 50 steps — the standard rubric-loss convention.*

![Baseline vs Trained](server/assets/baseline_vs_trained.png)
*Per-episode reward for the random baseline (purple) and trained policy (green) on the **same axes**, with horizontal mean lines for direct comparison.*

![Before vs After](server/assets/before_after.png)
*Aggregate reward improvement from random baseline to trained policy.*

![Difficulty breakdown](server/assets/difficulty_breakdown.png)
*Per-difficulty improvement. Hard multi-step tasks (T15/T21/T22) now resolve fully thanks to the order-enforcing grader and ordered remediation logic.*

Reproduce locally with:

```bash
python scripts/run_training_demo.py    # heuristic, no GPU
# or
python train_grpo.py                   # full GRPO on T4
```

---

## 🧠 Why This Matters

Today's frontier models can read a single error message, but they have **no closed-loop training signal** for SRE-style work — diagnosing a fault, picking the right tool, parameterising it, and recovering from wrong moves within an SLA. NetWeaver SRE is a self-contained training surface for exactly that capability:

- **Structured telemetry parsing**, not just text
- **Multi-step ordered remediation** — T15/T21/T22 punish out-of-order moves
- **Randomised entities** — the policy must extract names from logs, not memorise them
- **Composable rubric** — diagnosis vs. resolution vs. best-practice are scored separately so the agent gets a *rich, informative* signal (not just a 0/1 at the end)

A model trained here would be measurably better at real-world on-call work — something no existing benchmark captures.

---

## 🛠️ Quick Start

### Local server

```bash
git clone https://github.com/Shasidharyadav/netweaver-sre
cd netweaver-sre
pip install -r server/requirements.txt
python -m server.app
# Server available at http://0.0.0.0:8000
```

### Smoke test

```bash
python scratch/smoke_test.py
# Verifies randomisation, multi-step tasks, score clamping, /grader structure
```

### Training (heuristic, no GPU)

```bash
python scripts/run_training_demo.py
# Outputs: training_results.json + server/assets/*.png
```

### Training (real GRPO on Colab T4)

Two Colab notebooks are provided:

| Notebook | Backend | Model | When to use |
|---|---|---|---|
| `notebooks/train_unsloth_netweaver.ipynb` | **Unsloth + TRL** | Qwen2.5-0.5B (or GPT-OSS 20B) | Recommended. Uses 4-bit + LoRA so it fits on free T4. Composed reward functions, TrackIO live plots. |
| `notebooks/train_netweaver_grpo.ipynb` | TRL only | Qwen2.5-0.5B | Simpler dependency footprint, no Unsloth. |

Open in Colab → Runtime → Change runtime type → T4 GPU → **Run all**.

### Inference benchmark

```bash
HF_TOKEN=hf_xxx ENV_URL=http://0.0.0.0:8000 python inference.py
```

### Docker (HF Spaces / general)

```bash
docker build -t netweaver-sre .
docker run -p 7860:7860 netweaver-sre
```

---

## 🔬 OpenEnv Compliance

- Built on top of `openenv-core` `Environment` base class with proper `reset`/`step`/`state` API
- Strict client/server separation (`client.py` only imports from `models.py`)
- All scores clamped to `(0.001, 0.999)` — never 0.0 or 1.0
- `start.sh` bridges port 8000 → 7860 via `socat` for HF Spaces
- Standardised `[START] / [STEP] / [END]` log lines in `inference.py`

---

## ✅ Hackathon Submission Checklist

| Requirement (from `hack.md`) | Status | Evidence |
|---|:---:|---|
| Use OpenEnv (latest release: 0.2.3) | ✓ | `openenv-core[core]>=0.2.3` in `pyproject.toml`; `Environment` + `Rubric` base classes both used |
| Working training script (Unsloth or TRL, ideally Colab notebook) | ✓ | `notebooks/train_unsloth_netweaver.ipynb` + `notebooks/train_netweaver_grpo.ipynb` + `train_grpo.py` |
| Evidence of actual training (loss + reward plots from a real run) | ✓ | `server/assets/{reward_curve,loss_curve,baseline_vs_trained,before_after,difficulty_breakdown}.png` from real rollouts |
| Uses OpenEnv's **Rubric system** (composable rubrics) | ✓ | `rubrics.py` — `NetWeaverSREComposedRubric` extends `openenv.core.rubrics.base.Rubric` with 3 child rubrics |
| README that motivates problem + explains env + shows results | ✓ | This document |
| Push environment to a Hugging Face Space | ✓ | [Shasidharyadavr/netweaver-sre](https://Shasidharyadavr-netweaver-sre.hf.space) |
| Short writeup (blog / video / slides) linked from README | 📝 | See **Writeup & Demo** below |
| No large video files in HF Hub submission | ✓ | All assets are PNGs ≤ 700 KB |
| `Environment` / `MCPEnvironment` base class used properly | ✓ | `NetweaverSreEnvironment(Environment)` |
| Standard Gym-style API (`reset`, `step`, `state`) | ✓ | All three implemented; `state` is a property |
| Valid `openenv.yaml` manifest | ✓ | 22 tasks, all with `grader: {type: deterministic, endpoint: /grader}` |
| Reserved tool names not used for MCP tools | ✓ | All commands use uppercase domain names like `DRAIN_TRAFFIC` |

### 🎬 Writeup & Demo

- **Live HF Space**: [Shasidharyadavr-netweaver-sre.hf.space](https://Shasidharyadavr-netweaver-sre.hf.space)
- **Architecture write-up**: see `📈 Training Evidence` and `🧠 Why This Matters` sections above
- **Reproduction**: `python scripts/run_training_demo.py` (no GPU) or `notebooks/train_unsloth_netweaver.ipynb` (Colab T4)
- **Verification suite**: `python scratch/smoke_test.py` and `python scratch/verify_all_tasks.py`

> ➕ **TODO before submission**: paste links to the 2-minute demo video / HF blog post / slide deck here so judges can find them in one click.

### Pre-Deploy Sanity Checks

Run these locally before pushing to HF Spaces:

```bash
# 1. All 22 tasks resolve with the optimal heuristic
python scratch/verify_all_tasks.py

# 2. Bug-fix invariants (randomization, multi-step, score clamping, /grader)
python scratch/smoke_test.py

# 3. Composed reward functions sane
python scratch/test_reward_funcs.py

# 4. Heuristic training reproduces the curves
python scripts/run_training_demo.py
```

All four should exit `0` with no traceback.

---

## 📁 Repository Map

```
netweaver-sre/
├── server/
│   ├── app.py                          # FastAPI server — delegates to env
│   ├── netweaver_sre_environment.py    # Single source of truth (22 tasks)
│   ├── playground.html                 # Cyberpunk Ops Center UI
│   └── assets/                         # Plot images + mission art
├── rubrics.py                          # OpenEnv Rubric subclasses (RFC 004)
├── graders.py                          # Backward-compat wrapper around rubrics.py
├── reward_shaper.py                    # Per-step shaping bonus
├── models.py                           # NetweaverSreAction / NetweaverSreObservation
├── client.py                           # OpenEnv client wrapper
├── inference.py                        # 22-task benchmark loop
├── train_grpo.py                       # GRPO trainer (Qwen2.5-0.5B-Instruct)
│                                       # — composed reward functions live here
├── scripts/
│   ├── run_training_demo.py            # Heuristic training (no GPU, real data)
│   └── plot_results.py                 # Re-render plots from JSON
├── notebooks/
│   ├── train_unsloth_netweaver.ipynb   # Unsloth + GRPO Colab notebook (recommended)
│   └── train_netweaver_grpo.ipynb      # Vanilla TRL Colab notebook
├── scratch/
│   ├── smoke_test.py                   # 10-step end-to-end verification
│   ├── verify_all_tasks.py             # All 22 tasks resolve
│   └── test_reward_funcs.py            # Composed reward function unit tests
└── openenv.yaml                        # 22-task manifest
```

---

Designed for the **Scaler × OpenEnv Hackathon 2026**.
