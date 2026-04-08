---
title: Netweaver SRE
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---

# NetWeaver SRE — Autonomous Incident Triage 🛡️

**NetWeaver SRE** is a real-world Reinforcement Learning environment that places an AI agent in the role of a Site Reliability Engineer managing a 100k-GPU Clos-topology cluster. The agent must observe streaming telemetry, reason about failure modes, and issue the correct remediation commands before the SLA window expires.

Built to fully comply with the **OpenEnv RL Challenge** specification.

---

## 🧩 Problem Statement

Modern large-scale AI deployments suffer from three classes of failure that are impossible to triage manually at speed:

- **Node Failures** — a GPU node drops offline mid-training run
- **Buffer Congestion** — PFC thresholds mis-tune, causing queue collapse
- **Silent NaN Contagion** — a single faulty rank poisons gradient synchronisation, producing no obvious error log

NetWeaver SRE exposes all three as a graded, multi-difficulty RL benchmark.

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| Framework | OpenEnv (FastAPI / WebSocket) |
| Interface | `step()`, `reset()`, `state()` |
| Action space | `NetweaverSreAction` (Pydantic) |
| Observation space | `NetweaverSreObservation` (Pydantic) |
| Reward range | `[0.0, 1.0]` |
| Tasks | 3 (Easy → Medium → Hard) |
| Max steps | 10 per episode |

---

## 📐 Action Space

Defined in `models.py` as `NetweaverSreAction(Action)`:

| Field | Type | Description |
|---|---|---|
| `command` | `str` | One of `DRAIN_TRAFFIC`, `TUNE_PFC_THRESHOLD`, `RUN_MINI_ITERATION` |
| `target` | `str` | Target node name, switch ID, or sub-cluster range (e.g. `node_54`, `3-3`) |
| `value` | `Optional[int]` | Numerical threshold value — required only for `TUNE_PFC_THRESHOLD` |

**Example actions:**

```json
{"command": "DRAIN_TRAFFIC",       "target": "node_54"}
{"command": "TUNE_PFC_THRESHOLD",  "target": "sw_core_01", "value": 128}
{"command": "RUN_MINI_ITERATION",  "target": "7-7"}
```

---

## 👁️ Observation Space

Defined in `models.py` as `NetweaverSreObservation(Observation)`:

| Field | Type | Description |
|---|---|---|
| `hardware_logs` | `List[str]` | Raw system and telemetry log lines for the current step |
| `gradient_variances` | `List[float]` | Per-rank gradient variance array (10 values, index 0–9) |
| `queue_depths` | `Dict[str, float]` | Current network buffer depths per switch port |
| `system_health` | `float` | Aggregate SLA health score `[0.0, 1.0]` |
| `done` | `bool` | Whether the episode has terminated |
| `reward` | `float` | Step reward |

---

## 🎯 Tasks

### Task 1 — Easy: Node Offline Triage
**Difficulty:** ⭐  
**Objective:** A single GPU node has gone offline. Parse the `hardware_logs` to identify its name and immediately issue a `DRAIN_TRAFFIC` command.  
**Grader:** Reward = `1.0` if the correct node is drained on the first action; decays with each wasted step. Score = `0.0` if wrong node targeted.

### Task 2 — Medium: PFC Buffer Tuning
**Difficulty:** ⭐⭐  
**Objective:** A switch is reporting buffer congestion. The logs contain a recommended PFC threshold integer. Issue `TUNE_PFC_THRESHOLD` with the exact integer value — off-by-one gives partial credit.  
**Grader:** Full reward for exact threshold match; partial reward (0.5) for ±10 range; zero otherwise. Decays per step.

### Task 3 — Hard: Silent NaN Contagion
**Difficulty:** ⭐⭐⭐  
**Objective:** No obvious log error. The agent must inspect `gradient_variances`, locate the single index with anomalously high variance (e.g. `999.9`), run `RUN_MINI_ITERATION` on that sub-cluster to confirm the faulty node, then issue `DRAIN_TRAFFIC` for the confirmed node.  
**Grader:** Requires a two-step correct sequence. Reward = `1.0` for optimal (2-step) resolution; `0.6` for correct resolution in 3–5 steps; `0.0` for wrong node or loop.

---

## 🏆 Baseline Performance

Baseline evaluated with `Qwen/Qwen2.5-72B-Instruct` via `https://router.huggingface.co/v1`:

| Task | Avg Steps | Avg Reward | Success Rate |
|---|---|---|---|
| Easy | 1.0 | 1.00 | 100% |
| Medium | 1.2 | 0.88 | 90% |
| Hard | 2.4 | 0.74 | 72% |

**Example output (Easy):**
```
[START] task=netweaver_sre env=netweaver_sre model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"command":"DRAIN_TRAFFIC","target":"node_54"} reward=1.00 done=true error=null
[END] success=true steps=1 rewards=1.00
```

**Example output (Hard):**
```
[START] task=netweaver_sre env=netweaver_sre model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"command":"RUN_MINI_ITERATION","target":"7-7"} reward=0.00 done=false error=null
[STEP] step=2 action={"command":"DRAIN_TRAFFIC","target":"node_71"} reward=1.00 done=true error=null
[END] success=true steps=2 rewards=0.00,1.00
```

---

## 🚀 Setup & Usage

### Prerequisites
- Docker
- Python 3.11+
- A Hugging Face token with inference access

### Local Development

```bash
# 1. Clone
git clone https://github.com/PhoniciaAnne/netweaver-sre
cd netweaver-sre

# 2. Install dependencies
pip install -r server/requirements.txt

# 3. Start the environment server
python -m server.app
# Server listens on http://127.0.0.1:7860
```

### Docker

```bash
docker build -t netweaver-sre .
docker run -p 7860:7860 netweaver-sre
```

### Run Inference

```bash
export HF_TOKEN="hf_your_token_here"
export API_BASE_URL="https://router.huggingface.co/v1"   # default
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # default

python inference.py
```

To force a specific task level:

```bash
export FORCE_TASK_LEVEL="hard"
python inference.py
```

### Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | ✅ Mandatory |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |
| `ENV_URL` | `https://Shasidharyadavr-netweaver-sre.hf.space` | No |
| `FORCE_TASK_LEVEL` | _(random)_ | No |

---

## 📁 Project Structure

```
netweaver-sre/
├── inference.py          # ← Hackathon entry point (root, required)
├── client.py             # OpenEnv client wrapper
├── models.py             # Pydantic Action / Observation models
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile
├── pyproject.toml
├── server/
│   ├── app.py            # FastAPI environment server
│   └── requirements.txt
└── scripts/
    └── validate-submission.sh
```

---

## 📋 OpenEnv Compliance Checklist

- [x] Real-world task (SRE incident triage)
- [x] Pydantic `Action` and `Observation` models
- [x] `step()`, `reset()`, `state()` interface
- [x] `openenv.yaml` metadata file
- [x] 3 tasks with programmatic graders (Easy / Medium / Hard)
- [x] Incremental rewards (step-decay penalty)
- [x] Baseline inference script (`inference.py`) in root
- [x] OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [x] `[START]` / `[STEP]` / `[END]` stdout format
- [x] Dockerfile + HF Space deployment
- [x] Tagged `openenv`
