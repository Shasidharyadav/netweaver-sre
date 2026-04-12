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

# NetWeaver SRE — Cyberpunk Ops Center 🛡️

[![OpenEnv](https://img.shields.io/badge/Spec-OpenEnv--0.2.0-indigo)](https://github.com/openenv/openenv)
[![Demo](https://img.shields.io/badge/Playground-Live%20Demo-emerald)](https://Shasidharyadavr-netweaver-sre.hf.space)
[![Difficulty](https://img.shields.io/badge/Missions-20%20Tasks-rose)](https://github.com/Shasidharyadav/netweaver-sre#missions)

**NetWeaver SRE** is a high-fidelity Reinforcement Learning environment designed for the next generation of Autonomous Site Reliability Engineers. Manage a 100-node GPU cluster, triage complex failure modes in real-time, and maintain 99.9% SLA uptime through precise, multi-turn decision making.

> [!IMPORTANT]
> **Phase 2 Compliant**: This environment and its inference suite are fully aligned with the strict OpenEnv Phase 2 validation protocols, including standardized logging, score clamping (0.001 - 0.999), and `socat` port-bridging for instant-start capability on Hugging Face Spaces.

---

## 🎮 The Playground Dashboard

Experience the **Cyberpunk Ops Center** — a vibrant, real-time dashboard for monitoring agent performance and cluster health.

- **Real-time Telemetry**: Live-updating charts for queue depths, gradient variances, and power consumption.
- **Deep Analytics**: Leaderboards for 13+ models, including benchmarking data for Qwen2.5, Llama-3.3, and DeepSeek.
- **Task Selector**: Instantly switch between all 20 specialized missions.

---

## 🛰️ Action & Observation Space

### Actions (`NetweaverSreAction`)
The agent has access to **20 specialized SRE commands** including:
- **Triage**: `DRAIN_TRAFFIC`, `RESTART_POD`, `KILL_ZOMBIE_PROCESS`
- **Network**: `TUNE_PFC_THRESHOLD`, `MITIGATE_ROUTE_FLAP`, `INCREASE_MTU`
- **Compute**: `ADJUST_POWER_CAP`, `PIN_CPU_THREADS`, `RESTART_GPU_DAEMON`
- **Deep Diagnostics**: `RUN_MINI_ITERATION`, `PURGE_CORRUPT_BLOCK`

### Observations (`NetweaverSreObservation`)
Agents receive high-density telemetry including:
- `hardware_logs`: Detailed system logs identifying specific error vectors.
- `gradient_variances`: A 10-rank array for identifying silent corruption (NaN contagion).
- `queue_depths`: Real-time network buffer monitoring.

---

## 🎯 Mission Gallery (20 Specialized Tasks)

| Difficulty | Tasks | Key Remediation |
|---|---|---|
| **Easy** | T01–T07 | Node Triage, DNS Cache, OOM Crashes, TLS Expiry |
| **Medium** | T08–T14 | PFC Buffer Tuning, Power Throttling, BGP Flapping |
| **Hard** | T15–T20 | Silent NaN Contagion, Broadcast Storms, Cluster Deadlock |

### Featured Missions:
![T15: NaN Contagion](server/assets/t15.png)
*T15: Silent NaN Contagion - Distortions in the gradient synchronisation layer require iterative isolation.*

![T08: PFC Tuning](server/assets/t08.png)
*T08: PFC Buffer Tuning - Network congestion requires precisely aligned hardware thresholds.*

---

## 📊 Performance Leaderboard

Verified scores from our real-time local benchmark suite:

| Model | Total Score | Avg Score | Resolved Tasks |
|:---|:---:|:---:|:---:|
| **Qwen2.5-72B-Instruct** | **17.97 / 20** | **0.899** | **18 / 20** |
| **Qwen2.5-Coder-32B** | **15.42 / 20** | **0.771** | **16 / 20** |
| **Llama-3.3-70B-Instruct** | **14.88 / 20** | **0.744** | **15 / 20** |
| **DeepSeek-R1-Distill-32B** | **13.10 / 20** | **0.655** | **14 / 20** |

---

## 🛠️ Setup & Deployment

### Quick Start (Local)
```bash
# Clone and setup
git clone https://github.com/Shasidharyadav/netweaver-sre
cd netweaver-sre
pip install -r server/requirements.txt

# Start the Ops Center
python -m server.app
```

### Docker Deployment (Hugging Face / General)
```bash
docker build -t netweaver-sre .
docker run -p 7860:7860 netweaver-sre
```

---

## 🔬 Phase 2 Validation Details
Netweaver SRE includes several infrastructure fixes derived from real validation failures:
- **`start.sh`**: Uses `socat` to bridge port 8000 to port 7860, ensuring the OpenEnv validator connects instantly.
- **Log Formatting**: Strictly adheres to the `[START]`, `[STEP]`, and `[END]` regex requirements.
- **Score Clamping**: Built-in `clamp_score` function ensures all reported rewards fall strictly within `(0.001, 0.999)`.

---

## 📁 Repository Map
```
netweaver-sre/
├── inference.py          # Main entry point for agent evaluation
├── start.sh              # Port-forwarding and startup script
├── openenv.yaml          # Environment metadata and inline graders
├── server/
│   ├── app.py            # FastAPI backend
│   ├── playground.html   # Cyberpunk Ops Center UI
│   └── assets/           # High-fidelity mission imagery
└── scripts/
    └── real_benchmark.py # Automated 20-task validation suite
```

Designed for the **Netweaver Hackathon 2026**.
