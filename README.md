---
title: NetWeaver SRE
emoji: 🚨
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Autonomous GPU cluster network fault triage RL environment
---

# NetWeaver-SRE: Autonomous Network Fabric Optimizer

An OpenEnv-compatible Reinforcement Learning environment for the **Meta PyTorch OpenEnv Hackathon**.

## Overview

This environment simulates a **100,000 GPU training cluster** where an agent must act as an elite AI Site Reliability Engineer (SRE) to:

- 🔴 **Easy**: Isolate offline nodes using `DRAIN_TRAFFIC`
- 🟡 **Medium**: Fix network buffer congestion by tuning PFC thresholds via `TUNE_PFC_THRESHOLD`
- 🔴 **Hard**: Track Silent Data Corruptions (NaN contagion) using binary search `RUN_MINI_ITERATION`, then isolate

## Action Space

| Command | Description |
|---|---|
| `DRAIN_TRAFFIC` | Reroute traffic away from a failing node |
| `TUNE_PFC_THRESHOLD` | Adjust Priority Flow Control buffer thresholds |
| `RUN_MINI_ITERATION` | Run a mini training pass on a sub-cluster range to detect gradient corruption |

## Observation Space

- `queue_depths`: Buffer utilization per spine switch
- `gradient_variances`: Per-rank gradient variance across GPU workers
- `hardware_logs`: Last 5 real-time system alerts
- `system_health`: Overall SLA health score

## Reward

- **1.0** for solving in 1 step (Easy/Medium) or minimal binary search turns (Hard)
- Penalty of **−0.1** per additional step taken
- **−0.2** for draining a healthy node
- **0.0** for timeout (15 steps max)

## API

```bash
# Reset environment
POST /reset

# Step with action
POST /step
{"command": "DRAIN_TRAFFIC", "target": "node_47"}

# Get state
GET /state
```
