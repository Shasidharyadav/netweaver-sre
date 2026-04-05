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

This environment simulates a large-scale AI training cluster dealing with **Network switch failures** and hardware faults. Your AI agent must act as an elite AI Site Reliability Engineer (SRE) to triage three distinct difficulties:

- 🟢 **Easy (Log Parsing & Action Execution)**: A node goes offline. The agent must parse the system hardware logs to identify the correct node ID and isolate it using `DRAIN_TRAFFIC`.
- 🟡 **Medium (Threshold Tuning & Variable Extraction)**: A spine switch incurs a buffer congestion overshoot. The agent must successfully parse and extract the target integer threshold from the logs and apply it via `TUNE_PFC_THRESHOLD`.
- 🔴 **Hard (Multi-step Deductive Reasoning & Binary Search)**: A Silent Data Corruption (NaN contagion) has infected the network with no specific log identifying the faulty node. The agent must use multi-step deductive reasoning by splitting the cluster into ranges and calling `RUN_MINI_ITERATION` to recursively binary-search the anomaly down to a single switch, and *then* isolate it.

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
