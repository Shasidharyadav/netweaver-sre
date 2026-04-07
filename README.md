# NetWeaver SRE - Autonomous Incident Triage 🛡️

NetWeaver SRE is a robust Reinforcement Learning mini-game designed to evaluate an AI agent's ability to act as a Site Reliability Engineer managing a massive GPU cluster. 

This environment perfectly aligns with the **OpenEnv Framework constraints** and has been aggressively tuned for competitive evaluation!

## 🧩 The Problem Statement
Modern scalable AI deployments (like 100k-GPU Clos Topologies) suffer from complex telemetry issues (Network Collisions, Node Failures, and Silent NaN Contagions). Humans cannot triage these networks effectively. 

**NetWeaver SRE** tasks your Agentic LLM models with observing complex streaming telemetry (Gradient Variances Arrays, Network Queue Depths, and Base Error Logs), executing triage actions (`[START]`, `[STEP]`, `[END]`), and cleanly mitigating the cluster degradation before the SLA expires!

## 🔥 Robustness & Evaluation Criteria
The grading logic scales mathematically to punish inefficient random-guessing (`Reward decreases dynamically per step_count`) while awarding a clean `1.0 Score` to an LLM running the optimal triage process smoothly!

There are exactly 3 Graded Tasks that rigorously pressure the execution loops:
1. **Easy Mode**: Single node offline. Requires immediate parsing and isolation.
2. **Medium Mode**: Buffer PFC tuning. The AI agent must read the logs to identify an objective PFC routing threshold and input exactly the parsed `int`.
3. **Hard Mode**: Silent NaN Variance. The logs give no clues. The AI must interpret a long variance array payload, locate anomalous spikes (e.g., `999.9`), initiate a rapid Binary Search `RUN_MINI_ITERATION (X-Y)`, identify the exact underlying `node_XX`, and isolate it!

## 🚀 How to Run End-to-End
### 1. Launch the Server
```bash
# Start the OpenEnv server on 127.0.0.1:8000
python -m server.app
```

### 2. Manual Gameplay via the Holographic Dashboard (Demo)
- Navigate your browser to `http://127.0.0.1:8000/`.
- The game will automatically load in **Hard Mode**.
- Check the **Gradient Variances Array** and find the anomaly visually. 
- Transmit a `RUN_MINI_ITERATION` matching the zero-based index block!
- Check the System Logs to find the exact compromised Node.
- Overwrite action to `DRAIN_TRAFFIC` for that Node. You will see a `1.00 Reward` execution!

### 3. Automated Agent Inference Evaluation
```bash
# The backend script complies entirely with the mandatory OpenEnv CLI JSON pipeline format.
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```
*Enjoy your flawless SRE telemetry triage operations module!*
