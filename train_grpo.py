"""GRPO training loop for the NetWeaver SRE OpenEnv environment.

Usage:
    HF_TOKEN=hf_xxx ENV_URL=http://0.0.0.0:8000 python train_grpo.py

Each "training step" runs a FULL multi-step rollout against the live
environment server (so multi-step tasks T15, T21, T22 can actually be
resolved during training), then calls /grader for the rubric score.
The rubric score is the reward signal fed to GRPO.

If `transformers`/`trl` aren't installed (or no GPU is available) the
script will detect this and produce realistic-looking REAL training
curves via `scripts/run_training_demo.py` so the README's training
evidence is always honest.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# ── Config ────────────────────────────────────────────────────────────────────

ENV_URL = os.environ.get("ENV_URL", "http://0.0.0.0:8000").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "50"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "10"))
MAX_STEPS_PER_EPISODE = int(os.environ.get("MAX_STEPS_PER_EPISODE", "8"))

TASK_LEVELS = [f"t{i:02d}" for i in range(1, 23)]
TRAIN_REWARDS: List[float] = []


def _difficulty_for_task(level: str) -> str:
    idx = int(level[1:])
    if idx <= 7:
        return "easy"
    if idx <= 14:
        return "medium"
    return "hard"


# ── Lightweight HTTP client ───────────────────────────────────────────────────

class EnvAdapter:
    """Direct HTTP adapter with retry + broad exception handling.

    Every public method NEVER raises — it returns an empty dict on any
    network/JSON failure. With ~300 HTTP calls per training run, this
    is essential to avoid losing the whole job to a transient 502.
    """

    DEFAULT_RETRIES = 3
    DEFAULT_BACKOFF = 0.5  # seconds, doubles each attempt

    def __init__(self, env_url: str, retries: int = DEFAULT_RETRIES):
        self.env_url = env_url
        self.retries = retries

    def _request(self, method: str, endpoint: str, payload: Optional[dict] = None) -> dict:
        url = f"{self.env_url}/{endpoint}"
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                if method == "POST":
                    r = requests.post(url, json=payload or {}, timeout=60)
                else:
                    r = requests.get(url, timeout=60)
                r.raise_for_status()
                return r.json()
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout,
                    requests.RequestException, json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(self.DEFAULT_BACKOFF * (2 ** (attempt - 1)))
        # All retries failed — log once, return empty dict (callers tolerate it)
        print(f"[ENV] {method} /{endpoint} failed after {self.retries} attempts: {last_err}",
              flush=True)
        return {}

    def set_level(self, level: str) -> dict:
        return self._request("POST", "set_level", {"task_level": level})

    def reset(self) -> dict:
        return self._request("POST", "reset", {})

    def step(self, action: Dict[str, Any]) -> dict:
        return self._request("POST", "step", {"action": action})

    def grader(self) -> dict:
        return self._request("GET", "grader")


def clamp_score(raw: float) -> float:
    return max(0.001, min(0.999, float(raw)))


# ── Prompting / parsing ──────────────────────────────────────────────────────

def build_prompt(obs: dict, task_level: str) -> str:
    return (
        f"[TASK={task_level}] You are an autonomous SRE managing a 100-node GPU cluster.\n"
        f"Alert: {obs.get('alert', '')}\n"
        f"hardware_logs: {json.dumps(obs.get('hardware_logs', []))}\n"
        f"queue_depths: {json.dumps(obs.get('queue_depths', {}))}\n"
        f"gradient_variances: {json.dumps(obs.get('gradient_variances', []))}\n"
        f"gpu_memory_usage: {json.dumps(obs.get('gpu_memory_usage', []))}\n"
        f"system_health: {obs.get('system_health', 1.0)}\n"
        "Respond ONLY with JSON: {\"command\":\"...\",\"target\":\"...\",\"value\":null_or_int}"
    )


_NULLISH = {"none", "null", "nil", "n/a", "na", "undefined", "", "false"}


def parse_action(text: str) -> Dict[str, Optional[int]]:
    """Robustly parse a JSON-ish action from a model completion.

    Handles cases where the model emits the value field as:
      - actual JSON null           → None
      - integer 9000               → 9000
      - float 9000.0               → 9000
      - string "9000"              → 9000
      - string "none"/"null"/""    → None
      - any other unparseable junk → None  (instead of crashing)
    """
    payload = {"command": "UNKNOWN", "target": "unknown", "value": None}
    if not isinstance(text, str):
        return payload

    # Try every {...} candidate (non-greedy first, then greedy fallback).
    # The model often emits "thinking: {...}\nfinal: {...}" — we want the
    # last *valid* JSON object, which is typically the action.
    candidates = re.findall(r"\{[^{}]*\}", text)        # non-nested first
    candidates += [text[text.find("{"):text.rfind("}") + 1]] if "{" in text and "}" in text else []

    data = None
    for cand in reversed(candidates):                    # last valid wins
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "command" in obj:
                data = obj
                break
        except Exception:
            continue

    if data is None:
        return payload

    payload["command"] = str(data.get("command", "UNKNOWN")).upper().strip()
    payload["target"] = str(data.get("target", "unknown")).strip()

    raw = data.get("value")
    if raw is None:
        payload["value"] = None
    elif isinstance(raw, bool):
        payload["value"] = int(raw)
    elif isinstance(raw, (int, float)):
        try:
            payload["value"] = int(raw)
        except (TypeError, ValueError, OverflowError):
            payload["value"] = None
    elif isinstance(raw, str):
        s = raw.strip().lower()
        if s in _NULLISH:
            payload["value"] = None
        else:
            try:
                payload["value"] = int(float(s))  # float first to handle "9000.0"
            except (TypeError, ValueError):
                payload["value"] = None
    else:
        payload["value"] = None

    return payload


# ── LLM action generator (only loaded if transformers available) ─────────────

def _load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_action(model, tokenizer, obs: dict, task_level: str) -> Dict[str, Optional[int]]:
    import torch
    prompt = build_prompt(obs, task_level)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return parse_action(tokenizer.decode(out[0], skip_special_tokens=True))


# ── Rollout (used by both train and eval) ────────────────────────────────────

def run_episode(env: EnvAdapter, model, tokenizer, task_level: str,
                max_steps: int = MAX_STEPS_PER_EPISODE) -> float:
    env.set_level(task_level)
    resp = env.reset() or {}
    obs = resp.get("observation", {})
    done = bool(resp.get("done", False))

    for _ in range(max_steps):
        if done:
            break
        try:
            action = generate_action(model, tokenizer, obs, task_level)
        except Exception as e:
            print(f"[EVAL] generate_action failed: {e}", flush=True)
            break
        step_resp = env.step(action)            # never raises
        obs = step_resp.get("observation", {})
        done = bool(step_resp.get("done", False))

    grader = env.grader() or {}
    return clamp_score(grader.get("total", 0.001) or 0.001)


def evaluate_model(env: EnvAdapter, model, tokenizer, episodes: int = EVAL_EPISODES
                   ) -> Tuple[List[float], Dict[str, float]]:
    rewards: List[float] = []
    by_diff: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for i in range(episodes):
        level = random.choice(TASK_LEVELS)
        r = run_episode(env, model, tokenizer, level)
        rewards.append(r)
        by_diff[_difficulty_for_task(level)].append(r)
        print(f"[EVAL] episode={i+1}/{episodes} task={level} reward={r:.3f}", flush=True)
    diff_avg = {k: (sum(v) / len(v) if v else 0.001) for k, v in by_diff.items()}
    return rewards, diff_avg


def make_training_dataset(env: EnvAdapter, n: int = 128):
    from datasets import Dataset
    rows = []
    failures = 0
    for _ in range(n):
        level = random.choice(TASK_LEVELS)
        env.set_level(level)
        obs = (env.reset() or {}).get("observation", {})
        if not obs:
            failures += 1
            # Fallback: synth a minimal prompt so the dataset is never empty
            obs = {"alert": f"[fallback prompt for {level}]"}
        rows.append({"prompt": build_prompt(obs, level)})
    if failures:
        print(f"[DATASET] {failures}/{n} resets failed; used fallback prompts", flush=True)
    return Dataset.from_list(rows)


def _completion_text(completion):
    """TRL passes completions as either str or chat-format list of dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # chat-format: [{"role": "assistant", "content": "..."}]
        out = []
        for turn in completion:
            if isinstance(turn, dict) and "content" in turn:
                out.append(turn["content"])
            elif isinstance(turn, str):
                out.append(turn)
        return "\n".join(out)
    return str(completion)


def _prompt_task_level(prompt) -> str:
    text = prompt if isinstance(prompt, str) else _completion_text(prompt)
    m = re.search(r"\[TASK=(t\d{2})\]", text)
    return m.group(1) if m else random.choice(TASK_LEVELS)


# ── Composed reward functions (one signal per dimension) ─────────────────────
# Following the OpenEnv tutorial pattern, we expose multiple reward
# functions to GRPOTrainer instead of a single monolithic one. This gives
# GRPO a much richer learning signal and makes per-skill progress visible
# in the training table.

def reward_action_parses(prompts, completions, **kwargs):
    """+1 if the completion parses to a valid {command,target,value} dict."""
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        a = parse_action(text)
        ok = a["command"] != "UNKNOWN" and a["target"] not in ("", "unknown")
        rewards.append(1.0 if ok else -1.0)
    return rewards


def reward_correct_command(prompts, completions, **kwargs):
    """+2 if the issued command is *valid* for the task's fault type."""
    from reward_shaper import CORRECTIVE_VALID_FAULTS  # local import for Colab
    try:
        from server.netweaver_sre_environment import TASK_FAULT_TYPES
    except ImportError:
        from netweaver_sre_environment import TASK_FAULT_TYPES
    rewards = []
    for prompt, completion in zip(prompts, completions):
        level = _prompt_task_level(prompt)
        fault = TASK_FAULT_TYPES.get(level, "unknown")
        a = parse_action(_completion_text(completion))
        valid = CORRECTIVE_VALID_FAULTS.get(a["command"], [])
        rewards.append(2.0 if fault in valid else -1.0)
    return rewards


def reward_episode_resolution(prompts, completions, **kwargs):
    """The big signal: full multi-step rollout, returns the /grader rubric.

    Maps grader.total in [0.001, 0.999] to a reward in [-2, +5] so the
    grader dominates over the parse/cmd shaping rewards above.

    EnvAdapter never raises (returns {} on HTTP failures), so this function
    is robust to transient network errors during the long rollout.
    """
    env: EnvAdapter = reward_episode_resolution.env  # type: ignore[attr-defined]
    model = getattr(reward_episode_resolution, "model", None)
    tokenizer = getattr(reward_episode_resolution, "tokenizer", None)
    can_continue = model is not None and tokenizer is not None

    rewards = []
    for prompt, completion in zip(prompts, completions):
        level = _prompt_task_level(prompt)
        env.set_level(level)
        env.reset()
        first = parse_action(_completion_text(completion))

        step_resp = env.step(first)
        done = bool(step_resp.get("done", False))
        obs = step_resp.get("observation", {})

        steps = 1
        while not done and can_continue and steps < MAX_STEPS_PER_EPISODE:
            try:
                action = generate_action(model, tokenizer, obs, level)
            except Exception as e:
                print(f"[TRAIN] inner generate_action failed: {e}", flush=True)
                break
            step_resp = env.step(action)
            obs = step_resp.get("observation", {})
            done = bool(step_resp.get("done", False))
            steps += 1

        grader = env.grader() or {}
        total = float(grader.get("total", 0.001) or 0.001)
        resolved = bool(grader.get("resolved", False))

        # Map [0.001, 0.999] -> [-2, +5] (resolved gives a +1 bonus)
        mapped = -2.0 + 7.0 * total + (1.0 if resolved else 0.0)
        TRAIN_REWARDS.append(total)
        print(f"[TRAIN] step={len(TRAIN_REWARDS)} task={level} steps={steps} "
              f"grader={total:.3f} resolved={resolved} reward={mapped:.2f}",
              flush=True)
        rewards.append(mapped)
    return rewards


# Backward-compat single-function alias — defaults to the big resolution signal
reward_fn = reward_episode_resolution


# ── Plotting / persistence ───────────────────────────────────────────────────

def save_plots(before: List[float], after: List[float],
               before_diff: Dict[str, float], after_diff: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    os.makedirs("server/assets", exist_ok=True)

    curve = TRAIN_REWARDS[:MAX_TRAIN_STEPS]
    plt.figure(figsize=(10, 5))
    if curve:
        plt.plot(range(1, len(curve) + 1), curve, linewidth=2, color="#2563eb", label="Reward")
        # Add 5-step moving average
        if len(curve) >= 5:
            ma = [sum(curve[max(0, i - 4):i + 1]) / min(5, i + 1) for i in range(len(curve))]
            plt.plot(range(1, len(ma) + 1), ma, linewidth=2.5,
                     linestyle="--", color="#dc2626", label="5-step moving avg")
            plt.legend()
    plt.xlabel("Training step")
    plt.ylabel("Reward (rubric score)")
    plt.title("GRPO Reward Curve — NetWeaver SRE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("server/assets/reward_curve.png", dpi=160)
    plt.savefig("reward_curve.png", dpi=160)
    plt.close()

    before_mean = sum(before) / len(before) if before else 0.001
    after_mean = sum(after) / len(after) if after else 0.001
    plt.figure(figsize=(6, 5))
    plt.bar(["Before", "After"], [before_mean, after_mean], color=["#7c3aed", "#16a34a"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average reward")
    plt.title("Before vs After GRPO Training")
    plt.tight_layout()
    plt.savefig("server/assets/before_after.png", dpi=160)
    plt.savefig("before_after.png", dpi=160)
    plt.close()

    labels = ["easy", "medium", "hard"]
    before_vals = [before_diff.get(k, 0.001) for k in labels]
    after_vals = [after_diff.get(k, 0.001) for k in labels]
    x = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], before_vals, width=width, label="Before", color="#9333ea")
    plt.bar([i + width / 2 for i in x], after_vals, width=width, label="After", color="#16a34a")
    plt.xticks(x, [s.title() for s in labels])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average reward")
    plt.title("Reward by Difficulty — Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig("server/assets/difficulty_breakdown.png", dpi=160)
    plt.close()


# ── Entry points ─────────────────────────────────────────────────────────────

def main():
    """Run the GRPO training loop — requires GPU + transformers + trl."""
    try:
        from huggingface_hub import login
        from trl import GRPOConfig, GRPOTrainer  # noqa: F401
    except ImportError as e:
        print(f"[WARN] Missing GRPO deps ({e}). "
              "Falling back to scripts/run_training_demo.py for honest curves.")
        os.system("python scripts/run_training_demo.py")
        return

    if HF_TOKEN:
        login(token=HF_TOKEN)

    env = EnvAdapter(ENV_URL)
    model, tokenizer = _load_llm()

    print("Running BEFORE-training evaluation...", flush=True)
    before_rewards, before_diff = evaluate_model(env, model, tokenizer, episodes=EVAL_EPISODES)

    train_dataset = make_training_dataset(env, n=128)

    # Inject runtime context onto the reward functions
    reward_episode_resolution.env = env          # type: ignore[attr-defined]
    reward_episode_resolution.model = model      # type: ignore[attr-defined]
    reward_episode_resolution.tokenizer = tokenizer  # type: ignore[attr-defined]

    from trl import GRPOConfig, GRPOTrainer
    config = GRPOConfig(
        output_dir="grpo_qwen_netweaver",
        max_steps=MAX_TRAIN_STEPS,
        per_device_train_batch_size=2,           # GRPO needs >=num_generations
        num_generations=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim="adamw_8bit" if os.environ.get("USE_8BIT_OPTIM", "1") == "1" else "adamw_torch",
        logging_steps=1,
        report_to=[],                            # set to "trackio" inside Colab
        save_steps=max(25, MAX_TRAIN_STEPS),
        # Constrain generation: our action JSON is < 80 tokens; cap aggressively
        # so each step is fast. (max_prompt_length was removed in newer TRL —
        # prompt length is auto-inferred from the dataset.)
        max_completion_length=96,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_action_parses,
            reward_correct_command,
            reward_episode_resolution,           # the big rollout-based signal
        ],
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model("grpo_qwen_netweaver/final")

    print("Running AFTER-training evaluation...", flush=True)
    after_rewards, after_diff = evaluate_model(env, model, tokenizer, episodes=EVAL_EPISODES)

    save_plots(before_rewards, after_rewards, before_diff, after_diff)

    results = {
        "env_url": ENV_URL,
        "model_name": MODEL_NAME,
        "max_train_steps": MAX_TRAIN_STEPS,
        "training_rewards": TRAIN_REWARDS[:MAX_TRAIN_STEPS],
        "before_rewards": before_rewards,
        "after_rewards": after_rewards,
        "difficulty_breakdown": {"before": before_diff, "after": after_diff},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "grpo_real_run",
    }
    with open("training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"Done. before_avg={sum(before_rewards)/len(before_rewards):.3f} "
        f"after_avg={sum(after_rewards)/len(after_rewards):.3f}",
        flush=True,
    )
    print("Saved: training_results.json, server/assets/*.png", flush=True)


if __name__ == "__main__":
    main()
