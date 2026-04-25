# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4.0",
#   "transformers>=4.44.0",
#   "trl>=0.11.0",
#   "datasets>=2.18.0",
#   "huggingface_hub>=0.24.0",
#   "openenv-core>=0.2.3",
#   "requests>=2.31.0",
#   "matplotlib>=3.8.0",
#   "pillow>=10.0.0",
#   "fastapi",
#   "uvicorn",
#   "pydantic>=2.0.0",
#   "accelerate>=0.31.0",
# ]
# ///
"""HF Jobs entrypoint for real GRPO training on the live NetWeaver SRE env.

Launch:
  hf jobs uv run --flavor t4-small -s HF_TOKEN \
      https://huggingface.co/spaces/Shasidharyadavr/netweaver_sre/raw/main/scripts/hf_job_train.py

This script:
  1. Clones the netweaver_sre HF Space repo (so we have train_grpo.py + rubrics.py)
  2. Trains Qwen/Qwen2.5-0.5B-Instruct with GRPO against the LIVE env at
     https://shasidharyadavr-netweaver-sre.hf.space (composed reward functions)
  3. Uploads training_results.json + the 5 plots back to the Space
     (replacing the heuristic placeholder data)

Total wall-clock on t4-small: ~15-25 minutes for 30 GRPO steps.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path


REPO_ID = "Shasidharyadavr/netweaver_sre"
WORKDIR = Path("/tmp/netweaver-sre-train")
ENV_URL = os.environ.get("ENV_URL", "https://shasidharyadavr-netweaver-sre.hf.space")

# Knobs (override via env)
os.environ.setdefault("MAX_TRAIN_STEPS", "30")
os.environ.setdefault("EVAL_EPISODES", "6")
os.environ.setdefault("MAX_STEPS_PER_EPISODE", "5")
os.environ.setdefault("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
os.environ["ENV_URL"] = ENV_URL
# Use adamw_torch (no bitsandbytes), more reliable across job containers
os.environ["USE_8BIT_OPTIM"] = "0"


def _run(cmd, **kw):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


# ── 1. Clone the Space repo ──────────────────────────────────────────────────
if WORKDIR.exists():
    shutil.rmtree(WORKDIR)
print(f"[1/4] Cloning {REPO_ID} into {WORKDIR}...", flush=True)
_run(["git", "clone", "--depth", "1",
      f"https://huggingface.co/spaces/{REPO_ID}", str(WORKDIR)])
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR))


# ── 2. Sanity-check the live env is reachable ────────────────────────────────
print(f"\n[2/4] Probing live env at {ENV_URL}...", flush=True)
import requests
for i in range(10):
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        if r.status_code == 200:
            print("  health:", r.json(), flush=True)
            break
    except Exception as e:
        print(f"  attempt {i+1}/10: {e}", flush=True)
        time.sleep(2)
else:
    raise RuntimeError(f"Live env at {ENV_URL} unreachable; aborting.")


# ── 3. Run GRPO training ─────────────────────────────────────────────────────
print(f"\n[3/4] Starting GRPO training...", flush=True)
print(f"  model           = {os.environ['MODEL_NAME']}", flush=True)
print(f"  max_train_steps = {os.environ['MAX_TRAIN_STEPS']}", flush=True)
print(f"  eval_episodes   = {os.environ['EVAL_EPISODES']}", flush=True)
print(f"  max_steps/ep    = {os.environ['MAX_STEPS_PER_EPISODE']}", flush=True)

import train_grpo  # noqa: E402

t0 = time.time()
train_grpo.main()
print(f"\n[3/4] Training done in {time.time() - t0:.1f}s", flush=True)


# ── 4. Upload results back to the Space ──────────────────────────────────────
print(f"\n[4/4] Uploading results to {REPO_ID}...", flush=True)
from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])

uploads = [
    "training_results.json",
    "server/assets/reward_curve.png",
    "server/assets/loss_curve.png",
    "server/assets/baseline_vs_trained.png",
    "server/assets/before_after.png",
    "server/assets/difficulty_breakdown.png",
]

uploaded = []
for rel in uploads:
    p = WORKDIR / rel
    if not p.exists():
        print(f"  SKIP (not found): {rel}", flush=True)
        continue
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=rel,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message=f"Real GRPO training run ({os.environ['MAX_TRAIN_STEPS']} steps, "
                       f"{os.environ['MODEL_NAME']})",
    )
    print(f"  uploaded: {rel}  ({p.stat().st_size} bytes)", flush=True)
    uploaded.append(rel)

# Echo the final summary
if (WORKDIR / "training_results.json").exists():
    with open(WORKDIR / "training_results.json") as f:
        results = json.load(f)
    bavg = sum(results["before_rewards"]) / max(1, len(results["before_rewards"]))
    aavg = sum(results["after_rewards"]) / max(1, len(results["after_rewards"]))
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"  source       : {results.get('source', '?')}")
    print(f"  before avg   : {bavg:.3f}")
    print(f"  after avg    : {aavg:.3f}")
    print(f"  delta        : {aavg - bavg:+.3f}")
    print(f"  uploaded {len(uploaded)} files to {REPO_ID}")
print("\nDone.")
