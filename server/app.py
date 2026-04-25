# server/app.py
# NetWeaver SRE — FastAPI environment server.
#
# This server is a THIN wrapper around `NetweaverSreEnvironment`. All fault
# scenario data, target randomisation, and step validation live in
# `server/netweaver_sre_environment.py`. The server adds:
#   - HTTP routing
#   - Per-episode state tracking for the rubric-based grader
#   - Optional shaping signal from `reward_shaper.compute_step_reward`

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Allow `from graders import ...` etc. when running as `uvicorn server.app:app`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graders import compute_grader_score, GRADER_CONFIG  # noqa: E402
from reward_shaper import (  # noqa: E402
    compute_step_reward,
    record_obs_fields,
    DESTRUCTIVE_COMMANDS,
)
from models import NetweaverSreAction, NetweaverSreGraderResponse  # noqa: E402

try:
    from server.netweaver_sre_environment import (  # type: ignore
        NetweaverSreEnvironment,
        set_task_level,
        clear_task_level,
        TASK_FAULT_TYPES,
        ALL_TASKS,
    )
except ImportError:
    from netweaver_sre_environment import (  # type: ignore
        NetweaverSreEnvironment,
        set_task_level,
        clear_task_level,
        TASK_FAULT_TYPES,
        ALL_TASKS,
    )

app = FastAPI(title="NetWeaver SRE", version="3.0.0")

# ── Static files (playground UI + plot images) ───────────────────────────────
_assets_dir = os.path.join(os.path.dirname(__file__), "assets")
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

# ── Singleton environment + per-episode state ────────────────────────────────
ENV = NetweaverSreEnvironment()

SESSION: Dict[str, Any] = {}


def _reset_session():
    SESSION.clear()
    SESSION["actions"] = []                       # list of {command,target,value}
    SESSION["obs_fields_seen"] = set()            # union of fields with data
    SESSION["rewarded_set"] = set()               # for compute_step_reward
    SESSION["action_history"] = set()             # "CMD:target" pairs
    SESSION["error_count"] = 0
    SESSION["had_fatal"] = False
    SESSION["cumulative_reward"] = 0.0
    SESSION["step_count"] = 0
    SESSION["done"] = False
    SESSION["last_grader"] = None
    SESSION["task_id"] = None
    SESSION["fault_type"] = None
    SESSION["task_level"] = None


_reset_session()


# ── Request/response models ──────────────────────────────────────────────────

class SetLevelRequest(BaseModel):
    task_level: str


class ActionPayload(BaseModel):
    command: str
    target: str
    value: Optional[int] = None


class StepRequest(BaseModel):
    action: ActionPayload


# ── Helpers ──────────────────────────────────────────────────────────────────

def _obs_to_dict(obs, *, step_count: int, reward: float, done: bool) -> dict:
    """Convert NetweaverSreObservation to a JSON dict + extras."""
    return {
        "alert": obs.alert,
        "hardware_logs": list(obs.hardware_logs),
        "queue_depths": dict(obs.queue_depths),
        "gradient_variances": list(obs.gradient_variances),
        "gpu_memory_usage": list(obs.gpu_memory_usage),
        "system_health": float(obs.system_health),
        "step_count": int(step_count),
        "reward": round(float(reward), 4),
        "done": bool(done),
        "active_connections": 100 + (step_count * 3 % 40),
        "error_rate": round(SESSION.get("error_count", 0) / max(1, step_count), 3),
    }


def _episode_state() -> Dict[str, Any]:
    return {
        "actions": list(SESSION.get("actions", [])),
        "steps": int(SESSION.get("step_count", 0)),
        "obs_fields_seen": set(SESSION.get("obs_fields_seen", set())),
        "had_fatal": bool(SESSION.get("had_fatal", False)),
        "error_count": int(SESSION.get("error_count", 0)),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/tasks")
def list_tasks():
    out = []
    for tid in ALL_TASKS:
        idx = int(tid[1:])
        difficulty = "Easy" if idx <= 7 else "Medium" if idx <= 14 else "Hard"
        out.append(
            {
                "id": f"netweaver_sre_{tid}",
                "level": tid,
                "fault_type": TASK_FAULT_TYPES[tid],
                "difficulty": difficulty,
            }
        )
    return {"tasks": out}


@app.post("/set_level")
def set_level(req: SetLevelRequest):
    set_task_level(req.task_level)
    return {"status": "ok", "task_level": req.task_level}


@app.post("/clear_level")
def clear_level():
    clear_task_level()
    return {"status": "ok"}


@app.post("/reset")
def reset(body: dict = {}):
    _reset_session()
    explicit = (body or {}).get("task_level") if isinstance(body, dict) else None
    obs = ENV.reset(task_level=explicit) if explicit else ENV.reset()

    SESSION["task_id"] = ENV.task_id
    SESSION["fault_type"] = ENV.fault_type
    SESSION["task_level"] = ENV.active_task
    SESSION["step_count"] = obs.step_count

    # Record which obs fields are populated this turn so the grader can credit
    # the agent for *available* diagnostic signals it could read.
    SESSION["obs_fields_seen"].update(record_obs_fields(_obs_dict_for_record(obs)))

    out = _obs_to_dict(obs, step_count=obs.step_count, reward=obs.reward, done=obs.done)
    return {
        "observation": out,
        "done": False,
        "reward": round(float(obs.reward), 4),
        "task_id": ENV.task_id,
        "task_level": ENV.active_task,
        "fault_type": ENV.fault_type,
    }


def _obs_dict_for_record(obs) -> dict:
    return {
        "hardware_logs": list(obs.hardware_logs),
        "queue_depths": dict(obs.queue_depths),
        "gradient_variances": list(obs.gradient_variances),
        "gpu_memory_usage": list(obs.gpu_memory_usage),
        "system_health": float(obs.system_health),
    }


@app.post("/step")
def step(req: StepRequest):
    if SESSION.get("done"):
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    if SESSION.get("task_id") is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")

    cmd = (req.action.command or "").upper()
    tgt = req.action.target or ""
    val = req.action.value

    # 1. Delegate to the environment for ground-truth validation
    obs = ENV.step(NetweaverSreAction(command=cmd, target=tgt, value=val))

    SESSION["step_count"] = obs.step_count

    # 2. Track destructive command usage (per spec)
    if cmd in DESTRUCTIVE_COMMANDS:
        SESSION["had_fatal"] = True

    # 3. Record action history for the rubric grader
    SESSION["actions"].append({"command": cmd, "target": tgt, "value": val})

    # 4. Update obs-fields-seen set with whatever data was visible this step
    SESSION["obs_fields_seen"].update(record_obs_fields(_obs_dict_for_record(obs)))

    # 5. Compute the optional shaping bonus (diagnostic awareness, dup penalty)
    fault_type = SESSION.get("fault_type") or "unknown"
    shaping, shaping_done = compute_step_reward(
        cmd, tgt, val, fault_type,
        SESSION["rewarded_set"],
        SESSION["action_history"],
        obs_fields_present=SESSION["obs_fields_seen"],
    )

    # 6. Combine ground-truth env reward with shaping (env is dominant)
    raw_reward = float(obs.reward) + 0.20 * float(shaping)
    step_reward = max(0.001, min(0.999, raw_reward))

    # 7. Track errors using the env's signal (env returns ~0.05 for wrong cmd)
    if obs.reward <= 0.06 and not obs.done:
        SESSION["error_count"] = SESSION.get("error_count", 0) + 1

    SESSION["cumulative_reward"] = max(
        0.001, min(0.999, float(SESSION.get("cumulative_reward", 0.0)) + step_reward * 0.1)
    )

    # 8. Episode termination handling
    episode_done = bool(obs.done) or bool(shaping_done)
    SESSION["done"] = episode_done

    grader_result = None
    if episode_done:
        grader_result = compute_grader_score(SESSION["task_id"], _episode_state())
        # Blend grader (rubric) with env outcome — both inform the final reward
        env_outcome = float(obs.reward)
        rubric_total = float(grader_result["total"])
        final_reward = max(0.001, min(0.999, 0.5 * env_outcome + 0.5 * rubric_total))
        SESSION["last_grader"] = grader_result
    else:
        final_reward = step_reward

    out = _obs_to_dict(obs, step_count=obs.step_count, reward=final_reward, done=episode_done)
    if episode_done and grader_result is not None:
        out["grader_score"] = grader_result["total"]
        out["grader_breakdown"] = grader_result["breakdown"]
        out["resolved"] = grader_result["resolved"]

    return {
        "observation": out,
        "done": episode_done,
        "reward": round(float(final_reward), 4),
        "shaped_reward": round(max(0.001, min(0.999, step_reward)), 4),
        "env_reward": round(float(obs.reward), 4),
        "grader": grader_result,
    }


@app.get("/grader", response_model=NetweaverSreGraderResponse)
def grader():
    """Return the grader breakdown for the current/last episode."""
    last = SESSION.get("last_grader")
    if last is None:
        if not SESSION.get("task_id"):
            return {
                "resolved": False,
                "total": 0.001,
                "breakdown": {"diagnosis": 0.0, "resolution": 0.0, "best_practice": 0.0},
            }
        last = compute_grader_score(SESSION["task_id"], _episode_state())
    return last


@app.get("/grader/{task_id}", response_model=NetweaverSreGraderResponse)
def grader_for_task(task_id: str):
    if SESSION.get("task_id") == task_id and SESSION.get("last_grader") is not None:
        return SESSION["last_grader"]
    return compute_grader_score(task_id, _episode_state())


@app.get("/state")
def state():
    return {
        "task_id": SESSION.get("task_id"),
        "task_level": SESSION.get("task_level"),
        "fault_type": SESSION.get("fault_type"),
        "step_count": SESSION.get("step_count", 0),
        "done": SESSION.get("done", False),
        "cumulative_reward": round(float(SESSION.get("cumulative_reward", 0.001)), 4),
        "error_count": SESSION.get("error_count", 0),
        "had_fatal": SESSION.get("had_fatal", False),
        "actions_taken": SESSION.get("actions", []),
        "episode_id": ENV.state.episode_id,
    }


@app.get("/", response_class=HTMLResponse)
def playground():
    html_path = os.path.join(os.path.dirname(__file__), "playground.html")
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>NetWeaver SRE</h1><p>Playground UI not found.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
