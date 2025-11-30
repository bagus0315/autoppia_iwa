#!/usr/bin/env python3
"""Replay leaderboard solutions and save action indices for BC training."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.demo_webs.classes import UseCase, WebProject
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv

PROJECT_MAP: dict[str, WebProject] = {project.id: project for project in demo_web_projects}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset JSONL path with tasks and actions.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for BC traces.")
    parser.add_argument("--limit", type=int, help="Max number of tasks to process.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--topk", type=int, default=50, help="Number of top-k candidates.")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode.")
    return parser.parse_args()


def _match_use_case(project: WebProject, name: str | None) -> UseCase | None:
    if not project.use_cases or not name:
        return None
    upper = name.upper()
    for use_case in project.use_cases:
        if use_case.name.upper() == upper:
            return use_case
    return None


def _normalize_action_payload(raw: dict[str, Any]) -> dict[str, Any]:
    attrs = raw.get("attributes") or {}
    payload = {k: v for k, v in attrs.items() if k != "selector"}
    if "selector" in attrs:
        payload["selector"] = attrs["selector"]

    action_type = (raw.get("type") or "").strip().lower()
    if action_type == "input":
        if "keys" in attrs:
            payload = {"type": "SendKeysIWAAction", "keys": attrs["keys"]}
        else:
            payload.setdefault("type", "TypeAction")
    elif action_type == "navigate":
        payload.setdefault("type", "NavigateAction")
    elif action_type == "click":
        payload.setdefault("type", "ClickAction")
    elif action_type == "scroll":
        payload.setdefault("type", "ScrollAction")
    elif action_type == "wait":
        payload.setdefault("type", "WaitAction")
    elif action_type == "selectdropdownoption":
        payload.setdefault("type", "SelectDropDownOptionAction")
    elif action_type == "select":
        payload.setdefault("type", "SelectAction")
    else:
        payload.setdefault("type", action_type.capitalize() + "Action")
    return payload


def _build_actions(raw_actions: list[dict[str, Any]]) -> list[BaseAction]:
    actions: list[BaseAction] = []
    for idx, raw in enumerate(raw_actions):
        payload = _normalize_action_payload(raw)
        action = BaseAction.create_action(payload)
        if action is None:
            raise ValueError(f"Unsupported action payload at index {idx}: {raw}")
        actions.append(action)
    return actions


async def replay_single_task(
    project: WebProject,
    task_data: dict[str, Any],
    env_config: dict[str, Any],
    output_dir: Path,
) -> bool:
    """Replay a single task and save BC-compatible traces with action indices."""
    
    task_id = task_data.get("task_id")
    use_case_name = task_data.get("use_case")
    intent = task_data.get("intent") or task_data.get("prompt")
    start_url = task_data.get("start_url") or project.frontend_url
    
    use_case = _match_use_case(project, use_case_name)
    
    task = Task(
        id=task_id,
        prompt=intent or f"Execute {use_case_name or 'task'} on {project.name}",
        url=start_url,
        web_project_id=project.id,
        use_case=use_case,
        is_web_real=project.is_web_real,
        relevant_data=project.relevant_data or {},
    )
    object.__setattr__(task, "assign_seed", False)
    task = task.prepare_for_agent("bc_replay")
    
    # Parse actions
    try:
        actions = _build_actions(task_data.get("actions", []))
    except Exception as exc:
        logger.warning(f"Failed to parse actions for {task_id}: {exc}")
        return False
    
    if not actions:
        return False
    
    # Create output directory
    trace_dir = output_dir / project.id / task_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"bc_trace_{task_id}.jsonl"
    
    # Replay with environment
    env = None
    try:
        env = IWAWebEnv(cfg=env_config)
        obs, info = env.reset(options={"task": task})
        
        steps_data = []
        
        for action_idx, action in enumerate(actions):
            # Find which topk candidate matches this action
            action_index = _find_matching_topk_index(action, env, obs)
            
            if action_index is None:
                logger.debug(f"Could not match action {action.__class__.__name__} to topk for task {task_id} step {action_idx}")
                continue
            
            # Save step data
            step_data = {
                "task_id": task_id,
                "step": action_idx,
                "observation": {
                    "dom_ids": obs["dom_ids"].tolist(),
                    "goal_ids": obs["goal_ids"].tolist(),
                    "url_id": int(obs["url_id"]),
                    "prev_actions": obs["prev_actions"].tolist(),
                    "topk_meta": obs["topk_meta"].tolist(),
                    "topk_text_ids": obs["topk_text_ids"].tolist(),
                    "score": float(obs["score"]),
                },
                "action_index": action_index,
                "done": False,
                "truncated": False,
            }
            steps_data.append(step_data)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action_index)
            
            if done or truncated:
                steps_data[-1]["done"] = bool(done)
                steps_data[-1]["truncated"] = bool(truncated)
                break
        
        # Save all steps to file
        if len(steps_data) >= 3:  # Minimum 3 steps
            with open(trace_file, "w") as f:
                for step_data in steps_data:
                    f.write(json.dumps(step_data) + "\n")
            logger.info(f"✅ Saved {len(steps_data)} steps for task {task_id}")
            return True
        else:
            logger.warning(f"Too few steps ({len(steps_data)}) for task {task_id}")
            return False
            
    except Exception as exc:
        logger.error(f"Error replaying task {task_id}: {exc}")
        return False
    finally:
        if env:
            try:
                # Close without running event loop (we're in async context)
                if hasattr(env, '_evaluator') and env._evaluator:
                    if hasattr(env._evaluator, '_context') and env._evaluator._context:
                        await env._evaluator._context.close()
                    if hasattr(env._evaluator, '_browser') and env._evaluator._browser:
                        await env._evaluator._browser.close()
            except:
                pass  # Best effort cleanup


def _find_matching_topk_index(action: BaseAction, env: IWAWebEnv, obs: dict) -> int | None:
    """Find which topk candidate index matches the given action."""
    topk_meta = obs["topk_meta"]  # Shape: (topk, 8)
    
    action_type = action.__class__.__name__.lower().replace("action", "")
    
    # For NavigateAction, return a special index (we can use 0 or skip)
    if action_type == "navigate":
        return None  # Skip navigate actions
    
    best_idx = None
    best_similarity = 0.0
    
    for idx in range(len(topk_meta)):
        candidate = topk_meta[idx]
        similarity = _compute_similarity(action, candidate)
        
        if similarity > best_similarity and similarity >= 0.6:  # Threshold
            best_similarity = similarity
            best_idx = idx
    
    return best_idx


def _compute_similarity(action: BaseAction, candidate) -> float:
    """Compute similarity between action and topk candidate."""
    # candidate is numpy array: [role_id/8.0, clickable, focusable, editable, visible, text_length_norm, center_x_norm, center_y_norm]
    
    role_id = int(candidate[0] * 8.0)
    clickable = bool(candidate[1])
    focusable = bool(candidate[2])
    editable = bool(candidate[3])
    visible = bool(candidate[4])
    text_length_norm = candidate[5]
    center_x_norm = candidate[6]
    center_y_norm = candidate[7]
    
    # Approximate coordinates (assuming 1920x1080)
    candidate_x = int(center_x_norm * 1920)
    candidate_y = int(center_y_norm * 1080)
    
    action_type = action.__class__.__name__.lower().replace("action", "")
    
    similarity = 0.0
    
    # Type matching
    candidate_type = {
        1: "click",  # button
        2: "click",  # link
        3: "click",  # submit
        4: "type",   # textbox
    }.get(role_id, "")
    
    if action_type == candidate_type:
        similarity += 0.5
        
        # Position matching for click actions
        if "click" in action_type:
            if hasattr(action, "x") and hasattr(action, "y"):
                distance = ((action.x - candidate_x) ** 2 + (action.y - candidate_y) ** 2) ** 0.5
                if distance < 20:
                    similarity += 0.5
                elif distance < 50:
                    similarity += 0.3
                elif distance < 100:
                    similarity += 0.1
        
        # Text length matching for type actions
        elif "type" in action_type:
            if hasattr(action, "text"):
                if abs(len(action.text) - (text_length_norm * 64)) < 5:
                    similarity += 0.3
    
    return min(similarity, 1.0)


async def replay_project_async(
    project: WebProject,
    tasks: list[dict[str, Any]],
    env_config: dict[str, Any],
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> tuple[int, int]:
    """Replay all tasks for a project."""
    async with semaphore:
        success_count = 0
        total = len(tasks)
        
        for task_data in tqdm(tasks, desc=f"{project.id} replays", unit="task"):
            success = await replay_single_task(project, task_data, env_config, output_dir)
            if success:
                success_count += 1
        
        return total, success_count


async def main_async(args: argparse.Namespace):
    # Load tasks
    logger.info(f"Loading tasks from {args.dataset}")
    tasks_by_project = defaultdict(list)
    
    with open(args.dataset, "r") as f:
        count = 0
        for line in f:
            if args.limit and count >= args.limit:
                break
            task_data = json.loads(line)
            project_id = task_data.get("website")
            if project_id in PROJECT_MAP:
                tasks_by_project[project_id].append(task_data)
                count += 1
    
    logger.info(f"Loaded {count} tasks across {len(tasks_by_project)} projects")
    
    # Environment config
    env_config = {
        "topk": args.topk,
        "max_steps": args.max_steps,
        "goal_vocab_size": 4096,
        "dom_vocab_size": 8192,
        "url_vocab_size": 1024,
        "max_goal_tokens": 48,
        "max_dom_tokens": 200,
        "max_element_tokens": 12,
        "action_history": 10,
        "headless": True,
    }
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Replay with parallelism
    semaphore = asyncio.Semaphore(args.parallel)
    tasks_list = [
        (PROJECT_MAP[project_id], tasks, env_config, args.output_dir, semaphore)
        for project_id, tasks in tasks_by_project.items()
    ]
    
    results = []
    for project, tasks, env_cfg, out_dir, sem in tqdm(tasks_list, desc="Projects", unit="project"):
        result = await replay_project_async(project, tasks, env_cfg, out_dir, sem)
        results.append(result)
    
    total_tasks = sum(r[0] for r in results)
    total_success = sum(r[1] for r in results)
    
    logger.info(f"✅ Replay complete: {total_success}/{total_tasks} tasks successful")
    logger.info(f"📁 Traces saved to: {args.output_dir}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

