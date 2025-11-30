#!/usr/bin/env python3
"""Generate BC traces synchronously using rule-based agent."""

import argparse
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.demo_webs.classes import WebProject
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv
from autoppia_iwa.src.web_agents.rule_based.rule_based_agent import RuleBasedAgent

PROJECT_MAP = {p.id: p for p in demo_web_projects}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    return parser.parse_args()


def generate_trace_for_task(task_data: dict, env_config: dict, output_dir: Path) -> bool:
    """Generate a single BC trace using rule-based agent."""
    
    project_id = task_data.get("website")
    task_id = task_data.get("task_id")
    
    project = PROJECT_MAP.get(project_id)
    if not project:
        return False
    
    # Create task
    task = Task(
        id=task_id,
        prompt=task_data.get("intent") or task_data.get("prompt", ""),
        url=task_data.get("start_url") or project.frontend_url,
        web_project_id=project.id,
        use_case=None,
        is_web_real=project.is_web_real,
        relevant_data={},
    )
    task = task.prepare_for_agent("bc_gen")
    
    # Create environment and agent
    env = None
    try:
        env = IWAWebEnv(cfg=env_config)
        agent = RuleBasedAgent(env_config=env_config)
        
        obs, info = env.reset(options={"task": task})
        
        steps = []
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated) and step_count < env_config["max_steps"]:
            # Agent predicts action index
            action_index = agent.predict_action_index(obs)
            
            # Record step
            step_data = {
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
            steps.append(step_data)
            
            # Execute
            obs, reward, done, truncated, info = env.step(action_index)
            step_count += 1
        
        # Update last step
        if steps:
            steps[-1]["done"] = bool(done)
            steps[-1]["truncated"] = bool(truncated)
        
        # Save if enough steps
        if len(steps) >= 3:
            trace_dir = output_dir / project_id / task_id
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = trace_dir / f"bc_trace_{task_id}.jsonl"
            
            with open(trace_file, "w") as f:
                for step in steps:
                    f.write(json.dumps(step) + "\n")
            
            return True
        
        return False
        
    except Exception as exc:
        logger.error(f"Error generating trace for {task_id}: {exc}")
        return False
    finally:
        if env:
            try:
                env.close()
            except:
                pass


def main():
    args = parse_args()
    
    logger.info(f"Loading tasks from: {args.dataset}")
    tasks = []
    with open(args.dataset, "r") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            tasks.append(json.loads(line))
    
    logger.info(f"Loaded {len(tasks)} tasks")
    
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
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for task_data in tqdm(tasks, desc="Generating traces"):
        if generate_trace_for_task(task_data, env_config, args.output_dir):
            success_count += 1
    
    logger.info(f"✅ Generated {success_count}/{len(tasks)} traces")
    logger.info(f"📁 Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

