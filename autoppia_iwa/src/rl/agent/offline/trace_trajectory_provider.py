#!/usr/bin/env python3
"""
Trajectory provider that converts evaluation traces to BC training format.

This provider replays trace files through IWAWebEnv to capture the internal
observations (goal_ids, dom_ids, etc.) needed for behavioral cloning training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.demo_webs.classes import WebProject
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv
from autoppia_iwa.src.rl.agent.offline.interfaces import StepRecord, Trajectory


PROJECT_MAP = {p.id: p for p in demo_web_projects}


class TraceTrajectoryProvider:
    """
    Converts evaluation traces to BC training trajectories by replaying them
    through IWAWebEnv to capture internal observations.
    """

    def __init__(
        self,
        traces_dir: Path,
        env_config: dict | None = None,
        max_trajectories: int | None = None,
    ):
        """
        Args:
            traces_dir: Directory containing trace JSONL files organized by project
            env_config: Configuration for IWAWebEnv
            max_trajectories: Maximum number of trajectories to load
        """
        self.traces_dir = Path(traces_dir)
        self.env_config = env_config or self._default_env_config()
        self.max_trajectories = max_trajectories

    @staticmethod
    def _default_env_config() -> dict:
        """Default environment configuration for BC data collection."""
        return {
            "topk": 12,
            "max_steps": 50,
            "goal_vocab_size": 4096,
            "dom_vocab_size": 8192,
            "url_vocab_size": 1024,
            "max_goal_tokens": 48,
            "max_dom_tokens": 200,
            "max_element_tokens": 12,
            "action_history": 10,
        }

    def fetch(self, limit: int | None = None) -> Iterable[Trajectory]:
        """
        Yield trajectories by replaying traces through the environment.
        
        Args:
            limit: Maximum number of trajectories to return
        """
        count = 0
        max_count = min(limit or float('inf'), self.max_trajectories or float('inf'))

        # Iterate through all project directories
        for project_dir in sorted(self.traces_dir.iterdir()):
            if not project_dir.is_dir():
                continue

            project_id = project_dir.name
            project = PROJECT_MAP.get(project_id)
            
            if project is None:
                logger.warning(f"Unknown project: {project_id}, skipping")
                continue

            # Iterate through task directories in this project
            # Structure is: project_id/task_id/*.jsonl
            for task_dir in sorted(project_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                    
                # Find trace files in task directory
                trace_files = list(task_dir.glob("*.jsonl"))
                if not trace_files:
                    continue
                    
                # Use first trace file (usually there's only one per task)
                trace_file = trace_files[0]
                
                if count >= max_count:
                    logger.info(f"Reached trajectory limit: {max_count}")
                    return

                try:
                    trajectory = self._convert_trace_to_trajectory(
                        trace_file, project, project_id
                    )
                    if trajectory and len(trajectory.steps) > 0:
                        yield trajectory
                        count += 1
                        
                        if count % 100 == 0:
                            logger.info(f"Converted {count} trajectories...")
                            
                except Exception as e:
                    logger.debug(
                        f"Failed to convert {trace_file.name}: {e}"
                    )
                    continue

        logger.info(f"Conversion complete: {count} trajectories")

    def _convert_trace_to_trajectory(
        self,
        trace_file: Path,
        project: WebProject,
        project_id: str,
    ) -> Trajectory | None:
        """
        Convert a single trace file to a Trajectory by replaying through env.
        
        Args:
            trace_file: Path to trace JSONL file
            project: WebProject for this trace
            project_id: Project identifier
            
        Returns:
            Trajectory with StepRecords, or None if conversion fails
        """
        # Read trace file
        trace_data = self._load_trace(trace_file)
        if not trace_data:
            return None

        # Extract task and actions from trace
        task = self._extract_task_from_trace(trace_data, project, project_id)
        if not task:
            return None

        actions = self._extract_actions_from_trace(trace_data)
        if not actions:
            return None

        # Create environment and replay to capture observations
        env = IWAWebEnv(self.env_config)
        step_records = []

        try:
            # Reset environment with task
            obs, info = env.reset(options={"task": task})
            
            # Replay each action and capture observations
            for action in actions:
                # Get action mask before step
                action_mask = env.get_action_mask()
                
                # Find action index in topk candidates
                action_idx = self._map_action_to_index(action, env, info)
                
                if action_idx is None:
                    logger.debug(
                        f"Skipping action {action.action_type} - not in topk"
                    )
                    continue

                # Create StepRecord before taking action
                step_record = StepRecord(
                    goal_ids=obs["goal_ids"].copy(),
                    dom_ids=obs["dom_ids"].copy(),
                    url_id=obs["url_id"].copy(),
                    prev_actions=obs["prev_actions"].copy(),
                    topk_text_ids=obs["topk_text_ids"].copy(),
                    topk_meta=obs["topk_meta"].copy(),
                    action_mask=action_mask,
                    action_index=action_idx,
                    score=obs.get("score"),
                )
                step_records.append(step_record)

                # Take action in environment
                obs, reward, done, truncated, info = env.step(action_idx)
                
                if done or truncated:
                    break

        except Exception as e:
            logger.warning(f"Error replaying {trace_file.name}: {e}")
            return None
        finally:
            env.close()

        if not step_records:
            return None

        return Trajectory(steps=step_records)

    def _load_trace(self, trace_file: Path) -> list[dict] | None:
        """Load trace JSONL file."""
        try:
            trace_data = []
            with trace_file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trace_data.append(json.loads(line))
            return trace_data if trace_data else None
        except Exception as e:
            logger.warning(f"Failed to load {trace_file}: {e}")
            return None

    def _extract_task_from_trace(
        self,
        trace_data: list[dict],
        project: WebProject,
        project_id: str,
    ) -> Task | None:
        """Extract Task from trace metadata."""
        # The first record should contain task metadata
        if not trace_data:
            return None

        first_record = trace_data[0]
        task_id = first_record.get("task_id", "unknown")
        
        # Try to extract prompt/goal from trace
        prompt = first_record.get("goal") or first_record.get("prompt") or "Unknown task"
        start_url = first_record.get("url") or project.frontend_url

        task = Task(
            id=task_id,
            prompt=prompt,
            url=start_url,
            web_project_id=project_id,
            use_case=None,
            is_web_real=project.is_web_real,
            relevant_data={},
        )
        
        # Prepare task for agent (assigns seed if needed)
        return task.prepare_for_agent("bc_training")

    def _extract_actions_from_trace(self, trace_data: list[dict]) -> list[BaseAction] | None:
        """Extract actions from trace data."""
        actions = []
        
        for record in trace_data:
            action_type = record.get("action_type") or record.get("type")
            if not action_type:
                continue

            # Try to reconstruct action from trace record
            try:
                action = self._reconstruct_action(record)
                if action:
                    actions.append(action)
            except Exception as e:
                logger.debug(f"Failed to reconstruct action: {e}")
                continue

        return actions if actions else None

    def _reconstruct_action(self, record: dict) -> BaseAction | None:
        """Reconstruct BaseAction from trace record."""
        # The action is already in the record as a dict
        action_dict = record.get("action")
        if not action_dict:
            return None
            
        # Action dict already contains type, selector, url, text, etc.
        # Just pass it to create_action
        return BaseAction.create_action(action_dict)

    def _map_action_to_index(
        self,
        action: BaseAction,
        env: IWAWebEnv,
        info: dict,
    ) -> int | None:
        """Map action to index in environment's topk candidates."""
        # This is a simplified mapping - in reality, we'd need to match
        # the action to one of the topk candidates
        # For now, we'll use a heuristic approach
        
        topk = info.get("topk", [])
        
        # Try to find matching action in topk
        for idx, candidate in enumerate(topk):
            if self._actions_match(action, candidate):
                return idx

        # If no match found, return None (will skip this step)
        return None

    def _actions_match(self, action: BaseAction, candidate: dict) -> bool:
        """Check if action matches a candidate."""
        # Simplified matching - could be improved
        action_type = action.action_type.lower()
        candidate_type = candidate.get("type", "").lower()
        
        return action_type in candidate_type or candidate_type in action_type

