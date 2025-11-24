#!/usr/bin/env python3
"""
Trajectory provider that converts flattened dataset + trace files to BC format.

This provider reads from flattened JSONL datasets (which have task + solution info)
and matches them with trace files (which have DOM snapshots) to create complete
training trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.demo_webs.classes import WebProject
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.execution.actions.base import BaseAction
from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv
from autoppia_iwa.src.rl.agent.offline.interfaces import StepRecord, Trajectory, TrajectoryProvider
from autoppia_iwa.src.rl.agent.offline.action_converter import convert_leaderboard_action_to_base_action_dict

PROJECT_MAP = {p.id: p for p in demo_web_projects}


class DatasetTrajectoryProvider(TrajectoryProvider):
    """
    Converts flattened dataset + traces to BC trajectories by replaying
    actions through IWAWebEnv to capture observations.
    """

    def __init__(
        self,
        dataset_paths: list[Path],
        traces_root: Path,
        env_config: dict | None = None,
        max_trajectories: int | None = None,
        min_steps: int = 3,
    ):
        """
        Args:
            dataset_paths: List of paths to flattened JSONL dataset files
            traces_root: Root directory containing raw traces (project_id/task_id/*.jsonl)
            env_config: Configuration for IWAWebEnv
            max_trajectories: Maximum number of trajectories to load
            min_steps: Minimum steps required for a valid trajectory
        """
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.traces_root = Path(traces_root)
        self.env_config = env_config or self._default_env_config()
        self.max_trajectories = max_trajectories
        self.min_steps = min_steps

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
        Yield trajectories by reading dataset and replaying through environment.
        
        Args:
            limit: Maximum number of trajectories to return
        """
        count = 0
        skipped = 0
        max_count = min(limit or float('inf'), self.max_trajectories or float('inf'))

        # Read all dataset files
        for dataset_path in self.dataset_paths:
            logger.info(f"Loading dataset: {dataset_path}")
            
            with dataset_path.open("r") as f:
                for line_num, line in enumerate(f, 1):
                    if count >= max_count:
                        logger.info(f"Reached trajectory limit: {max_count}")
                        return

                    try:
                        record = json.loads(line)
                        
                        # Extract task info
                        task = self._create_task_from_record(record)
                        if not task:
                            continue
                        
                        # Extract actions
                        actions = self._extract_actions_from_record(record)
                        if not actions or len(actions) < self.min_steps:
                            skipped += 1
                            continue

                        # Convert to trajectory by replaying in environment
                        trajectory = self._replay_to_trajectory(task, actions)
                        
                        if trajectory and len(trajectory.steps) >= self.min_steps:
                            yield trajectory
                            count += 1
                            
                            if count % 100 == 0:
                                logger.info(f"Converted {count} trajectories (skipped {skipped})...")
                        else:
                            skipped += 1
                            
                    except Exception as e:
                        logger.debug(f"Failed to process record {line_num}: {e}")
                        skipped += 1
                        continue

        logger.info(f"Conversion complete: {count} trajectories (skipped {skipped})")

    def _create_task_from_record(self, record: dict) -> Task | None:
        """Create Task object from dataset record."""
        try:
            task_id = record.get("task_id")
            website = record.get("website")
            intent = record.get("intent")
            start_url = record.get("start_url")
            
            if not all([task_id, website, intent, start_url]):
                return None

            # Get project
            project = PROJECT_MAP.get(website)
            if not project:
                logger.warning(f"Unknown project: {website}")
                return None

            task = Task(
                id=task_id,
                prompt=intent,
                url=start_url,
                web_project_id=website,
                use_case=record.get("use_case"),
                is_web_real=project.is_web_real,
                relevant_data={},
            )
            
            return task.prepare_for_agent("bc_training")
            
        except Exception as e:
            logger.debug(f"Failed to create task: {e}")
            return None

    def _extract_actions_from_record(self, record: dict) -> list[BaseAction] | None:
        """Extract actions from dataset record."""
        try:
            actions_data = record.get("actions", [])
            if not actions_data:
                return None

            actions = []
            for action_data in actions_data:
                try:
                    # Convert leaderboard format to BaseAction format
                    converted_dict = convert_leaderboard_action_to_base_action_dict(action_data)
                    if not converted_dict:
                        continue
                    
                    # Create BaseAction from converted dict
                    action = BaseAction.create_action(converted_dict)
                    if action:
                        actions.append(action)
                except Exception as e:
                    logger.debug(f"Failed to create action: {e}")
                    continue

            return actions if actions else None
            
        except Exception as e:
            logger.debug(f"Failed to extract actions: {e}")
            return None

    def _replay_to_trajectory(
        self,
        task: Task,
        actions: list[BaseAction],
    ) -> Trajectory | None:
        """
        Replay actions in environment to capture observations.
        
        Args:
            task: Task object
            actions: List of expert actions to replay
            
        Returns:
            Trajectory with StepRecords, or None if replay fails
        """
        env = IWAWebEnv(self.env_config)
        step_records = []

        try:
            # Reset environment with task
            obs, info = env.reset(options={"task": task})
            
            # Replay each action
            for action_idx, action in enumerate(actions):
                # Get action mask before step
                action_mask = env.get_action_mask()
                
                # Map expert action to environment action index
                env_action_idx = self._map_action_to_topk(action, env, obs)
                
                if env_action_idx is None:
                    # Expert action not in topk - this is expected sometimes
                    # For BC, we skip steps where expert action isn't available
                    action_type_name = action.__class__.__name__
                    logger.debug(f"Skipping action {action_type_name} - not in topk")
                    continue

                # Create StepRecord with current observation + expert action
                step_record = StepRecord(
                    goal_ids=obs["goal_ids"].copy(),
                    dom_ids=obs["dom_ids"].copy(),
                    url_id=obs["url_id"].copy(),
                    prev_actions=obs["prev_actions"].copy(),
                    topk_text_ids=obs["topk_text_ids"].copy(),
                    topk_meta=obs["topk_meta"].copy(),
                    action_mask=action_mask,
                    action_index=env_action_idx,
                    score=np.array([1.0], dtype=np.float32),  # Expert actions are "good"
                )
                step_records.append(step_record)

                # Take action in environment
                obs, reward, done, truncated, info = env.step(env_action_idx)
                
                # Convert numpy arrays to Python booleans
                done_bool = bool(done) if hasattr(done, '__bool__') or hasattr(done, 'item') else done
                truncated_bool = bool(truncated) if hasattr(truncated, '__bool__') or hasattr(truncated, 'item') else truncated
                
                if done_bool or truncated_bool:
                    break

        except Exception as e:
            import traceback
            logger.warning(f"Error replaying trajectory: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            try:
                env.close()
            except:
                pass

        if not step_records:
            return None

        return Trajectory(steps=step_records)

    def _map_action_to_topk(
        self,
        expert_action: BaseAction,
        env: IWAWebEnv,
        obs: dict,
    ) -> int | None:
        """
        Map expert action to topk candidate using similarity matching.
        
        Returns action index if similarity above threshold, None otherwise.
        """
        # Get topk candidates from observation
        topk_candidates = obs.get("topk_meta", [])
        if topk_candidates is None or len(topk_candidates) == 0:
            return None
        
        # Find best matching candidate using similarity
        best_idx = None
        best_similarity = 0.0
        
        try:
            for idx, candidate in enumerate(topk_candidates):
                try:
                    similarity = self._compute_action_similarity(expert_action, candidate)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = idx
                except Exception as e:
                    logger.debug(f"Error computing similarity for candidate {idx}: {e}")
                    continue
            
            # Similarity threshold: accept if ≥60% match
            SIMILARITY_THRESHOLD = 0.6
            
            if best_similarity >= SIMILARITY_THRESHOLD:
                logger.debug(
                    f"Matched {expert_action.__class__.__name__} with "
                    f"similarity {best_similarity:.2f} (idx={best_idx})"
                )
                return best_idx
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in _map_action_to_topk: {e}")
            return None

    def _compute_action_similarity(self, action: BaseAction, candidate_meta: np.ndarray) -> float:
        """
        Compute similarity score between expert action and candidate.
        
        Args:
            action: Expert action (BaseAction instance)
            candidate_meta: Numpy array with 8 features:
                [0] role_id / 8.0 (button=1, link=2, submit=3, textbox=4, other=0)
                [1] clickable (0.0 or 1.0)
                [2] focusable (0.0 or 1.0)
                [3] editable (0.0 or 1.0)
                [4] visible (0.0 or 1.0)
                [5] text_length (normalized 0.0 to 1.0)
                [6] center_x (normalized 0.0 to 1.0)
                [7] center_y (normalized 0.0 to 1.0)
        
        Returns: 0.0 (no match) to 1.0 (perfect match)
        """
        try:
            # Decode candidate features
            role_id = int(candidate_meta[0] * 8.0)  # Denormalize role
            is_clickable = bool(candidate_meta[1] > 0.5)
            is_focusable = bool(candidate_meta[2] > 0.5)
            is_editable = bool(candidate_meta[3] > 0.5)
            is_visible = bool(candidate_meta[4] > 0.5)
            text_len_norm = float(candidate_meta[5])
            center_x_norm = float(candidate_meta[6])
            center_y_norm = float(candidate_meta[7])
            
            # Convert normalized coordinates to pixels (assuming 1920x1080)
            center_x_px = center_x_norm * 1920.0
            center_y_px = center_y_norm * 1080.0
            
            similarity = 0.0
            action_type = action.__class__.__name__
            
            # CLICK ACTIONS: Match clickable elements by position
            if action_type == "ClickAction":
                if is_clickable and is_visible:
                    similarity += 0.4  # Base score for clickable + visible
                    
                    # Check position proximity
                    if hasattr(action, "x") and hasattr(action, "y"):
                        if action.x is not None and action.y is not None:
                            # Euclidean distance
                            distance = ((action.x - center_x_px) ** 2 + (action.y - center_y_px) ** 2) ** 0.5
                            
                            # Very close (within 20 pixels): +0.5
                            if distance < 20:
                                similarity += 0.5
                            # Close (within 50 pixels): +0.3
                            elif distance < 50:
                                similarity += 0.3
                            # Moderately close (within 100 pixels): +0.1
                            elif distance < 100:
                                similarity += 0.1
                    
                    # Bonus for button/link role
                    if role_id in (1, 2):  # button or link
                        similarity += 0.1
                    
            # TYPE/INPUT ACTIONS: Match editable + focusable elements
            elif action_type == "TypeAction":
                if is_editable and is_focusable and is_visible:
                    similarity += 0.6  # Base score for textbox
                    
                    # Bonus for textbox role
                    if role_id == 4:  # textbox
                        similarity += 0.3
                    
                    # If expert action has coordinates, check position too
                    if hasattr(action, "x") and hasattr(action, "y"):
                        if action.x is not None and action.y is not None:
                            distance = ((action.x - center_x_px) ** 2 + (action.y - center_y_px) ** 2) ** 0.5
                            if distance < 50:
                                similarity += 0.1
                                
            # SELECT ACTIONS: Match editable elements (dropdowns)
            elif action_type == "SelectAction":
                if is_editable and is_focusable and is_visible:
                    similarity += 0.7  # Base score for select element
                    
                    # Position check if available
                    if hasattr(action, "x") and hasattr(action, "y"):
                        if action.x is not None and action.y is not None:
                            distance = ((action.x - center_x_px) ** 2 + (action.y - center_y_px) ** 2) ** 0.5
                            if distance < 50:
                                similarity += 0.2
            
            # SCROLL ACTIONS: Hard to match from metadata alone
            # Just give a small base score for visible elements
            elif action_type == "ScrollAction":
                if is_visible:
                    similarity += 0.3
            
            # NAVIGATE/WAIT ACTIONS: Cannot match from element metadata
            # These don't correspond to DOM elements
            elif action_type in ("NavigateAction", "WaitAction"):
                # These actions don't have corresponding topk candidates
                similarity = 0.0
            
            return min(similarity, 1.0)  # Cap at 1.0
            
        except Exception as e:
            # Catch array comparison errors and other issues
            logger.debug(f"Error computing similarity for {action.__class__.__name__}: {e}")
            return 0.0
    