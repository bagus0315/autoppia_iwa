"""
Load tasks from the full dataset for PPO training diversity.
"""
import json
import random
from pathlib import Path
from typing import List, Optional

from loguru import logger

from autoppia_iwa.src.data_generation.tasks.classes import Task
from autoppia_iwa.src.demo_webs.classes import WebProject
from autoppia_iwa.src.demo_webs.config import demo_web_projects

PROJECT_MAP = {p.id: p for p in demo_web_projects}


class DatasetTaskLoader:
    """Load tasks from flattened dataset for diverse PPO training."""
    
    def __init__(self, dataset_path: str, project_filter: Optional[str] = None):
        """
        Args:
            dataset_path: Path to flattened JSONL dataset
            project_filter: Optional project ID to filter tasks (e.g., "autoppia_cinema")
        """
        self.dataset_path = Path(dataset_path)
        self.project_filter = project_filter
        self.tasks: List[Task] = []
        self._load_tasks()
    
    def _load_tasks(self):
        """Load all tasks from dataset."""
        logger.info(f"Loading tasks from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found: {self.dataset_path}")
            return
        
        count = 0
        skipped = 0
        
        with open(self.dataset_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    # Filter by project if specified
                    website = record.get("website")
                    if self.project_filter and website != self.project_filter:
                        skipped += 1
                        continue
                    
                    # Create task from record
                    task = self._create_task_from_record(record)
                    if task:
                        self.tasks.append(task)
                        count += 1
                    else:
                        skipped += 1
                        
                except Exception as e:
                    logger.debug(f"Failed to parse record: {e}")
                    skipped += 1
                    continue
        
        logger.info(f"Loaded {count} tasks (skipped {skipped})")
        if self.project_filter:
            logger.info(f"Filtered for project: {self.project_filter}")
    
    def _create_task_from_record(self, record: dict) -> Optional[Task]:
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
                logger.debug(f"Unknown project: {website}")
                return None
            
            task = Task(
                id=task_id,
                prompt=intent,
                url=start_url,  # Contains seed!
                web_project_id=website,
                use_case=record.get("use_case"),
                is_web_real=project.is_web_real,
                relevant_data={},
            )
            
            return task.prepare_for_agent("ppo_training")
            
        except Exception as e:
            logger.debug(f"Failed to create task: {e}")
            return None
    
    def sample_task(self) -> Optional[Task]:
        """Randomly sample a task."""
        if not self.tasks:
            logger.error("No tasks available")
            return None
        return random.choice(self.tasks)
    
    def get_all_tasks(self) -> List[Task]:
        """Get all loaded tasks."""
        return self.tasks
    
    def get_task_count(self) -> int:
        """Get number of loaded tasks."""
        return len(self.tasks)

