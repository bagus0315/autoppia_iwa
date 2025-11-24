#!/usr/bin/env python3
"""
Fetch tasks with solutions from the new paginated API endpoint.
Handles the new response format: {success: true, data: {tasks: [...], total: ...}}
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger


def fetch_all_tasks(
    base_url: str,
    api_key: str,
    limit_per_page: int = 500,
    max_tasks: int | None = None,
    delay_seconds: float = 0.5,
) -> list[dict]:
    """
    Fetch all tasks from the paginated API endpoint.
    
    Args:
        base_url: Base URL (e.g., https://api-leaderboard.autoppia.com)
        api_key: API key for authentication
        limit_per_page: Tasks per page (default: 500)
        max_tasks: Maximum tasks to fetch (default: all)
        delay_seconds: Delay between requests to avoid rate limiting
        
    Returns:
        List of task records (each with task, solution, evaluation, agentRun)
    """
    all_tasks = []
    page = 1
    total_available = None
    
    while True:
        # Build request
        url = f"{base_url}/api/v1/tasks/with-solutions"
        params = {
            "key": api_key,
            "limit": limit_per_page,
            "page": page,
        }
        
        logger.info(f"Fetching page {page} (limit {limit_per_page})...")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Handle new API format
            if not data.get("success"):
                logger.error(f"API returned success=false: {data}")
                break
            
            api_data = data.get("data", {})
            tasks = api_data.get("tasks", [])
            
            # Get total count on first page
            if total_available is None:
                total_available = api_data.get("total", 0)
                logger.info(f"Total tasks available: {total_available:,}")
            
            if not tasks:
                logger.info("No more tasks returned, pagination complete")
                break
            
            # Add tasks to collection
            all_tasks.extend(tasks)
            logger.info(
                f"✓ Page {page}: fetched {len(tasks)} tasks "
                f"(total so far: {len(all_tasks):,}/{total_available:,})"
            )
            
            # Check if we've reached the limit
            if max_tasks and len(all_tasks) >= max_tasks:
                logger.info(f"Reached max_tasks limit: {max_tasks:,}")
                all_tasks = all_tasks[:max_tasks]
                break
            
            # Check if we've fetched everything
            if len(all_tasks) >= total_available:
                logger.info("Fetched all available tasks")
                break
            
            # Move to next page
            page += 1
            
            # Rate limiting delay
            if delay_seconds > 0:
                time.sleep(delay_seconds)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed on page {page}: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error on page {page}: {e}")
            break
    
    return all_tasks


def save_tasks(tasks: list[dict], output_path: Path) -> None:
    """Save tasks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False))
            f.write("\n")
    
    logger.info(f"✅ Saved {len(tasks):,} tasks to {output_path}")
    
    # Also save metadata
    metadata = {
        "total_tasks": len(tasks),
        "fetched_at": datetime.utcnow().isoformat(),
        "output_file": str(output_path),
    }
    
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch tasks with solutions from new paginated API"
    )
    
    parser.add_argument(
        "--base-url",
        default="https://api-leaderboard.autoppia.com",
        help="API base URL (default: https://api-leaderboard.autoppia.com)",
    )
    parser.add_argument(
        "--api-key",
        default="AIagent2025",
        help="API key for authentication (default: AIagent2025)",
    )
    parser.add_argument(
        "--limit-per-page",
        type=int,
        default=500,
        help="Tasks per page (default: 500)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum tasks to fetch (default: all available)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/score_model_pipeline/datasets/leaderboard_all_tasks.jsonl"),
        help="Output JSONL file path",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("FETCHING TASKS WITH SOLUTIONS FROM API")
    logger.info("=" * 80)
    logger.info(f"Base URL: {args.base_url}")
    logger.info(f"Limit per page: {args.limit_per_page}")
    logger.info(f"Max tasks: {args.max_tasks or 'all available'}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)
    
    # Fetch tasks
    tasks = fetch_all_tasks(
        base_url=args.base_url,
        api_key=args.api_key,
        limit_per_page=args.limit_per_page,
        max_tasks=args.max_tasks,
        delay_seconds=args.delay,
    )
    
    if not tasks:
        logger.error("❌ No tasks fetched!")
        return 1
    
    # Save to file
    save_tasks(tasks, args.output)
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total tasks fetched: {len(tasks):,}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"File size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

