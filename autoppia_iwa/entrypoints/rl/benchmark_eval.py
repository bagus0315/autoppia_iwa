"""
Benchmark evaluation for trained RL agent.

Evaluates a trained PPO policy using RLModelAgent wrapper.
This script allows RL agents to be evaluated like any other web agent.

Run with:
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval --model-path <path>
  
Examples:
  # Evaluate smoke test checkpoint on autocinema
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval \\
    --model-path data/rl/checkpoints/ppo_smoke/ppo_final.zip \\
    --projects autocinema --tasks-per-project 3
  
  # Evaluate on multiple projects
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval \\
    --model-path data/rl/checkpoints/ppo_10k/best_model.zip \\
    --projects autocinema autowork autolodge \\
    --tasks-per-project 5 --max-steps 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from autoppia_iwa.entrypoints.benchmark.benchmark import Benchmark
from autoppia_iwa.entrypoints.benchmark.config import BenchmarkConfig
from autoppia_iwa.entrypoints.benchmark.task_generation import get_projects_by_ids
from autoppia_iwa.src.demo_webs.config import demo_web_projects
from autoppia_iwa.src.web_agents.rl import RLModelAgent, RLModelAgentConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent on benchmark tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test evaluation
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval \\
    --model-path data/rl/checkpoints/ppo_smoke/ppo_final.zip \\
    --projects autocinema --tasks-per-project 3 --max-steps 20
  
  # Full evaluation on multiple projects
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval \\
    --model-path data/rl/checkpoints/ppo_10k/best_model.zip \\
    --projects autocinema autowork autolodge autodrive \\
    --tasks-per-project 10 --max-steps 30 --deterministic
  
  # List available models
  python -m autoppia_iwa.entrypoints.rl.benchmark_eval --list-models
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to trained PPO checkpoint (.zip file). If not provided, uses latest.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model checkpoints and exit",
    )
    
    # Task selection
    parser.add_argument(
        "--projects",
        nargs="+",
        default=["autocinema"],
        help="Project IDs to evaluate on (default: autocinema)",
    )
    parser.add_argument(
        "--tasks-per-project",
        type=int,
        default=5,
        help="Number of tasks to evaluate per project (default: 5)",
    )
    parser.add_argument(
        "--use-cached-tasks",
        action="store_true",
        default=True,
        help="Use cached tasks if available (default: True)",
    )
    
    # Agent configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps per episode (default: 30)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Top-k elements to consider for action space (default: 12)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)",
    )
    parser.add_argument(
        "--random-fallback",
        action="store_true",
        help="Use random policy if model fails to load (default: False)",
    )
    
    # Execution options
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each task (default: 1)",
    )
    parser.add_argument(
        "--record-gif",
        action="store_true",
        help="Record GIFs of agent interactions (slower)",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rl/eval_results"),
        help="Directory to save evaluation results (default: data/rl/eval_results)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="Save results as JSON (default: True)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate result plots (default: False)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def list_available_models() -> None:
    """List all available model checkpoints."""
    ckpt_dir = Path("data/rl/checkpoints")
    
    logger.info("=" * 80)
    logger.info("AVAILABLE RL MODEL CHECKPOINTS")
    logger.info("=" * 80)
    
    if not ckpt_dir.exists():
        logger.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    found_models = False
    for subdir in sorted(ckpt_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        ckpt_files = list(subdir.glob("*.zip"))
        if not ckpt_files:
            continue
        
        found_models = True
        logger.info(f"\n📁 {subdir.name}/")
        for ckpt_file in sorted(ckpt_files):
            size_mb = ckpt_file.stat().st_size / 1024 / 1024
            logger.info(f"  ✅ {ckpt_file.name} ({size_mb:.1f} MB)")
            logger.info(f"     Path: {ckpt_file}")
    
    if not found_models:
        logger.warning("No model checkpoints found!")
        logger.info("\nTo train a model first:")
        logger.info("  python -m autoppia_iwa.src.rl.agent.entrypoints.train_ppo_with_score_model \\")
        logger.info("    --config autoppia_iwa/src/rl/agent/configs/ppo_smoke.yaml")
    
    logger.info("=" * 80)


def find_latest_checkpoint() -> Path | None:
    """Find the most recent checkpoint file."""
    ckpt_dir = Path("data/rl/checkpoints")
    if not ckpt_dir.exists():
        return None
    
    all_checkpoints = list(ckpt_dir.rglob("*.zip"))
    if not all_checkpoints:
        return None
    
    # Sort by modification time
    latest = max(all_checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def summarize_results(results_dir: Path) -> None:
    """Print a summary of evaluation results."""
    json_file = results_dir / "results.json"
    if not json_file.exists():
        logger.warning(f"Results file not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # Per-agent summary
    for agent_name, agent_results in results.get("agents", {}).items():
        logger.info(f"\n🤖 Agent: {agent_name}")
        
        if "projects" in agent_results:
            for project_name, project_results in agent_results["projects"].items():
                success_rate = project_results.get("success_rate", 0.0) * 100
                total_tasks = project_results.get("total_tasks", 0)
                avg_time = project_results.get("avg_time", 0.0)
                
                logger.info(f"  📊 {project_name}:")
                logger.info(f"     Success Rate: {success_rate:.1f}%")
                logger.info(f"     Total Tasks: {total_tasks}")
                logger.info(f"     Avg Time: {avg_time:.2f}s")
    
    logger.info("=" * 80)


def main():
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        list_available_models()
        return
    
    # Set log level
    if args.verbose:
        logger.add(lambda msg: None, level="DEBUG")
    
    # Find or validate model path
    if args.model_path is None:
        logger.info("No model path specified, searching for latest checkpoint...")
        args.model_path = find_latest_checkpoint()
        if args.model_path is None:
            logger.error("No model checkpoints found!")
            logger.info("\nRun with --list-models to see available models, or specify --model-path")
            return
        logger.info(f"Using latest checkpoint: {args.model_path}")
    
    if not args.model_path.exists():
        logger.error(f"Model checkpoint not found: {args.model_path}")
        logger.info("\nAvailable checkpoints:")
        list_available_models()
        return
    
    logger.info(f"✅ Model checkpoint: {args.model_path}")
    logger.info(f"   Size: {args.model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Setup RL agent
    rl_agent = RLModelAgent(
        id="rl-eval",
        name=f"RLAgent-{args.model_path.parent.name}",
        config=RLModelAgentConfig(
            model_path=str(args.model_path),
            topk=args.topk,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            random_fallback=args.random_fallback,
        ),
    )
    
    # Get projects
    try:
        projects = get_projects_by_ids(demo_web_projects, args.projects)
    except ValueError as e:
        logger.error(f"Invalid project IDs: {e}")
        logger.info("\nAvailable projects:")
        for proj in demo_web_projects:
            logger.info(f"  - {proj.id}: {proj.name}")
        return
    
    logger.info(f"✅ Evaluating on {len(projects)} projects: {[p.id for p in projects]}")
    
    # Setup benchmark config
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = BenchmarkConfig(
        projects=projects,
        agents=[rl_agent],
        # Tasks
        use_cached_tasks=args.use_cached_tasks,
        prompts_per_use_case=args.tasks_per_project,
        num_use_cases=1,  # RL agent works with single use case per evaluation
        # Execution
        runs=args.runs,
        max_parallel_agent_calls=1,  # RL agent uses browser, must be serial
        use_cached_solutions=False,  # Always run fresh for RL evaluation
        record_gif=args.record_gif,
        # Dynamic HTML
        enable_dynamic_html=False,
        # Persistence
        save_results_json=args.save_json,
        plot_results=args.plot,
    )
    
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STARTING RL AGENT BENCHMARK EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Model: {args.model_path.name}")
        logger.info(f"Projects: {[p.name for p in projects]}")
        logger.info(f"Tasks per project: {args.tasks_per_project}")
        logger.info(f"Max steps per task: {args.max_steps}")
        logger.info(f"Top-k: {args.topk}")
        logger.info(f"Deterministic: {args.deterministic}")
        logger.info(f"Runs per task: {args.runs}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("=" * 80 + "\n")
        
        benchmark = Benchmark(cfg)
        asyncio.run(benchmark.run())
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {cfg.results_directory}")
        
        # Print summary
        summarize_results(cfg.results_directory)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}", exc_info=args.verbose)
        if not args.verbose:
            logger.info("Run with --verbose for full traceback")
        raise


if __name__ == "__main__":
    main()

