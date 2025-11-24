#!/usr/bin/env python3
"""
Behavioral Cloning (BC) training script.

Trains a policy using expert demonstrations from replay traces.
The trained BC checkpoint can then be used to warm-start PPO training.
"""

import argparse
from pathlib import Path

from loguru import logger
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv
from autoppia_iwa.src.rl.agent.offline.bc_trainer import (
    BehaviorCloningConfig,
    BehaviorCloningTrainer,
)
from autoppia_iwa.src.rl.agent.offline.dataset_trajectory_provider import DatasetTrajectoryProvider


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train policy using Behavioral Cloning from expert traces"
    )
    
    # Data
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/score_model_pipeline/datasets/leaderboard_8_7k_flattened.jsonl"),
        help="Dataset JSONL file (default: 8.7k dataset)",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=Path("data/score_model_pipeline/raw_traces/leaderboard_8_7k_20251118"),
        help="Directory containing trace files (default: 8.7k traces)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum trajectories to use (default: all)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=3,
        help="Minimum steps per trajectory (default: 3)",
    )
    
    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split (default: 0.1)",
    )
    
    # Environment
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Top-k candidates per step (default: 12)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode (default: 50)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rl/checkpoints/bc_training"),
        help="Output directory for BC checkpoint (default: data/rl/checkpoints/bc_training)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="bc_policy",
        help="Checkpoint filename (default: bc_policy)",
    )
    
    # Options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, default: auto)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logger.info("Verbose mode enabled")
    
    logger.info("=" * 80)
    logger.info("BEHAVIORAL CLONING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Traces directory: {args.traces_dir}")
    logger.info(f"Max trajectories: {args.max_trajectories or 'all'}")
    logger.info(f"Min steps per trajectory: {args.min_steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Validation split: {args.validation_split}")
    logger.info(f"Top-k: {args.topk}")
    logger.info(f"Output: {args.output_dir / args.checkpoint_name}.zip")
    logger.info("=" * 80)
    
    # Create environment config
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
    }
    
    # Create trajectory provider
    logger.info("\n📂 Loading expert trajectories (this will take ~10-30 minutes)...")
    logger.info("   • Converting dataset entries to trajectories")
    logger.info("   • Replaying actions through environment")
    logger.info("   • Capturing observations and action indices")
    logger.info("")
    provider = DatasetTrajectoryProvider(
        dataset_paths=[args.dataset],
        traces_root=args.traces_dir,
        env_config=env_config,  # FIXED: Pass env_config to match policy dimensions
        max_trajectories=args.max_trajectories,
        min_steps=args.min_steps,
    )
    
    # Create dummy environment for policy initialization
    logger.info("🏗️  Initializing policy...")
    dummy_env = IWAWebEnv(env_config)
    
    # Wrap environment with ActionMasker for MaskablePPO
    def mask_fn(env):
        return env.get_action_mask()
    
    wrapped_env = ActionMasker(dummy_env, mask_fn)
    
    # Create MaskablePPO model (we'll use its policy for BC training)
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=wrapped_env,
        learning_rate=args.learning_rate,
        verbose=0,
    )
    
    # Create BC trainer
    bc_config = BehaviorCloningConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        max_trajectories=args.max_trajectories,
        device=args.device,
        log_interval=50,
    )
    
    trainer = BehaviorCloningTrainer(model.policy, bc_config)
    
    # Train
    logger.info("\n🚀 Starting BC training...")
    logger.info("-" * 80)
    
    try:
        history = trainer.fit(provider)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ BC TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Final training loss: {history['train_loss']:.4f}")
        logger.info(f"Final training accuracy: {history['train_acc']:.2%}")
        import math
        if not math.isnan(history['val_loss']):
            logger.info(f"Final validation loss: {history['val_loss']:.4f}")
            logger.info(f"Final validation accuracy: {history['val_acc']:.2%}")
        logger.info(f"Total steps: {int(history['num_steps'])}")
        logger.info("=" * 80)
        
        # Save BC checkpoint
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{args.checkpoint_name}.zip"
        
        logger.info(f"\n💾 Saving BC checkpoint to {output_path}...")
        model.save(output_path)
        
        logger.info(f"✅ BC checkpoint saved: {output_path}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Also save training history
        import json
        history_path = args.output_dir / f"{args.checkpoint_name}_history.json"
        with history_path.open("w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"✅ Training history saved: {history_path}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
    finally:
        dummy_env.close()


if __name__ == "__main__":
    main()

