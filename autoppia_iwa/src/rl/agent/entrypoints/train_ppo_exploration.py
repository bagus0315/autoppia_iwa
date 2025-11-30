#!/usr/bin/env python3
"""Train PPO with exploration - skips BC, uses improved components."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from autoppia_iwa.src.rl.agent.envs.iwa_env import IWAWebEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with exploration (no BC)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rl/checkpoints/ppo_exploration"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="TopK candidates (use variable K if --use-variable-k)",
    )
    parser.add_argument(
        "--use-variable-k",
        action="store_true",
        help="Use variable K instead of fixed K",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=30,
        help="Minimum K when using variable K",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=70,
        help="Maximum K when using variable K",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=True,
        help="Use reranker for topK selection",
    )
    parser.add_argument(
        "--reranker-model-path",
        type=str,
        default=None,
        help="Path to trained reranker model",
    )
    parser.add_argument(
        "--use-llm-rerank",
        action="store_true",
        help="Use LLM for reranking (slower but more accurate)",
    )
    parser.add_argument(
        "--use-tokenizer",
        action="store_true",
        default=True,
        help="Use proper tokenizer instead of hash",
    )
    parser.add_argument(
        "--reward-model-path",
        type=str,
        default=None,
        help="Path to reward model for reward shaping",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset for diverse task sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None = random seed each episode)",
    )
    return parser.parse_args()


def make_env(cfg: dict, rank: int = 0) -> gym.Env:
    """Create environment with improved components."""
    def _init():
        env = IWAWebEnv(cfg=cfg)
        env = Monitor(env, filename=None, allow_early_resets=True)
        return env
    return _init


def main() -> None:
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting PPO Exploration Training (No BC)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total timesteps: {args.total_timesteps}")
    logger.info(f"TopK: {args.topk} (variable: {args.use_variable_k}, range: {args.k_min}-{args.k_max})")
    logger.info(f"Reranker: {args.use_reranker} (LLM: {args.use_llm_rerank})")
    logger.info(f"Tokenizer: {args.use_tokenizer}")
    
    # Environment configuration
    env_cfg = {
        "topk": args.topk,
        "max_steps": 50,
        "goal_vocab_size": 4096,
        "dom_vocab_size": 8192,
        "url_vocab_size": 1024,
        "max_goal_tokens": 48,
        "max_dom_tokens": 200,
        "max_element_tokens": 12,
        "action_history": 10,
        "use_variable_k": args.use_variable_k,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "use_reranker": args.use_reranker,
        "reranker_model_path": args.reranker_model_path,
        "use_llm_rerank": args.use_llm_rerank,
        "use_tokenizer": args.use_tokenizer,
    }
    
    if args.reward_model_path:
        env_cfg["reward_model_path"] = args.reward_model_path
        logger.info(f"Using reward model: {args.reward_model_path}")
    
    if args.dataset_path:
        env_cfg["dataset_path"] = args.dataset_path
        logger.info(f"Using dataset: {args.dataset_path}")
    
    # Create environment
    env = DummyVecEnv([make_env(env_cfg, rank=0)])
    
    # PPO configuration
    ppo_cfg = {
        "policy": "MultiInputPolicy",
        "env": env,
        "learning_rate": 3e-4,  # Standard PPO LR (higher than BC+PPO)
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Encourage exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": str(args.output_dir / "tensorboard"),
    }
    
    # Create PPO agent (no BC checkpoint - start from scratch)
    logger.info("Creating PPO agent (no BC warm-start)")
    model = PPO(**ppo_cfg)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(args.output_dir),
        name_prefix="ppo_exploration",
    )
    
    # Evaluation environment (same config)
    eval_env = DummyVecEnv([make_env(env_cfg, rank=1)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir / "best_model"),
        log_path=str(args.output_dir / "eval_logs"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    
    # Train
    logger.info("Starting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_path = args.output_dir / "ppo_exploration_final.zip"
    model.save(str(final_path))
    logger.info(f"✅ Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    main()

