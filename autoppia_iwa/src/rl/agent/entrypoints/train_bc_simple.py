#!/usr/bin/env python3
"""Train BC policy on pre-replayed traces with action indices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, random_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train BC on pre-replayed traces")
    parser.add_argument("--traces-dir", type=Path, required=True, help="Directory containing BC trace files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for trained model")
    parser.add_argument("--checkpoint-name", type=str, default="bc_policy", help="Checkpoint name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


class BCTraceDataset(Dataset):
    """Dataset that loads BC traces directly."""
    
    def __init__(self, traces_dir: Path):
        self.steps = []
        logger.info(f"Loading BC traces from {traces_dir}")
        
        # Find all trace files
        trace_files = list(traces_dir.rglob("bc_trace_*.jsonl"))
        logger.info(f"Found {len(trace_files)} trace files")
        
        # Load all steps
        for trace_file in trace_files:
            with open(trace_file, "r") as f:
                for line in f:
                    step_data = json.loads(line)
                    self.steps.append(step_data)
        
        logger.info(f"Loaded {len(self.steps)} total steps for training")
    
    def __len__(self):
        return len(self.steps)
    
    def __getitem__(self, idx):
        step = self.steps[idx]
        obs = step["observation"]
        action_index = step["action_index"]
        
        # Convert to tensors
        obs_tensors = {
            "dom_ids": torch.tensor(obs["dom_ids"], dtype=torch.int32),
            "goal_ids": torch.tensor(obs["goal_ids"], dtype=torch.int32),
            "url_id": torch.tensor(obs["url_id"], dtype=torch.int32),
            "prev_actions": torch.tensor(obs["prev_actions"], dtype=torch.int32),
            "topk_meta": torch.tensor(obs["topk_meta"], dtype=torch.float32),
            "topk_text_ids": torch.tensor(obs["topk_text_ids"], dtype=torch.int32),
            "score": torch.tensor(obs["score"], dtype=torch.float32),
        }
        action_tensor = torch.tensor(action_index, dtype=torch.long)
        
        return obs_tensors, action_tensor


def collate_fn(batch):
    """Collate function for DataLoader."""
    obs_batch = {
        key: torch.stack([item[0][key] for item in batch])
        for key in batch[0][0].keys()
    }
    action_batch = torch.stack([item[1] for item in batch])
    return obs_batch, action_batch


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for obs_batch, action_batch in dataloader:
        # Move to device
        obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
        action_batch = action_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get action logits from policy
        with torch.no_grad():
            # Use the policy network to get distribution
            distribution = model.policy.get_distribution(obs_batch)
        
        logits = distribution.distribution.logits
        
        # Compute loss
        loss = criterion(logits, action_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == action_batch).sum().item()
        total += action_batch.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for obs_batch, action_batch in dataloader:
            # Move to device
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}
            action_batch = action_batch.to(device)
            
            # Forward pass
            distribution = model.policy.get_distribution(obs_batch)
            logits = distribution.distribution.logits
            
            # Compute loss
            loss = criterion(logits, action_batch)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == action_batch).sum().item()
            total += action_batch.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("BC TRAINING ON PRE-REPLAYED TRACES")
    logger.info("=" * 80)
    logger.info(f"Traces directory: {args.traces_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    # Load dataset
    full_dataset = BCTraceDataset(args.traces_dir)
    
    if len(full_dataset) == 0:
        logger.error("No traces loaded. Cannot train.")
        return
    
    # Split into train and validation
    val_size = int(len(full_dataset) * args.validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Training steps: {len(train_dataset)}")
    logger.info(f"Validation steps: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for debugging
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Create a dummy environment for PPO initialization
    # We'll use PPO's policy network but train it with supervised learning
    from stable_baselines3.common.policies import MultiInputPolicy
    from gymnasium import spaces
    import numpy as np
    
    # Define observation space (matching IWAWebEnv)
    observation_space = spaces.Dict({
        "dom_ids": spaces.Box(low=0, high=8192, shape=(200,), dtype=np.int32),
        "goal_ids": spaces.Box(low=0, high=4096, shape=(48,), dtype=np.int32),
        "url_id": spaces.Box(low=0, high=1024, shape=(), dtype=np.int32),
        "prev_actions": spaces.Box(low=0, high=50, shape=(10,), dtype=np.int32),
        "topk_meta": spaces.Box(low=0, high=1, shape=(50, 8), dtype=np.float32),
        "topk_text_ids": spaces.Box(low=0, high=8192, shape=(50, 12), dtype=np.int32),
        "score": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
    })
    action_space = spaces.Discrete(50)
    
    # Create PPO model (we'll only use its policy network)
    logger.info("Initializing PPO policy network...")
    model = PPO(
        MultiInputPolicy,
        env=None,  # We don't need an actual env
        learning_rate=args.learning_rate,
        device=args.device,
    )
    
    # Manually set spaces
    model.observation_space = observation_space
    model.action_space = action_space
    model.policy.observation_space = observation_space
    model.policy.action_space = action_space
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("\n🚀 Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        logger.info(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.2%}")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        logger.info(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.2%}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = args.output_dir / f"{args.checkpoint_name}_best.zip"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save(output_path)
            logger.info(f"✅ Saved best model (val_acc: {val_acc:.2%})")
    
    # Save final model
    final_path = args.output_dir / f"{args.checkpoint_name}.zip"
    model.save(final_path)
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2%}")
    logger.info(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()

