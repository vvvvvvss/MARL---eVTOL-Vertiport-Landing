"""
PPO Training Script for Vertiport Scheduling
Trains a PPO agent on the VertiportRLEnv using Stable Baselines3

This script:
1. Creates and trains a PPO agent on the vertiport scheduling problem
2. Saves checkpoints during training
3. Logs metrics to TensorBoard and CSV
4. Evaluates performance periodically
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.logger import configure
import gymnasium as gym

from vertiport_rl_env import VertiportRLEnv
from baselines import FCFSScheduler, GreedyScheduler
from vertiport_env import VertiportEnv


class MetricsCallback(BaseCallback):
    """
    Custom callback to log additional metrics during training.
    """
    
    def __init__(self, log_dir: str = ""):
        super().__init__()
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Log episode reward
        if "episode" in self.locals:
            episode_info = self.locals["episode"]
            self.episode_rewards.append(episode_info["r"])
            self.episode_lengths.append(episode_info["l"])
        
        return True


def create_train_env(
    arrival_rate: float = 20.0,
    num_pads: int = 8,
    **kwargs
) -> VertiportRLEnv:
    """Create a training environment."""
    env = VertiportRLEnv(
        num_pads=num_pads,
        arrival_rate=arrival_rate,
        **kwargs
    )
    return env


def create_eval_env(
    arrival_rate: float = 20.0,
    num_pads: int = 8,
    **kwargs
) -> VertiportRLEnv:
    """Create an evaluation environment."""
    env = VertiportRLEnv(
        num_pads=num_pads,
        arrival_rate=arrival_rate,
        **kwargs
    )
    return env


def train_ppo(
    total_timesteps: int = 500000,
    arrival_rate: float = 20.0,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    model_name: str = "evtol_ppo",
    log_dir: str = "./evtol_training/",
    eval_freq: int = 10000,
    save_freq: int = 5000,
    verbose: int = 1,
):
    """
    Train a PPO agent for vertiport scheduling.
    
    Args:
        total_timesteps: Total training steps
        arrival_rate: Aircraft arrival rate (per hour)
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps before update
        batch_size: Batch size for training
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping range
        ent_coef: Entropy coefficient (exploration)
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        model_name: Name for saved model
        log_dir: Directory for logs and checkpoints
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        verbose: Logging verbosity
    
    Returns:
        Trained PPO model
    """
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{arrival_rate}ac_per_hr_{timestamp}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PPO Training for Vertiport Scheduling")
    print("=" * 80)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Arrival rate: {arrival_rate} aircraft/hour")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)
    
    # Create training environment
    print("\nCreating training environment...")
    train_env = create_train_env(
        arrival_rate=arrival_rate,
        max_aircraft=50,
        dt=0.1
    )
    
    # Verify environment is Gymnasium compatible
    print("Checking environment compatibility...")
    try:
        check_env(train_env)
        print("✓ Environment is Gymnasium compatible")
    except Exception as e:
        print(f"⚠ Environment check failed: {e}")
    
    # Setup logging
    logger = configure(str(run_dir / "logs"), ["stdout", "tensorboard", "csv"])
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=verbose,
        tensorboard_log=str(run_dir / "tensorboard"),
        device="auto"  # Auto-detect GPU if available
    )
    
    model.set_logger(logger)
    
    # Setup evaluation
    print("Setting up evaluation callback...")
    eval_env = create_eval_env(arrival_rate=arrival_rate)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=False,
        render=False
    )
    
    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=model_name,
        save_replay_buffer=False
    )
    
    # Metrics callback
    metrics_callback = MetricsCallback(log_dir=str(run_dir))
    
    # Train
    print(f"\nStarting training... (this will take a while)")
    print(f"You can monitor progress at: {run_dir / 'tensorboard'}")
    print("Command: tensorboard --logdir=evtol_training/\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, metrics_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    # Save final model
    final_model_path = str(run_dir / f"final_{model_name}")
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Save training config
    config = {
        'total_timesteps': total_timesteps,
        'arrival_rate': arrival_rate,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'run_dir': str(run_dir),
    }
    
    import json
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Training config saved to: {run_dir / 'config.json'}")
    
    return model, run_dir


def train_curriculum(
    stages: list = None,
    model_name: str = "evtol_ppo_curriculum",
    log_dir: str = "./evtol_training/",
):
    """
    Train with curriculum learning: gradually increase difficulty.
    
    Args:
        stages: List of (arrival_rate, timesteps) tuples
        model_name: Name for saved models
        log_dir: Directory for logs
    
    Returns:
        Trained model and results
    """
    
    if stages is None:
        # Default curriculum: start easy, get harder
        stages = [
            (10, 100000),   # 10 ac/hr for 100k steps
            (20, 150000),   # 20 ac/hr for 150k steps
            (30, 150000),   # 30 ac/hr for 150k steps
            (40, 100000),   # 40 ac/hr for 100k steps
        ]
    
    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING: Progressive Difficulty Increase")
    print("=" * 80)
    
    model = None
    results = {}
    
    for stage_num, (arrival_rate, timesteps) in enumerate(stages, 1):
        print(f"\n\nSTAGE {stage_num}: Arrival Rate = {arrival_rate} ac/hr, Steps = {timesteps}")
        print("-" * 80)
        
        stage_model_name = f"{model_name}_stage{stage_num}"
        
        trained_model, run_dir = train_ppo(
            total_timesteps=timesteps,
            arrival_rate=arrival_rate,
            model_name=stage_model_name,
            log_dir=log_dir,
            learning_rate=3e-4,
            eval_freq=5000,
        )
        
        model = trained_model
        results[f"stage_{stage_num}"] = {
            "arrival_rate": arrival_rate,
            "timesteps": timesteps,
            "model_path": str(run_dir / f"final_{stage_model_name}")
        }
    
    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final model saved at: {results[f'stage_{len(stages)}']['model_path']}")
    
    return model, results


def main():
    """Main training script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO for vertiport scheduling")
    parser.add_argument("--total-timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--arrival-rate", type=float, default=20.0,
                       help="Aircraft arrival rate (per hour)")
    parser.add_argument("--curriculum", action="store_true",
                       help="Use curriculum learning (progressive difficulty)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--log-dir", type=str, default="./evtol_training/",
                       help="Directory for logs and checkpoints")
    parser.add_argument("--model-name", type=str, default="evtol_ppo",
                       help="Name for saved model")
    
    args = parser.parse_args()
    
    if args.curriculum:
        model, results = train_curriculum(log_dir=args.log_dir)
    else:
        model, run_dir = train_ppo(
            total_timesteps=args.total_timesteps,
            arrival_rate=args.arrival_rate,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            model_name=args.model_name,
            log_dir=args.log_dir
        )
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("1. View training progress: tensorboard --logdir=evtol_training/")
    print("2. Evaluate the model: python evaluate_ppo.py")
    print("3. Compare with baselines: python compare_policies.py")


if __name__ == "__main__":
    main()
