"""
B3: Curriculum Learning Optimization

Implements progressive difficulty training for multi-agent RL:
1. Gradually increases aircraft arrival rate
2. Multi-stage training with adaptive scheduling
3. Dynamic performance tracking
4. Automatic stage progression based on metrics

This enables faster convergence and better generalization.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import gymnasium as gym

from vertiport_rl_env import VertiportRLEnv


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    stage_id: int
    name: str
    arrival_rate: float  # Aircraft per hour
    timesteps: int  # Training steps for this stage
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_epochs: int = 10
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'stage_id': self.stage_id,
            'name': self.name,
            'arrival_rate': self.arrival_rate,
            'timesteps': self.timesteps,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'description': self.description
        }


class CurriculumCallback(BaseCallback):
    """Custom callback to track curriculum learning progress."""
    
    def __init__(self, stage: CurriculumStage, eval_freq: int = 5000,
                 log_dir: str = "./evtol_training/"):
        super().__init__()
        self.stage = stage
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.stage_metrics = {
            'stage_id': stage.stage_id,
            'stage_name': stage.name,
            'arrival_rate': stage.arrival_rate,
            'timesteps': stage.timesteps,
            'milestones': []
        }
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        if self.num_timesteps % self.eval_freq == 0:
            # Log progress
            milestone = {
                'timestep': self.num_timesteps,
                'current_progress': self.num_timesteps / self.stage.timesteps * 100
            }
            self.stage_metrics['milestones'].append(milestone)
            
            if self.verbose > 0:
                progress = (self.num_timesteps / self.stage.timesteps) * 100
                print(f"Stage {self.stage.stage_id}: {progress:.1f}% complete "
                      f"({self.num_timesteps}/{self.stage.timesteps} steps)")
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        if self.verbose > 0:
            print(f"Stage {self.stage.stage_id} ({self.stage.name}) complete!")


class CurriculumScheduler:
    """Manages curriculum learning stages and progression."""
    
    def __init__(self, log_dir: str = "./evtol_training/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stages: List[CurriculumStage] = []
        self.current_stage_idx = 0
        self.training_history: List[Dict[str, Any]] = []
        
    def add_stage(self, stage: CurriculumStage) -> None:
        """Add a curriculum stage."""
        self.stages.append(stage)
        
    def create_default_curriculum(self) -> None:
        """Create 4-stage default curriculum."""
        self.stages = [
            CurriculumStage(
                stage_id=1,
                name="Easy",
                arrival_rate=5.0,
                timesteps=100000,
                learning_rate=3e-4,
                description="Sparse traffic (5 ac/hr) - learn basics"
            ),
            CurriculumStage(
                stage_id=2,
                name="Medium-Easy",
                arrival_rate=15.0,
                timesteps=100000,
                learning_rate=2e-4,
                description="Light traffic (15 ac/hr) - improve coordination"
            ),
            CurriculumStage(
                stage_id=3,
                name="Medium-Hard",
                arrival_rate=25.0,
                timesteps=100000,
                learning_rate=1e-4,
                description="Moderate traffic (25 ac/hr) - handle congestion"
            ),
            CurriculumStage(
                stage_id=4,
                name="Hard",
                arrival_rate=40.0,
                timesteps=100000,
                learning_rate=5e-5,
                description="Heavy traffic (40 ac/hr) - expert policy"
            )
        ]
        
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get current stage."""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def advance_stage(self) -> bool:
        """Move to next stage. Returns True if more stages exist."""
        self.current_stage_idx += 1
        return self.current_stage_idx < len(self.stages)
    
    def save_progress(self, filepath: Optional[str] = None) -> None:
        """Save curriculum progress to JSON."""
        if filepath is None:
            filepath = self.log_dir / "curriculum_progress.json"
        
        progress = {
            'timestamp': datetime.now().isoformat(),
            'current_stage': self.current_stage_idx,
            'total_stages': len(self.stages),
            'stages': [s.to_dict() for s in self.stages],
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def log_stage_completion(self, stage: CurriculumStage, 
                            metrics: Dict[str, float]) -> None:
        """Log completion of a stage with metrics."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage_id': stage.stage_id,
            'stage_name': stage.name,
            'arrival_rate': stage.arrival_rate,
            'metrics': metrics
        }
        self.training_history.append(entry)
        self.save_progress()


class CurriculumTrainer:
    """Main trainer for curriculum learning."""
    
    def __init__(self, 
                 model_dir: str = "./evtol_training/",
                 verbose: int = 1):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.scheduler = CurriculumScheduler(str(self.model_dir))
        self.models_by_stage = {}
        self.metrics_by_stage = {}
        
    def train(self, 
              num_stages: Optional[int] = None,
              use_pretrained: bool = False,
              save_interval: int = 50000) -> Tuple[PPO, Dict[str, Any]]:
        """
        Train through all curriculum stages.
        
        Args:
            num_stages: Override number of stages (None = use all)
            use_pretrained: Resume from existing model
            save_interval: Save checkpoint interval (steps)
            
        Returns:
            (final_model, training_summary)
        """
        self.scheduler.create_default_curriculum()
        
        if num_stages:
            self.scheduler.stages = self.scheduler.stages[:num_stages]
        
        total_stages = len(self.scheduler.stages)
        print(f"\n{'='*70}")
        print(f"CURRICULUM LEARNING: {total_stages}-STAGE TRAINING")
        print(f"{'='*70}\n")
        
        model = None
        total_timesteps_trained = 0
        
        for stage_idx, stage in enumerate(self.scheduler.stages):
            self.scheduler.current_stage_idx = stage_idx
            
            print(f"\n{'─'*70}")
            print(f"STAGE {stage.stage_id}/{total_stages}: {stage.name.upper()}")
            print(f"{'─'*70}")
            print(f"Arrival Rate:  {stage.arrival_rate} ac/hr")
            print(f"Timesteps:     {stage.timesteps:,}")
            print(f"Learning Rate: {stage.learning_rate}")
            print(f"Description:   {stage.description}\n")
            
            # Create environment for this stage
            env = VertiportRLEnv(
                arrival_rate=stage.arrival_rate,
                num_pads=8
            )
            
            # Create or load model
            if model is None or not use_pretrained:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=stage.learning_rate,
                    n_steps=stage.n_steps,
                    batch_size=stage.batch_size,
                    n_epochs=stage.n_epochs,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    verbose=self.verbose,
                    tensorboard_log=str(self.model_dir / f"stage_{stage.stage_id}_logs")
                )
            else:
                # Continue training previous model
                model.set_env(env)
                model.learning_rate = stage.learning_rate
            
            # Create callbacks
            eval_env = VertiportRLEnv(arrival_rate=stage.arrival_rate)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / f"stage_{stage.stage_id}_best"),
                log_path=str(self.model_dir / f"stage_{stage.stage_id}_eval"),
                eval_freq=max(1, stage.timesteps // 5),
                deterministic=True,
                render=False
            )
            
            curriculum_callback = CurriculumCallback(
                stage=stage,
                eval_freq=max(1, stage.timesteps // 10),
                log_dir=str(self.model_dir)
            )
            
            # Train for this stage
            try:
                model.learn(
                    total_timesteps=stage.timesteps,
                    callback=[eval_callback, curriculum_callback],
                    tb_log_name=f"stage_{stage.stage_id}",
                    progress_bar=True
                )
                
                # Save stage model
                stage_model_path = self.model_dir / f"stage_{stage.stage_id}_model"
                model.save(str(stage_model_path))
                self.models_by_stage[stage.stage_id] = stage_model_path
                
                # Evaluate stage
                metrics = self._evaluate_stage(env, model, stage)
                self.metrics_by_stage[stage.stage_id] = metrics
                self.scheduler.log_stage_completion(stage, metrics)
                
                total_timesteps_trained += stage.timesteps
                
                print(f"\n✓ Stage {stage.stage_id} complete!")
                print(f"  Average Reward: {metrics.get('avg_reward', 0):.2f}")
                print(f"  Total Steps:    {total_timesteps_trained:,}\n")
                
            except Exception as e:
                print(f"\n✗ Error training stage {stage.stage_id}: {e}")
                raise
            finally:
                env.close()
                eval_env.close()
        
        # Generate summary
        summary = self._generate_summary(total_timesteps_trained)
        
        # Save final results
        self.scheduler.save_progress()
        self._save_summary(summary)
        
        return model, summary
    
    def _evaluate_stage(self, env: gym.Env, model: PPO, 
                       stage: CurriculumStage, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate model on stage environment."""
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        
        return {
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_length': float(np.mean(episode_lengths)),
            'num_eval_episodes': num_episodes
        }
    
    def _generate_summary(self, total_timesteps: int) -> Dict[str, Any]:
        """Generate training summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_timesteps': total_timesteps,
            'total_stages': len(self.scheduler.stages),
            'stages_trained': len(self.metrics_by_stage),
            'stage_progress': []
        }
        
        for stage in self.scheduler.stages:
            if stage.stage_id in self.metrics_by_stage:
                summary['stage_progress'].append({
                    'stage_id': stage.stage_id,
                    'name': stage.name,
                    'arrival_rate': stage.arrival_rate,
                    'metrics': self.metrics_by_stage[stage.stage_id]
                })
        
        return summary
    
    def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Save training summary to file."""
        filepath = self.model_dir / "curriculum_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save as CSV
        csv_filepath = self.model_dir / "curriculum_results.csv"
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Stage ID', 'Name', 'Arrival Rate (ac/hr)', 
                           'Avg Reward', 'Std Reward', 'Avg Episode Length'])
            
            for stage_info in summary['stage_progress']:
                metrics = stage_info['metrics']
                writer.writerow([
                    stage_info['stage_id'],
                    stage_info['name'],
                    stage_info['arrival_rate'],
                    f"{metrics['avg_reward']:.2f}",
                    f"{metrics['std_reward']:.2f}",
                    f"{metrics['avg_length']:.0f}"
                ])
    
    def plot_curriculum_progress(self) -> None:
        """Plot curriculum learning progress."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        if not self.metrics_by_stage:
            print("No metrics to plot")
            return
        
        stages = sorted(self.metrics_by_stage.keys())
        rewards = [self.metrics_by_stage[s]['avg_reward'] for s in stages]
        arrival_rates = [self.scheduler.stages[s-1].arrival_rate for s in stages]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Reward vs Stage
        ax1.plot(stages, rewards, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Curriculum Stage')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Curriculum Learning Progress')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward vs Arrival Rate
        ax2.plot(arrival_rates, rewards, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Aircraft Arrival Rate (ac/hr)')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Policy Performance vs Difficulty')
        ax2.grid(True, alpha=0.3)
        
        filepath = self.model_dir / "curriculum_progress.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        print(f"\nCurriculum progress plot saved to {filepath}")
        plt.close()


def train_with_curriculum(
    total_timesteps: Optional[int] = None,
    num_stages: int = 4,
    model_dir: str = "./evtol_training/",
    verbose: int = 1
) -> Tuple[PPO, Dict[str, Any]]:
    """
    High-level function to train with curriculum learning.
    
    Args:
        total_timesteps: Ignore (stages have own timesteps)
        num_stages: Number of curriculum stages (1-4)
        model_dir: Directory for model outputs
        verbose: Verbosity level
        
    Returns:
        (trained_model, summary)
    """
    trainer = CurriculumTrainer(model_dir=model_dir, verbose=verbose)
    model, summary = trainer.train(num_stages=num_stages)
    
    # Plot progress
    trainer.plot_curriculum_progress()
    
    return model, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Curriculum Learning Training for eVTOL Vertiport Scheduling"
    )
    parser.add_argument("--stages", type=int, default=4,
                       help="Number of curriculum stages (1-4, default 4)")
    parser.add_argument("--model-dir", type=str, default="./evtol_training/",
                       help="Directory for model output")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    model, summary = train_with_curriculum(
        num_stages=args.stages,
        model_dir=args.model_dir,
        verbose=args.verbose
    )
    
    print(f"\n{'='*70}")
    print("CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total timesteps: {summary['total_timesteps']:,}")
    print(f"Stages completed: {summary['stages_trained']}/{summary['total_stages']}")
    print(f"Final model saved to: {args.model_dir}")
