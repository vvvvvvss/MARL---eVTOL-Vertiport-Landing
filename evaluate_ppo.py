"""
Evaluation and Learning Curve Tracking for PPO Training
Provides tools to evaluate trained models and visualize training progress
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from stable_baselines3 import PPO
from vertiport_rl_env import VertiportRLEnv
from baselines import FCFSScheduler, GreedyScheduler
from vertiport_env import VertiportEnv


class PolicyEvaluator:
    """Evaluate different policies on the vertiport problem."""
    
    def __init__(self, num_pads: int = 8):
        self.num_pads = num_pads
        self.results = {}
    
    def evaluate_policy(
        self,
        policy_name: str,
        policy_fn,
        arrival_rate: float,
        num_episodes: int = 5,
        episode_steps: int = 500
    ) -> Dict:
        """
        Evaluate a policy.
        
        Args:
            policy_name: Name of the policy
            policy_fn: Function that returns action given observation (or env)
            arrival_rate: Aircraft arrival rate
            num_episodes: Number of episodes to run
            episode_steps: Steps per episode
            
        Returns:
            Metrics dict
        """
        
        metrics = {
            'landings': [],
            'avg_delay': [],
            'throughput': [],
            'pad_utilization': [],
            'violations': [],
            'battery_violations': [],  # Aircraft that ran out of battery
            'total_reward': []
        }
        
        for episode in range(num_episodes):
            # Reset environment based on baseline type
            if policy_name.upper() in ['FCFS', 'GREEDY']:
                # Use regular environment for baselines
                env = VertiportEnv(
                    num_pads=self.num_pads,
                    arrival_rate=arrival_rate,
                    max_aircraft=50
                )
                scheduler = FCFSScheduler(env) if policy_name.upper() == 'FCFS' else GreedyScheduler(env)
                obs, info = env.reset()
                scheduler.reset()
                
                episode_reward = 0.0
                for step in range(episode_steps):
                    actions = scheduler.select_actions(obs)
                    obs, rewards, terminated, truncated, info = env.step(actions)
                    episode_reward += sum(rewards.values())
                
            else:
                # Use RL environment
                env = VertiportRLEnv(
                    num_pads=self.num_pads,
                    arrival_rate=arrival_rate,
                    max_aircraft=50
                )
                obs, info = env.reset()
                
                episode_reward = 0.0
                for step in range(episode_steps):
                    # Call policy function (e.g., model.predict)
                    # RL policy handling
                    if hasattr(policy_fn, "predict"):
                        action, _ = policy_fn.predict(obs, deterministic=True)
                    else:
                        action = policy_fn(obs)


                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
            
            # Record metrics
            # Handle wrapper vs base env
            if hasattr(env, "get_state_dict"):
                state = env.get_state_dict()
            else:
                state = {
                    "total_landings": env.total_landings,
                    "total_delay": env.total_delay,
                    "separation_violations": env.separation_violations,
                    "current_time": env.current_time
                }
            
            landings = state["total_landings"]
            total_delay = state["total_delay"]
            violations = state["separation_violations"]
            current_time = state["current_time"]
            
            metrics['landings'].append(landings)
            metrics['avg_delay'].append(total_delay / max(landings, 1))
            metrics['throughput'].append(
                landings / max(current_time / 60.0, 0.1)
            )
            metrics['violations'].append(violations)
            
            metrics['total_reward'].append(episode_reward)
        
        # Aggregate metrics
        aggregated = {
            'landings': {
                'mean': float(np.mean(metrics['landings'])),
                'std': float(np.std(metrics['landings']))
            },
            'avg_delay': {
                'mean': float(np.mean(metrics['avg_delay'])),
                'std': float(np.std(metrics['avg_delay']))
            },
            'throughput': {
                'mean': float(np.mean(metrics['throughput'])),
                'std': float(np.std(metrics['throughput']))
            },
            'pad_utilization': {
                'mean': float(np.mean(metrics['pad_utilization'])),
                'std': float(np.std(metrics['pad_utilization']))
            },
            'violations': {
                'mean': float(np.mean(metrics['violations'])),
                'std': float(np.std(metrics['violations']))
            },
            'total_reward': {
                'mean': float(np.mean(metrics['total_reward'])),
                'std': float(np.std(metrics['total_reward']))
            }
        }
        
        self.results[policy_name] = aggregated
        return aggregated
    
    def compare_policies(
        self,
        policies: Dict[str, callable],
        arrival_rates: List[float] = [10, 20, 30],
        num_episodes: int = 3
    ) -> Dict:
        """
        Compare multiple policies across arrival rates.
        
        Args:
            policies: Dict of {policy_name: policy_function}
            arrival_rates: List of arrival rates to test
            num_episodes: Episodes per configuration
            
        Returns:
            Comparison results
        """
        
        comparison = {}
        
        for arrival_rate in arrival_rates:
            print(f"\nArrival Rate: {arrival_rate} ac/hr")
            print("-" * 80)
            
            comparison[arrival_rate] = {}
            
            for policy_name, policy_fn in policies.items():
                print(f"  Evaluating {policy_name}...", end=" ")
                results = self.evaluate_policy(
                    policy_name,
                    policy_fn,
                    arrival_rate,
                    num_episodes=num_episodes
                )
                comparison[arrival_rate][policy_name] = results
                print("✓")
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print comparison results in formatted table."""
        
        for arrival_rate in sorted(comparison.keys()):
            print(f"\n{'='*80}")
            print(f"ARRIVAL RATE: {arrival_rate} aircraft/hour")
            print(f"{'='*80}\n")
            
            policies = comparison[arrival_rate]
            
            # Create formatted table
            metrics = ['landings', 'avg_delay', 'throughput', 'pad_utilization', 'violations']
            
            for metric in metrics:
                print(f"\n{metric.upper()}:")
                print(f"  {'Policy':<15} {'Mean':<12} {'Std':<12}")
                print("  " + "-" * 40)
                
                for policy_name in sorted(policies.keys()):
                    if metric in policies[policy_name]:
                        mean = policies[policy_name][metric]['mean']
                        std = policies[policy_name][metric]['std']
                        
                        if metric == 'avg_delay':
                            print(f"  {policy_name:<15} {mean:>10.2f} min  {std:>10.2f}")
                        elif metric == 'throughput':
                            print(f"  {policy_name:<15} {mean:>10.1f} ac/hr {std:>10.1f}")
                        elif metric == 'pad_utilization':
                            print(f"  {policy_name:<15} {mean:>10.1%}     {std:>10.1%}")
                        else:
                            print(f"  {policy_name:<15} {mean:>10.1f}     {std:>10.1f}")
    
    def save_comparison_csv(self, comparison: Dict, output_path: str = "policy_comparison.csv"):
        """Save comparison results to CSV."""
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Arrival_Rate', 'Policy', 'Metric', 'Mean', 'Std'])
            
            for arrival_rate in sorted(comparison.keys()):
                for policy_name in sorted(comparison[arrival_rate].keys()):
                    for metric in sorted(comparison[arrival_rate][policy_name].keys()):
                        mean = comparison[arrival_rate][policy_name][metric]['mean']
                        std = comparison[arrival_rate][policy_name][metric]['std']
                        writer.writerow([arrival_rate, policy_name, metric, mean, std])
        
        print(f"✓ Comparison saved to {output_path}")


class TrainingPlotter:
    """Plot training curves from TensorBoard logs."""
    
    @staticmethod
    def plot_training_curves(log_dir: str, output_dir: str = None):
        """
        Plot training curves from logs.
        
        Args:
            log_dir: Path to training log directory
            output_dir: Directory to save plots (defaults to log_dir)
        """
        
        if output_dir is None:
            output_dir = log_dir
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Try to load events from TensorBoard
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            event_dir = Path(log_dir) / "tensorboard"
            if event_dir.exists():
                ea = EventAccumulator(str(event_dir))
                ea.Reload()
                
                # Plot episode reward
                if 'rollout/ep_rew_mean' in ea.tags()['scalars']:
                    episodes, steps, rewards = zip(*ea.Scalars('rollout/ep_rew_mean'))
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(steps, rewards, linewidth=2, label='Mean Episode Reward')
                    ax.set_xlabel('Training Steps')
                    ax.set_ylabel('Episode Reward')
                    ax.set_title('Training Progress: Episode Reward')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(Path(output_dir) / "training_reward.png", dpi=150)
                    plt.close()
                    print("✓ Saved training_reward.png")
        
        except ImportError:
            print("⚠ TensorBoard not available, skipping event file parsing")
        
        # Try to load CSV logs
        csv_log_path = Path(log_dir) / "logs" / "progress.csv"
        if csv_log_path.exists():
            plot_csv_training(csv_log_path, output_dir)


def plot_csv_training(csv_path: str, output_dir: str):
    """Plot training metrics from CSV log."""
    
    data = {'step': [], 'reward': [], 'ep_len': []}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'step' in row and row['step']:
                    data['step'].append(int(row['step']))
                if 'rollout/ep_rew_mean' in row and row['rollout/ep_rew_mean']:
                    data['reward'].append(float(row['rollout/ep_rew_mean']))
    except:
        pass
    
    if data['step'] and data['reward']:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['step'], data['reward'], linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Training Progress: Mean Episode Reward')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "training_reward_csv.png", dpi=150)
        plt.close()
        print(f"✓ Saved training_reward_csv.png")


def evaluate_ppo_model(
    model_path: str,
    arrival_rate: float = 20.0,
    num_episodes: int = 5,
    compare_to_baselines: bool = True
):
    """
    Evaluate a trained PPO model.
    
    Args:
        model_path: Path to saved PPO model
        arrival_rate: Arrival rate for evaluation
        num_episodes: Number of episodes
        compare_to_baselines: Compare to FCFS and Greedy
    """
    
    print("=" * 80)
    print("PPO MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Arrival Rate: {arrival_rate} ac/hr")
    print()
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Evaluate PPO
    evaluator = PolicyEvaluator()
    ppo_results = evaluator.evaluate_policy(
        "PPO",
        model,
        arrival_rate,
        num_episodes=num_episodes
    )
    
    print("\n" + "=" * 80)
    print("PPO RESULTS")
    print("=" * 80)
    print(f"Total Landings:      {ppo_results['landings']['mean']:.1f} ± {ppo_results['landings']['std']:.1f}")
    print(f"Avg Delay:           {ppo_results['avg_delay']['mean']:.2f} ± {ppo_results['avg_delay']['std']:.2f} min")
    print(f"Throughput:          {ppo_results['throughput']['mean']:.1f} ± {ppo_results['throughput']['std']:.1f} ac/hr")
    print(f"Pad Utilization:     {ppo_results['pad_utilization']['mean']:.1%} ± {ppo_results['pad_utilization']['std']:.1%}")
    print(f"Separation Violations: {ppo_results['violations']['mean']:.2f} ± {ppo_results['violations']['std']:.2f}")
    
    if compare_to_baselines:
        # Compare to baselines
        print("\n" + "=" * 80)
        print("COMPARING TO BASELINES")
        print("=" * 80)
        
        comparison = evaluator.compare_policies(
            {
                'FCFS': None,  # Will use scheduler
                'Greedy': None,
                'PPO': model
            },
            arrival_rates=[arrival_rate],
            num_episodes=num_episodes
        )
        
        evaluator.print_comparison(comparison)
        
        # Save results
        evaluator.save_comparison_csv(comparison, "ppo_vs_baselines.csv")


def main():
    """Main evaluation script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--model", type=str, required=False,
                       help="Path to trained PPO model")
    parser.add_argument("--arrival-rate", type=float, default=20.0,
                       help="Aircraft arrival rate")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of evaluation episodes")
    parser.add_argument("--log-dir", type=str, default="./evtol_training/",
                       help="Training log directory")
    parser.add_argument("--plot", action="store_true",
                       help="Plot training curves")
    
    args = parser.parse_args()
    
    if args.plot:
        TrainingPlotter.plot_training_curves(args.log_dir)
        print("✓ Training curves plotted")
    
    if args.model:
        evaluate_ppo_model(
            args.model,
            arrival_rate=args.arrival_rate,
            num_episodes=args.episodes,
            compare_to_baselines=True
        )
    else:
        print("Provide a model path with --model")
        
        # Try to find latest model
        latest = max(Path(args.log_dir).glob("**/final_*.zip"), default=None)
        if latest:
            print(f"\nGuessed model path: {latest}")
            evaluate_ppo_model(str(latest))


if __name__ == "__main__":
    main()
