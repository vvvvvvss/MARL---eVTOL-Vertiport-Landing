"""
Comprehensive Policy Comparison Script
Compares PPO, Greedy, and FCFS schedulers across multiple metrics and scenarios
"""

import json
import glob
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from stable_baselines3 import PPO
from vertiport_rl_env import VertiportRLEnv
from baselines import FCFSScheduler, GreedyScheduler
from vertiport_env import VertiportEnv


class ComprehensiveComparison:
    """Compare multiple policies comprehensively."""
    
    def __init__(self, num_pads: int = 8):
        self.num_pads = num_pads
        self.results = {}
    
    def run_baseline_episode(
        self,
        scheduler_class,
        arrival_rate: float,
        episode_steps: int = 500
    ) -> Dict:
        """Run one episode with a baseline scheduler."""
        
        env = VertiportEnv(
            num_pads=self.num_pads,
            arrival_rate=arrival_rate,
            max_aircraft=50
        )
        scheduler = scheduler_class(env)
        
        obs, _ = env.reset()
        scheduler.reset()
        
        total_reward = 0.0
        for _ in range(episode_steps):
            actions = scheduler.select_actions(obs)
            obs, _, _, _, _ = env.step(actions)
        
        return {
            'landings': env.total_landings,
            'avg_delay': env.total_delay / max(env.total_landings, 1),
            'throughput': env.total_landings / max(env.current_time / 60.0, 0.1),
            'pad_utilization': sum(1 for p in env.pads if p.occupied) / self.num_pads,
            'violations': env.separation_violations,
            'total_reward': total_reward
        }
    
    def run_ppo_episode(
        self,
        model: PPO,
        arrival_rate: float,
        episode_steps: int = 500
    ) -> Dict:
        """Run one episode with a trained PPO model."""
        
        env = VertiportRLEnv(
            num_pads=self.num_pads,
            arrival_rate=arrival_rate,
            max_aircraft=50
        )
        
        obs, _ = env.reset()
        total_reward = 0.0
        
        for _ in range(episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        return {
            'landings': env.env.total_landings,
            'avg_delay': env.env.total_delay / max(env.env.total_landings, 1),
            'throughput': env.env.total_landings / max(env.env.current_time / 60.0, 0.1),
            'pad_utilization': sum(1 for p in env.env.pads if p.occupied) / self.num_pads,
            'violations': env.env.separation_violations,
            'total_reward': total_reward
        }
    
    def compare_all(
        self,
        ppo_model_path: str = None,
        arrival_rates: List[float] = [10, 20, 30, 40],
        episodes_per_config: int = 5,
        episode_steps: int = 500
    ) -> Dict:
        """
        Run comprehensive comparison across all policies and scenarios.
        
        Args:
            ppo_model_path: Path to trained PPO model (supports wildcards like */final_*.zip)
            arrival_rates: List of arrival rates to test
            episodes_per_config: Episodes per configuration
            episode_steps: Steps per episode
            
        Returns:
            Comparison results dictionary
        """
        
        # Load PPO model if provided
        ppo_model = None
        if ppo_model_path:
            # Handle wildcard paths
            if '*' in ppo_model_path:
                matching_files = glob.glob(ppo_model_path)
                if matching_files:
                    ppo_model_path = matching_files[0]  # Use first match
                    print(f"ℹ Resolved wildcard path to: {ppo_model_path}")
                else:
                    print(f"⚠ No files matching pattern: {ppo_model_path}")
                    ppo_model_path = None
            
            # Check if file exists
            if ppo_model_path and Path(ppo_model_path).exists():
                try:
                    ppo_model = PPO.load(ppo_model_path)
                    print(f"✓ Loaded PPO model from {ppo_model_path}\n")
                except Exception as e:
                    print(f"✗ Failed to load PPO model: {e}")
                    print(f"  File exists at: {ppo_model_path}")
                    print(f"  But loading failed: {type(e).__name__}\n")
                    ppo_model = None
            elif ppo_model_path:
                print(f"✗ Model file not found: {ppo_model_path}")
                print(f"  Searched for: {ppo_model_path}\n")
                ppo_model = None
        
        results = {}
        
        print("\n" + "=" * 100)
        print("COMPREHENSIVE POLICY COMPARISON")
        print("=" * 100)
        
        for arrival_rate in arrival_rates:
            print(f"\n\nArrival Rate: {arrival_rate} aircraft/hour")
            print("-" * 100)
            
            results[arrival_rate] = {}
            
            policies = [
                ("FCFS", lambda: self._run_multiple_episodes(
                    lambda: self.run_baseline_episode(FCFSScheduler, arrival_rate, episode_steps),
                    episodes_per_config
                )),
                ("Greedy", lambda: self._run_multiple_episodes(
                    lambda: self.run_baseline_episode(GreedyScheduler, arrival_rate, episode_steps),
                    episodes_per_config
                )),
            ]
            
            if ppo_model is not None:
                policies.append(("PPO", lambda model=ppo_model: self._run_multiple_episodes(
                    lambda m=model: self.run_ppo_episode(m, arrival_rate, episode_steps),
                    episodes_per_config
                )))
            
            for policy_name, run_fn in policies:
                print(f"  Evaluating {policy_name:8s}...", end=" ")
                episodes = run_fn()
                results[arrival_rate][policy_name] = self._aggregate_episodes(episodes)
                print("✓")
        
        return results
    
    def _run_multiple_episodes(self, episode_fn, num_episodes: int) -> List[Dict]:
        """Run multiple episodes and collect results."""
        return [episode_fn() for _ in range(num_episodes)]
    
    def _aggregate_episodes(self, episodes: List[Dict]) -> Dict:
        """Aggregate results from multiple episodes."""
        
        metrics = {}
        for key in episodes[0].keys():
            values = [ep[key] for ep in episodes]
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return metrics
    
    def print_results(self, results: Dict):
        """Print results in formatted tables."""
        
        metrics_to_show = ['landings', 'avg_delay', 'throughput', 'pad_utilization', 'violations']
        
        for arrival_rate in sorted(results.keys()):
            print(f"\n\n{'='*100}")
            print(f"ARRIVAL RATE: {arrival_rate} aircraft/hour")
            print(f"{'='*100}\n")
            
            policies = sorted(results[arrival_rate].keys())
            
            for metric in metrics_to_show:
                print(f"\n{metric.upper()}")
                print(f"  {'Policy':<12} {'Mean':<14} {'Std':<14} {'Min':<14} {'Max':<14}")
                print("  " + "-" * 54)
                
                for policy in policies:
                    m = results[arrival_rate][policy][metric]
                    mean_str = f"{m['mean']:>12.2f}"
                    std_str = f"{m['std']:>12.2f}"
                    min_str = f"{m['min']:>12.2f}"
                    max_str = f"{m['max']:>12.2f}"
                    print(f"  {policy:<12} {mean_str:<14} {std_str:<14} {min_str:<14} {max_str:<14}")
    
    def plot_comparison(self, results: Dict, output_dir: str = "./comparison_plots/"):
        """Plot comparison results."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        metrics = ['avg_delay', 'throughput', 'pad_utilization', 'landings', 'violations']
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            arrival_rates = sorted(results.keys())
            
            for policy in sorted(set(p for rates in results.values() for p in rates.keys())):
                means = []
                stds = []
                
                for ar in arrival_rates:
                    if policy in results[ar] and metric in results[ar][policy]:
                        m = results[ar][policy][metric]
                        means.append(m['mean'])
                        stds.append(m['std'])
                    else:
                        means.append(None)
                        stds.append(None)
                
                # Filter out None values
                valid_ar = [ar for i, ar in enumerate(arrival_rates) if means[i] is not None]
                valid_means = [m for m in means if m is not None]
                valid_stds = [s for s in stds if s is not None]
                
                if valid_means:
                    ax.errorbar(valid_ar, valid_means, yerr=valid_stds, marker='o', 
                               label=policy, linewidth=2, markersize=8, capsize=5)
            
            ax.set_xlabel('Arrival Rate (aircraft/hour)', fontsize=12)
            
            if metric == 'avg_delay':
                ax.set_ylabel('Average Delay (minutes)', fontsize=12)
                ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='Target (<5 min)')
            elif metric == 'throughput':
                ax.set_ylabel('Throughput (aircraft/hour)', fontsize=12)
            elif metric == 'pad_utilization':
                ax.set_ylabel('Pad Utilization (%)', fontsize=12)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            elif metric == 'landings':
                ax.set_ylabel('Total Landings', fontsize=12)
            elif metric == 'violations':
                ax.set_ylabel('Separation Violations', fontsize=12)
            
            ax.set_title(f'Policy Comparison: {metric.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            plt.tight_layout()
            
            output_path = Path(output_dir) / f"comparison_{metric}.png"
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"✓ Saved {output_path}")
    
    def save_results_csv(self, results: Dict, output_path: str = "policy_comparison_results.csv"):
        """Save results to CSV."""
        
        rows = []
        for arrival_rate in sorted(results.keys()):
            for policy in sorted(results[arrival_rate].keys()):
                for metric in sorted(results[arrival_rate][policy].keys()):
                    m = results[arrival_rate][policy][metric]
                    rows.append({
                        'Arrival_Rate': arrival_rate,
                        'Policy': policy,
                        'Metric': metric,
                        'Mean': m['mean'],
                        'Std': m['std'],
                        'Min': m['min'],
                        'Max': m['max']
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"✓ Saved results to {output_path}")


def main():
    """Main comparison script."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PPO with baselines")
    parser.add_argument("--ppo-model", type=str, default=None,
                       help="Path to trained PPO model (supports wildcards like */final_*.zip)")
    parser.add_argument("--arrival-rates", type=float, nargs='+', default=[10, 20, 30, 40],
                       help="Arrival rates to test")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Episodes per configuration")
    parser.add_argument("--steps", type=int, default=500,
                       help="Steps per episode")
    parser.add_argument("--output-dir", type=str, default="./comparison_results/",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # If no PPO model provided, find it automatically
    if not args.ppo_model:
        print("\n" + "=" * 100)
        print("PPO MODEL NOT PROVIDED")
        print("=" * 100)
        matching = glob.glob("evtol_training/*/final_evtol_ppo.zip")
        if matching:
            args.ppo_model = matching[0]
            print(f"✓ Found PPO model: {args.ppo_model}\n")
        else:
            print("⚠ No trained PPO model found in evtol_training/")
            print("  Run 'python train_ppo.py' to train a model first\n")
    
    print("Starting comprehensive comparison...\n")
    
    comparator = ComprehensiveComparison()
    results = comparator.compare_all(
        ppo_model_path=args.ppo_model,
        arrival_rates=args.arrival_rates,
        episodes_per_config=args.episodes,
        episode_steps=args.steps
    )
    
    # Print results
    comparator.print_results(results)
    
    # Plot and save
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    comparator.plot_comparison(results, args.output_dir)
    comparator.save_results_csv(results, str(Path(args.output_dir) / "comparison_results.csv"))
    
    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)
    print(f"Results saved to: {args.output_dir}")
    
    # Summary
    if results:
        print(f"\n📊 Summary:")
        print(f"  - Tested {len(results)} arrival rates")
        for ar in sorted(results.keys()):
            policies_tested = list(results[ar].keys())
            print(f"  - {ar} ac/hr: {', '.join(policies_tested)}")



if __name__ == "__main__":
    main()
