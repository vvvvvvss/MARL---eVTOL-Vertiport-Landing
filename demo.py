"""
MARL eVTOL Vertiport Scheduling - Prototype Demo
Week 1-2 Deliverable: Environment + Baselines

This script demonstrates the core simulation environment and baseline schedulers.
"""

import numpy as np
from vertiport_env import VertiportEnv
from baselines import FCFSScheduler, GreedyScheduler
from visualization import VertiportVisualizer
import matplotlib.pyplot as plt


def demo_environment():
    """Demonstrate the basic environment functionality"""
    print("\n" + "="*70)
    print("DEMO 1: ENVIRONMENT BASICS")
    print("="*70)
    
    env = VertiportEnv(num_pads=8, arrival_rate=15, max_aircraft=30)
    obs, info = env.reset()
    
    print(f"\n✓ Environment initialized")
    print(f"  • Number of pads: {env.num_pads}")
    print(f"  • Arrival rate: {env.arrival_rate} aircraft/hour")
    print(f"  • Initial aircraft: {info['num_aircraft']}")
    print(f"  • Observation dimension: {list(obs.values())[0].shape[0]}")
    
    # Show state space
    print(f"\n✓ State Space Breakdown:")
    print(f"  • Aircraft state: 7 features (position, velocity, heading, battery, priority)")
    print(f"  • Pad states: {env.num_pads * 3} features (available, cooldown, distance)")
    print(f"  • Altitude rings: 4 features (aircraft count per ring)")
    print(f"  • Total: {7 + env.num_pads * 3 + 4} dimensions")
    
    # Show action space
    print(f"\n✓ Action Space:")
    print(f"  • Actions 0-7: Land on pad 0-7")
    print(f"  • Action 8: Hold/wait at current altitude")
    print(f"  • Action masking enforces safety constraints")
    
    # Demonstrate action masking
    if env.aircraft:
        aircraft = env.aircraft[0]
        mask = env.get_action_mask(aircraft)
        print(f"\n✓ Action Mask Example (Aircraft {aircraft.id}):")
        print(f"  • Valid pads: {np.where(mask[:env.num_pads])[0].tolist()}")
        print(f"  • Can hold: {mask[-1]}")
        print(f"  • Battery: {aircraft.battery_soc:.1f}%")
        print(f"  • Altitude: {aircraft.position[2]:.0f}m")
    
    return env


def demo_baseline_comparison():
    """Compare FCFS and Greedy schedulers"""
    print("\n" + "="*70)
    print("DEMO 2: BASELINE SCHEDULER COMPARISON")
    print("="*70)
    
    arrival_rate = 20
    num_episodes = 3
    episode_steps = 150
    
    print(f"\nConfiguration:")
    print(f"  • Arrival rate: {arrival_rate} aircraft/hour")
    print(f"  • Episodes: {num_episodes}")
    print(f"  • Steps per episode: {episode_steps}")
    
    results = {}
    
    for scheduler_name in ['FCFS', 'Greedy']:
        print(f"\n{'─'*70}")
        print(f"Testing {scheduler_name} Scheduler...")
        print(f"{'─'*70}")
        
        metrics = {
            'landings': [],
            'delays': [],
            'throughput': [],
            'utilization': []
        }
        
        for ep in range(num_episodes):
            env = VertiportEnv(num_pads=8, arrival_rate=arrival_rate)
            
            if scheduler_name == 'FCFS':
                scheduler = FCFSScheduler(env)
            else:
                scheduler = GreedyScheduler(env)
            
            obs, info = env.reset()
            scheduler.reset()
            
            for step in range(episode_steps):
                actions = scheduler.select_actions(obs)
                obs, rewards, terminated, truncated, info = env.step(actions)
            
            metrics['landings'].append(env.total_landings)
            metrics['delays'].append(env.total_delay / max(env.total_landings, 1))
            metrics['throughput'].append(env.total_landings / (env.current_time / 60))
            metrics['utilization'].append(info['pad_utilization'])
            
            print(f"  Episode {ep+1}: {env.total_landings} landings, "
                  f"{metrics['delays'][-1]:.2f} min delay, "
                  f"{metrics['utilization'][-1]:.1%} utilization")
        
        results[scheduler_name] = {
            key: (np.mean(values), np.std(values))
            for key, values in metrics.items()
        }
    
    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<25} {'FCFS':<20} {'Greedy':<20} {'Improvement'}")
    print(f"{'-'*70}")
    
    for metric in ['landings', 'delays', 'throughput', 'utilization']:
        fcfs_mean, fcfs_std = results['FCFS'][metric]
        greedy_mean, greedy_std = results['Greedy'][metric]
        
        if metric == 'delays':
            improvement = (fcfs_mean - greedy_mean) / fcfs_mean * 100
            better = improvement > 0
        else:
            improvement = (greedy_mean - fcfs_mean) / fcfs_mean * 100
            better = improvement > 0
        
        metric_names = {
            'landings': 'Total Landings',
            'delays': 'Avg Delay (min)',
            'throughput': 'Throughput (ac/hr)',
            'utilization': 'Pad Utilization'
        }
        
        improvement_str = f"{'↑' if better else '↓'} {abs(improvement):.1f}%"
        
        if metric == 'utilization':
            fcfs_str = f"{fcfs_mean:.1%}±{fcfs_std:.1%}"
            greedy_str = f"{greedy_mean:.1%}±{greedy_std:.1%}"
            print(f"{metric_names[metric]:<25} {fcfs_str:<20} {greedy_str:<20} {improvement_str}")
        else:
            fcfs_str = f"{fcfs_mean:.1f}±{fcfs_std:.1f}"
            greedy_str = f"{greedy_mean:.1f}±{greedy_std:.1f}"
            print(f"{metric_names[metric]:<25} {fcfs_str:<20} {greedy_str:<20} {improvement_str}")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}")
    print(f"✓ Greedy scheduler improves upon FCFS by prioritizing:")
    print(f"  • Low-battery aircraft (battery urgency)")
    print(f"  • High-priority passengers")
    print(f"  • Nearest available pad (reduces travel time)")
    print(f"\n✓ This demonstrates the potential for learning-based methods")
    print(f"  • MARL can learn even better policies through experience")
    print(f"  • Target: <5 min delay (vs FCFS baseline ~15-20 min)")
    
    return results


def demo_visualization():
    """Create visualization demo"""
    print("\n" + "="*70)
    print("DEMO 3: VISUALIZATION")
    print("="*70)
    
    print("\nCreating visual dashboard...")
    
    env = VertiportEnv(num_pads=8, arrival_rate=25)
    scheduler = GreedyScheduler(env)
    visualizer = VertiportVisualizer(env)
    
    obs, info = env.reset()
    scheduler.reset()
    
    print("Running simulation...")
    for step in range(80):
        actions = scheduler.select_actions(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 20 == 0:
            print(f"  Step {step}: {len(env.aircraft)} aircraft in airspace, "
                  f"{env.total_landings} total landings")
    
    visualizer.render_frame(save_path='vertiport_dashboard.png')
    print(f"\n✓ Dashboard saved to: vertiport_dashboard.png")
    print(f"\n  The dashboard includes:")
    print(f"  • Top-down view of vertiport with aircraft and pads")
    print(f"  • Real-time metrics panel")
    print(f"  • Delay history plot")
    print(f"  • Pad utilization over time")
    
    print(f"\n✓ Final Statistics:")
    print(f"  • Total landings: {env.total_landings}")
    print(f"  • Average delay: {env.total_delay / max(env.total_landings, 1):.2f} min")
    print(f"  • Throughput: {env.total_landings / (env.current_time / 60):.1f} aircraft/hour")
    print(f"  • Pad utilization: {info['pad_utilization']:.1%}")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" MARL EVTOL VERTIPORT SCHEDULING - PROTOTYPE DEMONSTRATION")
    print(" Week 1-2 Deliverable: Core Environment + Baseline Schedulers")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Demo 1: Environment basics
    env = demo_environment()
    
    # Demo 2: Baseline comparison
    results = demo_baseline_comparison()
    
    # Demo 3: Visualization
    demo_visualization()
    
    print("\n" + "="*70)
    print("NEXT STEPS (Week 3-4):")
    print("="*70)
    print("✓ Implement Independent PPO training")
    print("✓ Add Ray RLlib integration")
    print("✓ Train on low-density scenarios (10 arrivals/hour)")
    print("✓ Validate convergence and learning curves")
    print("\n" + "="*70)
    print("PROTOTYPE COMPLETE ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
