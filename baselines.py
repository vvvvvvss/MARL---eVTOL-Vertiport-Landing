"""
FCFS (First-Come-First-Served) Baseline Scheduler
Implements traditional vertiport scheduling policy for comparison
"""

import numpy as np
from typing import Dict, List
from vertiport_env import VertiportEnv, Aircraft


class FCFSScheduler:
    """
    First-Come-First-Served Scheduler
    
    Policy:
    - Aircraft are served in order of arrival
    - Assign first available pad
    - Priority queue for low-battery aircraft
    """
    
    def __init__(self, env: VertiportEnv):
        self.env = env
        self.arrival_order = {}  # aircraft_id -> arrival_time
        
    def reset(self):
        """Reset scheduler state"""
        self.arrival_order = {}
        
    def select_actions(self, observations: Dict) -> Dict[int, int]:
        """
        Select actions for all aircraft using FCFS policy
        
        Returns:
            actions: Dict mapping aircraft_id to action (pad or hold)
        """
        actions = {}
        
        # Track new arrivals
        for aircraft in self.env.aircraft:
            if aircraft.id not in self.arrival_order:
                self.arrival_order[aircraft.id] = self.env.current_time
        
        # Sort aircraft by priority
        # 1. Low battery aircraft first (< 20%)
        # 2. Then by arrival order (FCFS)
        aircraft_sorted = sorted(
            self.env.aircraft,
            key=lambda a: (
                0 if a.battery_soc < 20 else 1,  # Low battery first
                self.arrival_order.get(a.id, float('inf'))  # Then FCFS
            )
        )
        
        # Assign pads in order
        for aircraft in aircraft_sorted:
            action_mask = self.env.get_action_mask(aircraft)
            
            # Try to land on first available pad
            landed = False
            for pad_id in range(self.env.num_pads):
                if action_mask[pad_id]:
                    actions[aircraft.id] = pad_id
                    landed = True
                    break
            
            # If no pad available, hold
            if not landed:
                actions[aircraft.id] = self.env.num_pads  # Hold action
        
        return actions


class GreedyScheduler:
    """
    Greedy Scheduler - Smarter baseline
    
    Policy:
    - Prioritize low-battery aircraft
    - Assign to nearest available pad
    - Consider passenger priority as tiebreaker
    """
    
    def __init__(self, env: VertiportEnv):
        self.env = env
        
    def reset(self):
        """Reset scheduler state"""
        pass
        
    def select_actions(self, observations: Dict) -> Dict[int, int]:
        """
        Select actions using greedy policy
        
        Priority score = battery_urgency * 100 + passenger_priority * 10 - distance_to_pad
        """
        actions = {}
        assigned_pads = set()
        
        # Calculate priority scores
        aircraft_priorities = []
        for aircraft in self.env.aircraft:
            # Battery urgency (higher when lower battery)
            battery_urgency = max(0, (30 - aircraft.battery_soc) / 30)
            
            # Find nearest available pad
            min_distance = float('inf')
            best_pad = None
            action_mask = self.env.get_action_mask(aircraft)
            
            for pad_id in range(self.env.num_pads):
                if action_mask[pad_id] and pad_id not in assigned_pads:
                    pad = self.env.pads[pad_id]
                    distance = np.linalg.norm(aircraft.position[:2] - pad.position)
                    if distance < min_distance:
                        min_distance = distance
                        best_pad = pad_id
            
            # Calculate priority score
            priority_score = (
                battery_urgency * 100 +
                aircraft.passenger_priority * 10 -
                (min_distance / 1000.0 if min_distance < float('inf') else 100)
            )
            
            aircraft_priorities.append((priority_score, aircraft, best_pad))
        
        # Sort by priority and assign
        aircraft_priorities.sort(key=lambda x: x[0], reverse=True)
        
        for priority_score, aircraft, best_pad in aircraft_priorities:
            if best_pad is not None and best_pad not in assigned_pads:
                actions[aircraft.id] = best_pad
                assigned_pads.add(best_pad)
            else:
                # Hold if no pad available
                actions[aircraft.id] = self.env.num_pads
        
        return actions


def run_baseline_comparison(num_episodes: int = 5, episode_steps: int = 200):
    """
    Compare FCFS and Greedy schedulers
    """
    print("=" * 60)
    print("BASELINE SCHEDULER COMPARISON")
    print("=" * 60)
    
    arrival_rates = [10, 20, 30]  # aircraft per hour
    
    for arrival_rate in arrival_rates:
        print(f"\n\n{'='*60}")
        print(f"ARRIVAL RATE: {arrival_rate} aircraft/hour")
        print(f"{'='*60}\n")
        
        results = {'FCFS': [], 'Greedy': []}
        
        for scheduler_name in ['FCFS', 'Greedy']:
            print(f"\nTesting {scheduler_name} Scheduler...")
            
            episode_metrics = {
                'landings': [],
                'avg_delay': [],
                'throughput': [],
                'pad_utilization': [],
                'violations': []
            }
            
            for episode in range(num_episodes):
                env = VertiportEnv(
                    num_pads=8,
                    arrival_rate=arrival_rate,
                    max_aircraft=50
                )
                
                if scheduler_name == 'FCFS':
                    scheduler = FCFSScheduler(env)
                else:
                    scheduler = GreedyScheduler(env)
                
                obs, info = env.reset()
                scheduler.reset()
                
                for step in range(episode_steps):
                    actions = scheduler.select_actions(obs)
                    obs, rewards, terminated, truncated, info = env.step(actions)
                
                # Record metrics
                episode_metrics['landings'].append(env.total_landings)
                episode_metrics['avg_delay'].append(
                    env.total_delay / max(env.total_landings, 1)
                )
                episode_metrics['throughput'].append(
                    env.total_landings / (env.current_time / 60.0)  # per hour
                )
                episode_metrics['pad_utilization'].append(info['pad_utilization'])
                episode_metrics['violations'].append(env.separation_violations)
            
            # Calculate averages
            avg_metrics = {
                key: np.mean(values) for key, values in episode_metrics.items()
            }
            std_metrics = {
                key: np.std(values) for key, values in episode_metrics.items()
            }
            
            results[scheduler_name] = avg_metrics
            
            print(f"\n  Results (avg ± std over {num_episodes} episodes):")
            print(f"    Total Landings: {avg_metrics['landings']:.1f} ± {std_metrics['landings']:.1f}")
            print(f"    Avg Landing Delay: {avg_metrics['avg_delay']:.2f} ± {std_metrics['avg_delay']:.2f} min")
            print(f"    Throughput: {avg_metrics['throughput']:.1f} ± {std_metrics['throughput']:.1f} aircraft/hour")
            print(f"    Pad Utilization: {avg_metrics['pad_utilization']:.1%} ± {std_metrics['pad_utilization']:.1%}")
            print(f"    Separation Violations: {avg_metrics['violations']:.2f} ± {std_metrics['violations']:.2f}")
        
        # Compare
        print(f"\n  {'='*50}")
        print(f"  COMPARISON:")
        print(f"  {'='*50}")
        improvement_delay = (
            (results['FCFS']['avg_delay'] - results['Greedy']['avg_delay']) /
            results['FCFS']['avg_delay'] * 100
        )
        improvement_throughput = (
            (results['Greedy']['throughput'] - results['FCFS']['throughput']) /
            results['FCFS']['throughput'] * 100
        )
        
        print(f"    Greedy reduces delay by: {improvement_delay:.1f}%")
        print(f"    Greedy improves throughput by: {improvement_throughput:.1f}%")


if __name__ == "__main__":
    run_baseline_comparison(num_episodes=5, episode_steps=200)
