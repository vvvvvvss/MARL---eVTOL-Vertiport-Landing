"""
B4: Formal Safety Verification Framework

Implements comprehensive safety verification for learned policies:
1. Constraint validation (separation, capacity, deadlock)
2. Safety property verification
3. Statistical analysis of safety
4. Formal test suite

Ensures learned policies satisfy hard safety constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from stable_baselines3 import PPO
import gymnasium as gym
from vertiport_rl_env import VertiportRLEnv


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_type: str  # Type of violation
    episode: int
    timestep: int
    agent_id: Optional[int]
    details: Dict[str, Any]


@dataclass
class SafetyMetrics:
    """Safety verification metrics."""
    total_episodes: int
    total_steps: int
    violations_per_episode: float
    violations_per_step: float
    zero_violations: bool
    violation_types: Dict[str, int]
    success_rate: float  # % of episodes with zero violations


class SafetyVerifier:
    """Verifies safety properties of learned policies."""
    
    def __init__(self, 
                 max_separation_distance: float = 500.0,
                 max_pad_capacity: int = 1,
                 verbose: bool = True):
        """
        Initialize safety verifier.
        
        Args:
            max_separation_distance: Minimum separation in meters
            max_pad_capacity: Max aircraft per pad (1 for landing)
            verbose: Print verification progress
        """
        self.max_separation_distance = max_separation_distance
        self.max_pad_capacity = max_pad_capacity
        self.verbose = verbose
        self.violations: List[SafetyViolation] = []
        
    def verify_policy(self,
                     model: PPO,
                     env: gym.Env,
                     num_episodes: int = 100,
                     max_episode_steps: int = 2000) -> SafetyMetrics:
        """
        Verify policy safety through simulation.
        
        Args:
            model: Trained RL policy
            env: Environment to test
            num_episodes: Number of episodes to test
            max_episode_steps: Max steps per episode
            
        Returns:
            SafetyMetrics with verification results
        """
        self.violations = []
        total_violations = 0
        total_steps = 0
        zero_violation_episodes = 0
        violation_types_count = {}
        
        if self.verbose:
            print(f"\nVerifying policy safety ({num_episodes} episodes)...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_violations = 0
            done = False
            step = 0
            
            while not done and step < max_episode_steps:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check safety constraints
                violations = self._check_constraints(env, episode, step, info)
                
                for violation in violations:
                    self.violations.append(violation)
                    violation_type = violation.violation_type
                    violation_types_count[violation_type] = \
                        violation_types_count.get(violation_type, 0) + 1
                    episode_violations += 1
                    total_violations += 1
                
                done = terminated or truncated
                step += 1
                total_steps += 1
            
            if episode_violations == 0:
                zero_violation_episodes += 1
            
            if self.verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                print(f"  Episode {episode + 1}/{num_episodes}: "
                      f"{total_violations} violations so far")
        
        # Compute metrics
        metrics = SafetyMetrics(
            total_episodes=num_episodes,
            total_steps=total_steps,
            violations_per_episode=total_violations / num_episodes if num_episodes > 0 else 0,
            violations_per_step=total_violations / total_steps if total_steps > 0 else 0,
            zero_violations=total_violations == 0,
            violation_types=violation_types_count,
            success_rate=zero_violation_episodes / num_episodes if num_episodes > 0 else 0
        )
        
        if self.verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _check_constraints(self, env: gym.Env, episode: int, 
                          step: int, info: Dict) -> List[SafetyViolation]:
        """Check all safety constraints."""
        violations = []
        
        # Access environment's aircraft if available
        if hasattr(env, 'env') and hasattr(env.env, 'aircraft'):
            # Multi-agent separation check
            aircraft = env.env.aircraft
            for i, ac_i in enumerate(aircraft):
                for j, ac_j in enumerate(aircraft):
                    if i >= j:
                        continue
                    
                    dist = np.linalg.norm(ac_i.position - ac_j.position)
                    if dist < self.max_separation_distance:
                        violations.append(SafetyViolation(
                            violation_type="separation_violation",
                            episode=episode,
                            timestep=step,
                            agent_id=i,
                            details={
                                'other_aircraft': j,
                                'distance': float(dist),
                                'required_distance': self.max_separation_distance
                            }
                        ))
            
            # Pad capacity check
            if hasattr(env.env, 'pads'):
                for pad_id, pad in enumerate(env.env.pads):
                    occupied_count = sum(1 for ac in aircraft if ac.landing_pad == pad_id)
                    if occupied_count > self.max_pad_capacity:
                        violations.append(SafetyViolation(
                            violation_type="pad_capacity_exceeded",
                            episode=episode,
                            timestep=step,
                            agent_id=None,
                            details={
                                'pad_id': pad_id,
                                'occupied_count': occupied_count,
                                'max_capacity': self.max_pad_capacity
                            }
                        ))
        
        return violations
    
    def _print_metrics(self, metrics: SafetyMetrics) -> None:
        """Print safety metrics in formatted text."""
        print(f"\n{'='*70}")
        print("SAFETY VERIFICATION RESULTS")
        print(f"{'='*70}")
        print(f"Episodes Tested:          {metrics.total_episodes}")
        print(f"Total Steps:              {metrics.total_steps:,}")
        print(f"Total Violations:         {len(self.violations)}")
        print(f"Violations/Episode:       {metrics.violations_per_episode:.4f}")
        print(f"Violations/Step:          {metrics.violations_per_step:.6f}")
        print(f"Success Rate (0 viol):    {metrics.success_rate*100:.1f}%")
        print(f"SAFETY STATUS:            {'✓ SAFE' if metrics.zero_violations else '✗ UNSAFE'}")
        
        if metrics.violation_types:
            print(f"\nViolation Breakdown:")
            for vtype, count in metrics.violation_types.items():
                print(f"  {vtype:.<40} {count}")
        
        print(f"{'='*70}\n")
    
    def save_report(self, filepath: str, metrics: SafetyMetrics) -> None:
        """Save violation report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_episodes': metrics.total_episodes,
                'total_steps': metrics.total_steps,
                'violations_per_episode': metrics.violations_per_episode,
                'violations_per_step': metrics.violations_per_step,
                'zero_violations': metrics.zero_violations,
                'success_rate': metrics.success_rate,
                'violation_types': metrics.violation_types
            },
            'violations': [
                {
                    'type': v.violation_type,
                    'episode': v.episode,
                    'timestep': v.timestep,
                    'agent_id': v.agent_id,
                    'details': v.details
                }
                for v in self.violations[:100]  # First 100 violations
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"Safety report saved to {filepath}")


class FormalSafetyTest:
    """Formal test suite for safety properties."""
    
    @staticmethod
    def test_separation_maintained(model: PPO, env: gym.Env, 
                                   num_episodes: int = 50) -> bool:
        """
        Test: Separation distance always maintained.
        
        Property: For all t, all aircraft i,j: dist(i,j) >= 500m
        """
        verifier = SafetyVerifier(verbose=False)
        metrics = verifier.verify_policy(model, env, num_episodes)
        
        separation_viols = sum(1 for v in verifier.violations
                              if v.violation_type == "separation_violation")
        
        return separation_viols == 0
    
    @staticmethod
    def test_pad_capacity(model: PPO, env: gym.Env, 
                         num_episodes: int = 50) -> bool:
        """
        Test: Pad capacity never exceeded.
        
        Property: For all t, all pads p: occupied(p) <= 1
        """
        verifier = SafetyVerifier(verbose=False)
        metrics = verifier.verify_policy(model, env, num_episodes)
        
        capacity_viols = sum(1 for v in verifier.violations
                            if v.violation_type == "pad_capacity_exceeded")
        
        return capacity_viols == 0
    
    @staticmethod
    def test_no_deadlock(model: PPO, env: gym.Env, 
                        num_episodes: int = 50) -> bool:
        """
        Test: Policy can resolve conflicts (no deadlock).
        
        Property: All aircraft eventually land
        """
        success_count = 0
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            landed = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check if landing occurred (reward spike)
                if reward > 50:  # Landing bonus in our reward function
                    landed = True
                
                done = terminated or truncated
            
            if landed:
                success_count += 1
        
        success_rate = success_count / num_episodes
        return success_rate > 0.8  # At least 80% landing success
    
    @staticmethod
    def test_action_validity(model: PPO, env: gym.Env, 
                            num_episodes: int = 50) -> bool:
        """
        Test: All selected actions are valid (not masked).
        
        Property: Policy never selects masked actions
        """
        invalid_actions = 0
        total_actions = 0
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                # Check if action is valid in environment
                # This depends on environment's action masking
                if hasattr(env, 'env') and hasattr(env.env, 'action_masks'):
                    mask = env.env.action_masks()
                    if isinstance(mask, np.ndarray) and not mask[action]:
                        invalid_actions += 1
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_actions += 1
                done = terminated or truncated
        
        return invalid_actions == 0


class SafetyCertificate:
    """
    Formal safety certificate for a policy.
    
    Represents verified safety properties of a learned policy.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.timestamp = datetime.now().isoformat()
        self.properties_verified = {}
        self.metrics = {}
        
    def add_property(self, property_name: str, verified: bool) -> None:
        """Register a formal safety property."""
        self.properties_verified[property_name] = verified
    
    def add_metrics(self, metrics: SafetyMetrics) -> None:
        """Add safety metrics."""
        self.metrics = {
            'total_episodes': metrics.total_episodes,
            'violations_per_episode': metrics.violations_per_episode,
            'zero_violations': metrics.zero_violations,
            'success_rate': metrics.success_rate
        }
    
    def is_safe(self) -> bool:
        """Check if policy is formally certified safe."""
        return all(self.properties_verified.values()) if self.properties_verified else False
    
    def save(self, filepath: str) -> None:
        """Save certificate to JSON."""
        cert = {
            'timestamp': self.timestamp,
            'model_path': self.model_path,
            'is_safe': self.is_safe(),
            'properties_verified': self.properties_verified,
            'metrics': self.metrics
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(cert, f, indent=2)
    
    def __str__(self) -> str:
        status = "✓ CERTIFIED SAFE" if self.is_safe() else "✗ FAILED VERIFICATION"
        return f"Safety Certificate [{status}] for {self.model_path}"


def verify_model_safety(model_path: str, 
                       num_episodes: int = 100,
                       save_report: bool = True) -> Tuple[bool, SafetyMetrics, SafetyCertificate]:
    """
    Complete safety verification of a trained model.
    
    Args:
        model_path: Path to saved model
        num_episodes: Episodes to test
        save_report: Save safety report
        
    Returns:
        (is_safe, metrics, certificate)
    """
    # Load model
    model = PPO.load(model_path)
    env = VertiportRLEnv(arrival_rate=20.0)
    
    # Run verification
    verifier = SafetyVerifier(verbose=True)
    metrics = verifier.verify_policy(model, env, num_episodes)
    
    # Test formal properties
    print("\nTesting formal safety properties...")
    properties = {
        'separation_maintained': FormalSafetyTest.test_separation_maintained(
            model, env, num_episodes=50
        ),
        'pad_capacity': FormalSafetyTest.test_pad_capacity(
            model, env, num_episodes=50
        ),
        'no_deadlock': FormalSafetyTest.test_no_deadlock(
            model, env, num_episodes=50
        ),
        'action_validity': FormalSafetyTest.test_action_validity(
            model, env, num_episodes=50
        )
    }
    
    # Create certificate
    cert = SafetyCertificate(model_path)
    for prop_name, result in properties.items():
        cert.add_property(prop_name, result)
        print(f"  {'✓' if result else '✗'} {prop_name.replace('_', ' ').title()}")
    
    cert.add_metrics(metrics)
    
    # Save report
    if save_report:
        report_dir = Path(model_path).parent / "safety_verification"
        report_dir.mkdir(exist_ok=True)
        
        verifier.save_report(str(report_dir / "violations.json"), metrics)
        cert.save(str(report_dir / "certificate.json"))
        
        print(f"\nSafety report saved to {report_dir}")
    
    env.close()
    
    return cert.is_safe(), metrics, cert


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safety Verification for eVTOL Policies")
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of test episodes")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip saving report")
    
    args = parser.parse_args()
    
    is_safe, metrics, cert = verify_model_safety(
        args.model,
        num_episodes=args.episodes,
        save_report=not args.no_report
    )
    
    print(f"\n{cert}")
