"""
Diagnostic Script for PPO Training Issues
Analyzes what went wrong with the trained PPO model
"""

import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from vertiport_rl_env import VertiportRLEnv


def diagnose_ppo_model(model_path: str):
    """Diagnose issues with a trained PPO model."""
    
    print("=" * 80)
    print("PPO MODEL DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Check if model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"✗ Model not found: {model_path}")
        return
    
    print(f"✓ Model found: {model_path}\n")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Training steps: {model.num_timesteps}")
        print(f"  Learning rate: {model.learning_rate}\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Check training config
    config_path = model_path_obj.parent / ".." / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("Training Configuration:")
        for key, value in config.items():
            if key != 'run_dir':
                print(f"  {key}: {value}")
        print()
    
    # Test model on multiple scenarios
    print("Testing Model Performance:")
    print("-" * 80)
    
    results = {}
    for arrival_rate in [10, 20, 30]:
        print(f"\nArrival Rate: {arrival_rate} ac/hr")
        
        env = VertiportRLEnv(num_pads=8, arrival_rate=arrival_rate)
        obs, _ = env.reset()
        
        total_landings = 0
        total_interactions = 0
        action_counts = {i: 0 for i in range(9)}  # 8 pads + 1 hold
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action[0]) if action.shape else int(action)
            else:
                action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            
            action_counts[action] += 1
            total_interactions += 1
            
            if terminated or truncated:
                break
        
        total_landings = env.env.total_landings
        
        # Analyze actions
        pad_actions = sum(action_counts[i] for i in range(8))
        hold_actions = action_counts[8]
        
        print(f"  Landings: {total_landings}")
        print(f"  Pad actions: {pad_actions}")
        print(f"  Hold actions: {hold_actions}")
        print(f"  Most used action: ", end="")
        
        max_action = max(action_counts, key=action_counts.get)
        if max_action < 8:
            print(f"Pad {max_action} ({action_counts[max_action]} times)")
        else:
            print(f"Hold ({action_counts[max_action]} times)")
        
        results[arrival_rate] = {
            'landings': total_landings,
            'pad_actions': pad_actions,
            'hold_actions': hold_actions,
            'most_used_pad': max_action if max_action < 8 else 'hold'
        }
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if model is stuck on one action
    has_diversity = any(
        results[ar]['most_used_pad'] != results[10]['most_used_pad'] 
        for ar in [20, 30]
    )
    
    avg_landings = np.mean([results[ar]['landings'] for ar in [10, 20, 30]])
    
    print("\n🔍 Issues Found:\n")
    
    if avg_landings < 5:
        print("  1. ⚠️  VERY LOW LANDING RATE")
        print("     - Model lands fewer than 5 aircraft on average")
        print("     - This suggests the model hasn't learned to land at all\n")
    
    if not has_diversity:
        print("  2. ⚠️  STUCK ON ONE PAD")
        pad = results[10]['most_used_pad']
        print(f"     - Model repeatedly selects the same pad/action: {pad}")
        print("     - This indicates poor exploration or converged to local minimum\n")
    
    hold_ratio = np.mean([results[ar]['hold_actions'] / 500 for ar in [10, 20, 30]])
    if hold_ratio > 0.8:
        print("  3. ⚠️  MODEL MOSTLY HOLDS")
        print(f"     - {hold_ratio:.1%} of actions are 'hold'")
        print("     - Model isn't attempting to land\n")
    
    print("\n💡 Recommendations:\n")
    
    print("  1. RETRAIN WITH LONGER STEPS")
    print("     - Current model: 100k steps")
    print("     - Recommended: 500k-1M steps")
    print("     - Command: python train_ppo.py --total-timesteps 500000\n")
    
    print("  2. USE CURRICULUM LEARNING")
    print("     - Start with easier scenarios (10 ac/hr)")
    print("     - Progressively increase difficulty")
    print("     - Command: python train_ppo.py --curriculum\n")
    
    print("  3. ADJUST HYPERPARAMETERS")
    print("     - Try lower learning rate: --learning-rate 0.00001")
    print("     - Increase entropy for exploration: --ent-coef 0.05")
    print("     - Increase n_steps: --n-steps 4096\n")
    
    print("  4. CHECK REWARD STRUCTURE")
    print("     - Verify reward function in vertiport_env.py")
    print("     - Ensure landing reward is positive")
    print("     - Check delay penalty magnitude\n")
    
    print("  5. VALIDATE ENVIRONMENT")
    print("     - Run: python vertiport_rl_env.py")
    print("     - Verify observations and rewards make sense\n")


if __name__ == "__main__":
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Diagnose PPO training issues")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to PPO model (supports wildcards)")
    
    args = parser.parse_args()
    
    if not args.model:
        # Find latest model
        models = glob.glob("evtol_training/*/final_evtol_ppo.zip")
        if models:
            args.model = sorted(models)[-1]
            print(f"Found model: {args.model}\n")
        else:
            print("No PPO model found. Train one first:")
            print("  python train_ppo.py")
            exit(1)
    else:
        # Handle wildcards
        if '*' in args.model:
            matches = glob.glob(args.model)
            if matches:
                args.model = matches[0]
            else:
                print(f"No models matching: {args.model}")
                exit(1)
    
    diagnose_ppo_model(args.model)
