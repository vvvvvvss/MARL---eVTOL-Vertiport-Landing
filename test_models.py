#!/usr/bin/env python3
"""Quick test to see if models load correctly."""

import sys
from pathlib import Path

print("=" * 80)
print("MARL eVTOL - MODEL DISCOVERY TEST")
print("=" * 80)

# List available models
models_dir = Path('./evtol_training')
if not models_dir.exists():
    print("❌ evtol_training/ directory not found!")
    sys.exit(1)

model_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir()])
print(f"\n✓ Found {len(model_dirs)} model directories:\n")

for i, model_dir in enumerate(model_dirs, 1):
    model_path = model_dir / 'best_model' / 'best_model.zip'
    or_path = model_dir / 'best_model.zip'
    
    exists_a = model_path.exists()
    exists_b = or_path.exists()
    
    status = "✓ Ready" if (exists_a or exists_b) else "❌ Missing"
    full_name = model_dir.name
    print(f"  {i:2d}. [{status}] {full_name[:60]}")
    if exists_a:
        print(f"       → {model_path}")
    elif exists_b:
        print(f"       → {or_path}")

print("\n" + "=" * 80)
print("NOW LOADING TRAINED MODELS WITH PPO...")
print("=" * 80)

try:
    from stable_baselines3 import PPO
    
    # Try loading the first curriculum stage
    best_model = None
    for model_dir in model_dirs:
        if 'curriculum_stage4' in model_dir.name:
            model_path = model_dir / 'best_model' / 'best_model.zip'
            if model_path.exists():
                print(f"\n🔄 Loading: {model_dir.name}")
                try:
                    model = PPO.load(str(model_path), device='cpu')
                    print(f"   ✓ Successfully loaded!")
                    best_model = model
                    break
                except Exception as e:
                    print(f"   ❌ Error: {e}")
    
    if best_model:
        print("\n" + "=" * 80)
        print("✓ MODELS READY FOR DASHBOARD!")
        print("=" * 80)
        print(f"Successfully loaded model with {sum(p.numel() for p in best_model.policy.parameters())} parameters")
    else:
        print("\n⚠️  Could not load any models")
        
except Exception as e:
    print(f"\n❌ Error importing PPO: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 Next: Launch gradio_dashboard_real.py")
