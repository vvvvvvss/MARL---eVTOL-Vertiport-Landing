"""
MARL eVTOL Vertiport Scheduling - Complete Training Pipeline

Orchestrates full project execution:
- Phase A: Foundation (Environments, Baselines, Training)
- Phase B: Advanced (Communication, GCN, Curriculum, Safety)

All components integrated and ready for deployment.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from curriculum_learning import train_with_curriculum, CurriculumTrainer
from safety_verification import verify_model_safety
from train_comm_gcn import train_with_communication_and_gcn


class ProjectOrchestrator:
    """Manages complete MARL project execution."""
    
    def __init__(self, output_dir: str = "./evtol_training/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_log = {
            'timestamp': datetime.now().isoformat(),
            'phases_completed': [],
            'models_trained': [],
            'verification_results': []
        }
    
    def run_phase_a_foundation(self, verbose: bool = True) -> None:
        """
        Run Phase A: Foundation Training
        - Environment setup
        - Baseline comparisons
        - Basic PPO training
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE A: FOUNDATION TRAINING")
            print("="*70)
            print("Running basic PPO training with 20 ac/hr arrival rate")
            print("This trains a baseline policy without curriculum or communication\n")
        
        # Import train_ppo from existing codebase
        from train_ppo import train_ppo
        
        model, log_path = train_ppo(
            total_timesteps=100000,
            arrival_rate=20.0,
            learning_rate=3e-4,
            model_name="evtol_ppo_baseline",
            log_dir=str(self.output_dir)
        )
        
        self.project_log['phases_completed'].append('A: Foundation')
        self.project_log['models_trained'].append({
            'name': 'PPO Baseline',
            'path': str(log_path),
            'phase': 'A',
            'completion_time': datetime.now().isoformat()
        })
        
        if verbose:
            print(f"✓ Phase A complete. Model saved to {log_path}")
    
    def run_phase_b_communication_gcn(self, verbose: bool = True) -> None:
        """
        Run Phase B.1-B2: Communication + GCN
        - Implement agent communication protocols
        - Train with GCN coordination
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE B.1-B.2: COMMUNICATION + GCN TRAINING")
            print("="*70)
            print("Training PPO with communication protocols (B1) and GCN coordination (B2)")
            print("Features: Message passing, Collective Awareness, Graph-based coordination\n")
        
        model, log_path = train_with_communication_and_gcn(
            total_timesteps=100000,
            arrival_rate=20.0,
            enable_communication=True,
            enable_gcn=True,
            learning_rate=3e-4
        )
        
        self.project_log['phases_completed'].append('B.1-B.2: Communication + GCN')
        self.project_log['models_trained'].append({
            'name': 'PPO + Communication + GCN',
            'path': str(log_path),
            'phase': 'B1-B2',
            'completion_time': datetime.now().isoformat()
        })
        
        if verbose:
            print(f"✓ Phase B.1-B.2 complete. Model saved to {log_path}")
    
    def run_phase_b3_curriculum(self, stages: int = 4, verbose: bool = True) -> None:
        """
        Run Phase B.3: Curriculum Learning
        - Progressive difficulty training across multiple stages
        """
        if verbose:
            print("\n" + "="*70)
            print(f"PHASE B.3: CURRICULUM LEARNING ({stages} STAGES)")
            print("="*70)
            print("Training with progressive difficulty increase")
            print("Stage 1: Easy (5 ac/hr) → Stage 2: Medium (15 ac/hr) → " + 
                  f"Stage 3: Hard (25 ac/hr) → Stage 4: Expert (40 ac/hr)\n")
        
        trainer = CurriculumTrainer(
            model_dir=str(self.output_dir),
            verbose=1 if verbose else 0
        )
        model, summary = trainer.train(num_stages=stages)
        trainer.plot_curriculum_progress()
        
        self.project_log['phases_completed'].append(f'B.3: Curriculum Learning ({stages} stages)')
        self.project_log['models_trained'].append({
            'name': f'Curriculum PPO ({stages} stages)',
            'path': str(self.output_dir / f"stage_{stages}_model"),
            'phase': 'B3',
            'total_timesteps': summary['total_timesteps'],
            'completion_time': datetime.now().isoformat()
        })
        
        if verbose:
            print(f"✓ Phase B.3 complete. Trained through {stages} curriculum stages")
    
    def run_phase_b4_safety_verification(self, model_path: Optional[str] = None, 
                                         verbose: bool = True) -> None:
        """
        Run Phase B.4: Formal Safety Verification
        - Verify learned policies meet safety constraints
        """
        if verbose:
            print("\n" + "="*70)
            print("PHASE B.4: FORMAL SAFETY VERIFICATION")
            print("="*70)
            print("Testing learned policy for:\n"
                  "  ✓ Separation constraint maintenance\n"
                  "  ✓ Pad capacity limits\n"
                  "  ✓ Deadlock-free execution\n"
                  "  ✓ Action validity\n")
        
        if model_path is None:
            # Use last trained model
            model_path = list(self.output_dir.glob("**/best_model.zip"))
            if model_path:
                model_path = str(model_path[0])
            else:
                print("✗ No trained model found. Run earlier phases first.")
                return
        
        is_safe, metrics, cert = verify_model_safety(
            model_path,
            num_episodes=100,
            save_report=True
        )
        
        self.project_log['verification_results'].append({
            'model': model_path,
            'is_safe': is_safe,
            'violations_per_episode': metrics.violations_per_episode,
            'success_rate': metrics.success_rate,
            'timestamp': datetime.now().isoformat()
        })
        
        if verbose:
            print(f"✓ Phase B.4 complete. {'SAFE' if is_safe else 'UNSAFE'}")
    
    def run_all_phases(self, curriculum_stages: int = 4, verbose: bool = True) -> None:
        """Execute all project phases in sequence."""
        print("\n" + "="*80)
        print(" "*15 + "MARL eVTOL VERTIPORT SCHEDULING - FULL PROJECT")
        print("="*80)
        
        phases = [
            ("Phase A: Foundation", self.run_phase_a_foundation),
            ("Phase B.1-B.2: Communication + GCN", self.run_phase_b_communication_gcn),
            (f"Phase B.3: Curriculum ({curriculum_stages} stages)", 
             lambda: self.run_phase_b3_curriculum(curriculum_stages, verbose)),
            ("Phase B.4: Safety Verification", self.run_phase_b4_safety_verification)
        ]
        
        for phase_name, phase_func in phases:
            try:
                if phase_name.startswith("Phase"):
                    if "Foundation" in phase_name:
                        phase_func(verbose)
                    elif "Communication" in phase_name:
                        phase_func(verbose)
                    elif "Curriculum" in phase_name:
                        phase_func()
                    elif "Safety" in phase_name:
                        phase_func(verbose=verbose)
            except Exception as e:
                print(f"✗ Error in {phase_name}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        self.save_project_summary()
    
    def save_project_summary(self) -> None:
        """Save complete project summary."""
        summary_path = self.output_dir / "PROJECT_SUMMARY.json"
        self.project_log['completion_time'] = datetime.now().isoformat()
        
        with open(summary_path, 'w') as f:
            json.dump(self.project_log, f, indent=2)
        
        print("\n" + "="*70)
        print("PROJECT SUMMARY")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Phases completed: {len(self.project_log['phases_completed'])}")
        for phase in self.project_log['phases_completed']:
            print(f"  ✓ {phase}")
        print(f"\nModels trained: {len(self.project_log['models_trained'])}")
        for model in self.project_log['models_trained']:
            print(f"  • {model['name']} ({model['phase']})")
        print(f"\nSummary saved to: {summary_path}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="MARL eVTOL Vertiport Scheduling - Complete Project Pipeline"
    )
    parser.add_argument(
        "--phase",
        choices=['A', 'B1-B2', 'B3', 'B4', 'all'],
        default='all',
        help="Which phase to run (default: all)"
    )
    parser.add_argument(
        "--curriculum-stages",
        type=int,
        default=4,
        help="Number of curriculum stages for Phase B3 (1-4, default 4)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model for B4 verification (auto-find if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evtol_training/",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    orchestrator = ProjectOrchestrator(args.output_dir)
    verbose = not args.quiet
    
    if args.phase == 'all':
        orchestrator.run_all_phases(args.curriculum_stages, verbose)
    elif args.phase == 'A':
        orchestrator.run_phase_a_foundation(verbose)
    elif args.phase == 'B1-B2':
        orchestrator.run_phase_b_communication_gcn(verbose)
    elif args.phase == 'B3':
        orchestrator.run_phase_b3_curriculum(args.curriculum_stages, verbose)
    elif args.phase == 'B4':
        orchestrator.run_phase_b4_safety_verification(args.model, verbose)


if __name__ == "__main__":
    main()
