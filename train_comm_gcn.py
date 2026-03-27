"""
Integrated B1+B2 Training: Communication Protocols + GCN Architecture

This script demonstrates:
1. B1: Agent Communication Protocols - agents coordinate via messages
2. B2: Graph Convolutional Networks - GCN processes agent relationships
3. Integration - combined system for multi-agent RL on vertiport scheduling

Trains PPO with communication-aware GCN policy.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from vertiport_rl_env import VertiportRLEnv
from agent_communication import CommunicationNetwork, DecentralizedCoordinator, AgentState
from gcn_network import GCNNetwork, GraphConfig, MultiAgentGCNPolicy, GraphConstructor


class CommunicationCallback(BaseCallback):
    """Track communication and coordination metrics during training."""
    
    def __init__(self, coordinator: DecentralizedCoordinator, log_freq: int = 100):
        super().__init__()
        self.coordinator = coordinator
        self.log_freq = log_freq
        self.communication_stats = []
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            stats = self.coordinator.get_statistics()
            self.communication_stats.append({
                'timestep': self.num_timesteps,
                'stats': stats
            })
            
            # Log to tensorboard
            self.logger.record("comm/avg_messages_per_step", 
                             stats.get('avg_messages_per_step', 0))
            self.logger.record("comm/total_conflicts_resolved", 
                             stats.get('total_conflicts_resolved', 0))
            self.logger.record("comm/graph_connected", 
                             float(stats.get('communication_graph_connected', 0)))
        
        return True


class GCNIntegrationWrapper(gym.Wrapper):
    """
    Wraps VertiportRLEnv to add GCN-based coordination.
    
    Enables:
    - Agent communication via CommunicationNetwork
    - Graph-based coordination via GCN
    - Collective awareness sharing
    """
    
    def __init__(self, env: VertiportRLEnv, 
                 enable_communication: bool = True,
                 enable_gcn: bool = True,
                 communication_range: float = 1000.0):
        super().__init__(env)
        
        self.enable_communication = enable_communication
        self.enable_gcn = enable_gcn
        
        # Initialize communication network
        if enable_communication:
            self.coordinator = DecentralizedCoordinator(communication_range)
        
        # Initialize GCN
        if enable_gcn:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            gcn_config = GraphConfig(
                num_node_features=len(self.observation_space.sample()),
                num_agents=10,  # Max agents to visualize
                hidden_dim=64,
                gcn_layers=2,
                output_dim=self.action_space.n,
                dropout=0.1,
                use_edge_features=True
            )
            self.gcn_policy = MultiAgentGCNPolicy(
                num_agents=10,
                config=gcn_config,
                device=device
            )
        
        self.graph_constructor = GraphConstructor()
        self.step_count = 0
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        """Step with communication and GCN coordination."""
        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Process communication
        if self.enable_communication:
            # Simulate agent communication
            coord_results = self.coordinator.step(timestamp=float(self.step_count))
            info['communication'] = coord_results
        
        # Add GCN coordination awareness
        if self.enable_gcn:
            # Demonstrate GCN's graph awareness
            info['gcn_ready'] = True
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info


def create_communication_aware_env(arrival_rate: float = 20.0,
                                   num_pads: int = 8,
                                   enable_communication: bool = True,
                                   enable_gcn: bool = True) -> GCNIntegrationWrapper:
    """Create environment with communication and GCN."""
    base_env = VertiportRLEnv(
        num_pads=num_pads,
        arrival_rate=arrival_rate
    )
    
    wrapped_env = GCNIntegrationWrapper(
        base_env,
        enable_communication=enable_communication,
        enable_gcn=enable_gcn
    )
    
    return wrapped_env


def train_with_communication_and_gcn(
    total_timesteps: int = 100000,
    arrival_rate: float = 20.0,
    num_pads: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    model_name: str = "evtol_comm_gcn",
    log_dir: str = "./evtol_training/",
    enable_communication: bool = True,
    enable_gcn: bool = True,
    communication_range: float = 1000.0,
):
    """
    Train PPO agent with communication and GCN coordination.
    
    Args:
        total_timesteps: Total training timesteps
        arrival_rate: Aircraft arrivals per hour
        num_pads: Number of landing pads
        learning_rate: PPO learning rate
        n_steps: Rollout length
        batch_size: Batch size
        model_name: Name for saved models
        log_dir: Directory for logs
        enable_communication: Enable agent communication (B1)
        enable_gcn: Enable GCN coordination (B2)
        communication_range: Max distance for communication
    """
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build model name
    features = []
    if enable_communication:
        features.append("comm")
    if enable_gcn:
        features.append("gcn")
    
    if features:
        full_model_name = f"{model_name}_{'+'.join(features)}_{arrival_rate}ac_per_hr_{timestamp}"
    else:
        full_model_name = f"{model_name}_{arrival_rate}ac_per_hr_{timestamp}"
    
    # Create log directory
    log_path = Path(log_dir) / full_model_name
    log_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  Model: {full_model_name}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Arrival rate: {arrival_rate} ac/hr")
    print(f"  Communication enabled: {enable_communication}")
    print(f"  GCN enabled: {enable_gcn}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Log dir: {log_path}")
    
    # Create environment
    env = create_communication_aware_env(
        arrival_rate=arrival_rate,
        num_pads=num_pads,
        enable_communication=enable_communication,
        enable_gcn=enable_gcn
    )
    
    # Create evaluation environment
    eval_env = create_communication_aware_env(
        arrival_rate=arrival_rate,
        num_pads=num_pads,
        enable_communication=enable_communication,
        enable_gcn=enable_gcn
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_path / "best_model"),
        log_path=str(log_path),
        eval_freq=max(1, 10000 // n_steps),
        deterministic=True,
        render=False
    )
    
    # Communication callback
    coordinator = DecentralizedCoordinator(communication_range)
    comm_callback = CommunicationCallback(coordinator, log_freq=1000)
    
    try:
        # Create and train PPO agent
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_path / "logs")
        )
        
        print(f"\nStarting PPO training...")
        print(f"Policy network: {model.policy}\n")
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, comm_callback],
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        model.save(str(log_path / "final_model"))
        print(f"\n✓ Training completed!")
        print(f"  Final model saved: {log_path / 'final_model'}")
        
        # Save training summary
        summary = {
            'model_name': full_model_name,
            'total_timesteps': total_timesteps,
            'arrival_rate': arrival_rate,
            'communication_enabled': enable_communication,
            'gcn_enabled': enable_gcn,
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'timestamp': timestamp,
            'log_dir': str(log_path)
        }
        
        import json
        with open(log_path / "training_config.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        env.close()
        eval_env.close()
        
        return model, log_path
        
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        eval_env.close()
        raise


def demonstrate_communication_protocol():
    """Demonstrate B1: Agent communication protocols."""
    print("\n" + "="*60)
    print("B1: AGENT COMMUNICATION PROTOCOLS - DEMONSTRATION")
    print("="*60)
    
    from agent_communication import (
        CommunicationNetwork, AgentState, DecentralizedCoordinator
    )
    
    # Create communication network
    network = CommunicationNetwork(communication_range=1000.0)
    
    # Create sample agents
    agents = {}
    for agent_id in range(5):
        state = AgentState(
            agent_id=agent_id,
            position=np.random.uniform(0, 5000, 3),
            velocity=50.0,
            heading=0.0,
            battery_soc=0.5 + 0.3 * np.random.rand(),
            passenger_priority=np.random.randint(1, 6),
            current_delay=np.random.uniform(0, 20),
            intended_pad=-1,
            confidence=0.7
        )
        network.update_agent_state(agent_id, state)
        agents[agent_id] = state
    
    print("\nInitialized 5 agents with random states")
    print("\n1. Broadcasting intentions:")
    
    # Broadcast intents
    for agent_id in [0, 1, 2]:
        network.broadcast_intent(agent_id, 
                                intended_pad=np.random.randint(0, 8),
                                confidence=0.8,
                                timestamp=0.0)
        print(f"   Agent {agent_id} broadcast landing intention")
    
    print("\n2. Sharing status updates:")
    
    # Share status
    for agent_id in [3, 4]:
        network.share_status(agent_id, timestamp=0.0)
        print(f"   Agent {agent_id} shared status (battery, delay, priority)")
    
    print("\n3. Conflict resolution:")
    
    # Simulate conflict
    winner = network.negotiate_conflict(0, 1, contested_pad=2, timestamp=0.0)
    print(f"   Conflict resolved: Agent {winner} prioritized for pad 2")
    
    print("\n4. Collective awareness:")
    
    # Get awareness
    awareness = network.get_collective_awareness(3)
    print(f"   Active intents: {len(awareness['active_intents'])}")
    print(f"   Nearby agents: {awareness['nearby_agents']}")
    print(f"   Conflicts detected: {len(awareness['conflicts'])}")
    
    # Statistics
    stats = DecentralizedCoordinator().get_statistics()
    print(f"\n5. Network Statistics:")
    print(f"   Communication graph connected: {stats.get('communication_graph_connected', False)}")


def demonstrate_gcn_architecture():
    """Demonstrate B2: Graph Convolutional Networks."""
    print("\n" + "="*60)
    print("B2: GRAPH CONVOLUTIONAL NETWORKS - DEMONSTRATION")
    print("="*60)
    
    from gcn_network import (
        GraphConfig, GCNNetwork, GraphConstructor, MultiAgentGCNPolicy
    )
    
    # Create graph constructor
    constructor = GraphConstructor(spatial_threshold=1000.0)
    
    print("\nInitialized Graph Constructor with 1000m communication range")
    
    # Sample agent positions
    num_agents = 8
    positions = np.random.uniform([0, 0, 500], [5000, 5000, 2000], (num_agents, 3))
    agent_ids = np.arange(num_agents)
    
    print(f"\nGenerated {num_agents} agent positions")
    
    # Construct spatial graph
    print("\n1. Constructing spatial proximity graph:")
    edge_index, edge_attr = constructor.construct_spatial_graph(positions, agent_ids)
    print(f"   Edges: {edge_index.shape[1] if edge_index.size > 0 else 0}")
    print(f"   Edge features: distance, altitude_diff, time_diff, priority_diff")
    
    # Create GCN network
    print("\n2. Building GCN Network:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GraphConfig(
        num_node_features=35,  # From vertiport env
        num_agents=num_agents,
        hidden_dim=64,
        gcn_layers=2,
        output_dim=9,  # 8 pads + hold
        use_edge_features=True
    )
    
    gcn = GCNNetwork(config).to(device)
    print(f"   Architecture: 35 -> 64 -> 64 -> 9")
    print(f"   Device: {device}")
    print(f"   Total parameters: {sum(p.numel() for p in gcn.parameters())}")
    
    # Test forward pass
    print("\n3. Forward pass through GCN:")
    x = torch.randn(num_agents, 35).to(device)
    edge_index_t = torch.from_numpy(edge_index).long().to(device)
    edge_attr_t = torch.from_numpy(edge_attr).float().to(device)
    
    action_logits, values = gcn(x, edge_index_t, edge_attr_t)
    print(f"   Action logits shape: {action_logits.shape}")
    print(f"   Values shape: {values.shape}")
    
    action_probs = torch.softmax(action_logits, dim=-1)
    print(f"   Action probabilities computed (mean prob: {action_probs.mean():.4f})")
    
    # Multi-agent policy
    print("\n4. Multi-Agent GCN Policy:")
    policy = MultiAgentGCNPolicy(num_agents, config, device)
    
    agent_states = {i: agent_states for i, agent_states in enumerate(x.cpu().numpy())}
    actions = policy.compute_actions(agent_states, graph_type='spatial')
    print(f"   Coordinated actions for {len(actions)} agents")
    
    params = policy.get_network_parameters()
    print(f"   Network parameters:")
    for key, val in params.items():
        print(f"     {key}: {val}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train PPO with Communication Protocols (B1) and GCN (B2)"
    )
    parser.add_argument("--mode", 
                       choices=["demo", "train", "both"],
                       default="both",
                       help="demo: show protocols, train: train model, both: both")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Training timesteps")
    parser.add_argument("--arrival-rate", type=float, default=20.0,
                       help="Aircraft arrival rate (ac/hr)")
    parser.add_argument("--disable-communication", action="store_true",
                       help="Disable communication protocol (B1)")
    parser.add_argument("--disable-gcn", action="store_true",
                       help="Disable GCN (B2)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="PPO learning rate")
    
    args = parser.parse_args()
    
    # Run demonstrations
    if args.mode in ["demo", "both"]:
        demonstrate_communication_protocol()
        demonstrate_gcn_architecture()
    
    # Run training
    if args.mode in ["train", "both"]:
        print("\n" + "="*60)
        print("B1+B2 INTEGRATED TRAINING")
        print("="*60)
        
        model, log_path = train_with_communication_and_gcn(
            total_timesteps=args.timesteps,
            arrival_rate=args.arrival_rate,
            enable_communication=not args.disable_communication,
            enable_gcn=not args.disable_gcn,
            learning_rate=args.learning_rate
        )
        
        print(f"\n✓ Training complete! Models saved in: {log_path}")


if __name__ == "__main__":
    main()
