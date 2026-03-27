"""
Graph Convolutional Network (GCN) Architecture for Multi-Agent RL

Implements GCN layers to process agent interaction graphs, enabling:
1. Spatial relationship learning
2. Agent interaction modeling
3. Hierarchical feature aggregation
4. Graph-based policy learning

GCN learns from graph structure where nodes are agents and edges represent
spatial proximity, communication links, or priority relationships.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GraphConfig:
    """Configuration for GCN network."""
    num_node_features: int  # Input feature dimension per agent
    num_agents: int  # Maximum number of agents
    hidden_dim: int = 64
    gcn_layers: int = 2
    output_dim: int = 9  # Action space (8 pads + hold)
    dropout: float = 0.1
    use_edge_features: bool = True
    edge_feature_dim: int = 4  # distance, relative_velocity, etc.


class GraphConstructor:
    """Builds dynamic graphs from agent states."""
    
    def __init__(self, 
                 spatial_threshold: float = 1000.0,  # meters
                 temporal_threshold: float = 30.0):   # seconds
        """
        Args:
            spatial_threshold: Max distance for spatial edges
            temporal_threshold: Time window for temporal relationships
        """
        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
    
    def construct_spatial_graph(self, 
                               agent_positions: np.ndarray,
                               agent_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on spatial proximity.
        
        Args:
            agent_positions: Array of shape (num_agents, 3) with [x, y, z]
            agent_ids: Array of agent IDs
            
        Returns:
            edge_index: Shape (2, num_edges) - source and target indices
            edge_attr: Shape (num_edges, 4) - [distance, relative_alt, time_diff, priority_diff]
        """
        num_agents = len(agent_positions)
        edges = []
        edge_features = []
        
        if num_agents == 0:
            return np.array([[],[]], dtype=np.int64), np.zeros((0, 4))
        
        # Compute pairwise distances
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                pos_a = agent_positions[i]
                pos_b = agent_positions[j]
                
                # Euclidean distance
                distance = np.linalg.norm(pos_a - pos_b)
                
                # Create edge if within threshold
                if distance <= self.spatial_threshold:
                    # Add bidirectional edges
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Edge features
                    rel_alt = abs(pos_a[2] - pos_b[2]) / 1000.0  # Normalize to km
                    norm_distance = distance / self.spatial_threshold
                    
                    features = np.array([norm_distance, rel_alt, 0.0, 0.0], 
                                      dtype=np.float32)
                    edge_features.append(features)
                    edge_features.append(features)
        
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.array([[],[]],dtype=np.int64)
        edge_attr = np.vstack(edge_features) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        return edge_index, edge_attr
    
    def construct_communication_graph(self,
                                    communication_ranges: Dict[int, float],
                                    communication_links: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on communication links.
        
        Args:
            communication_ranges: Dict mapping agent_id to communication range
            communication_links: Array indicating which agents can communicate
            
        Returns:
            edge_index: Shape (2, num_edges)
            edge_attr: Shape (num_edges, 4) - communication quality, latency, etc.
        """
        edges = []
        edge_features = []
        
        for i in range(len(communication_links)):
            for j in range(i + 1, len(communication_links)):
                if communication_links[i, j] > 0:
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Quality and latency features
                    quality = communication_links[i, j]  # 0-1
                    latency = 0.01 * (1 - quality)  # Inversely related
                    
                    features = np.array([quality, latency, 0.0, 0.0], 
                                      dtype=np.float32)
                    edge_features.append(features)
                    edge_features.append(features)
        
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.array([[],[]],dtype=np.int64)
        edge_attr = np.vstack(edge_features) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        return edge_index, edge_attr
    
    def construct_priority_graph(self,
                                priorities: np.ndarray,
                                priority_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on priority relationships.
        
        Args:
            priorities: Array of agent priorities (0-1 or 1-5 scale)
            priority_threshold: Min priority difference to create edge
            
        Returns:
            edge_index: Shape (2, num_edges)
            edge_attr: Shape (num_edges, 4)
        """
        num_agents = len(priorities)
        edges = []
        edge_features = []
        
        # Normalize priorities to 0-1
        if len(priorities) > 0:
            max_priority = max(priorities)
            if max_priority > 0:
                normalized = priorities / max_priority
            else:
                normalized = priorities
        else:
            normalized = priorities
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                priority_diff = abs(normalized[i] - normalized[j])
                
                if priority_diff >= priority_threshold:
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    features = np.array([priority_diff, 0.0, 0.0, 0.0], 
                                      dtype=np.float32)
                    edge_features.append(features)
                    edge_features.append(features)
        
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.array([[],[]],dtype=np.int64)
        edge_attr = np.vstack(edge_features) if edge_features else np.zeros((0, 4), dtype=np.float32)
        
        return edge_index, edge_attr


class GCNLayer(nn.Module):
    """Single Graph Convolutional Network layer."""
    
    def __init__(self, in_features: int, out_features: int, 
                 use_bias: bool = True, use_edge_features: bool = False,
                 edge_feature_dim: int = 4):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_edge_features = use_edge_features
        
        # Linear transformation for nodes
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Edge feature processing (if enabled)
        if use_edge_features:
            self.edge_weight = nn.Parameter(torch.Tensor(edge_feature_dim, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                degree: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        GCN forward pass.
        
        Args:
            x: Node features (num_nodes, in_features)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_feature_dim)
            degree: Node degrees (num_nodes,)
            
        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Linear transformation
        x = torch.matmul(x, self.weight)
        
        if edge_index.shape[1] == 0:
            # No edges, return features with bias
            if self.bias is not None:
                x = x + self.bias
            return x
        
        # Compute degree and normalization
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        
        if degree is None:
            degree = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, edge_index[0], torch.ones(num_edges, 
                                                             device=x.device, dtype=x.dtype))
        
        # Normalization: D^-1/2 * A * D^-1/2
        degree = torch.clamp(degree, min=1)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        
        # Efficient vectorized aggregation
        src, dst = edge_index[0], edge_index[1]
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Gather source features and normalize
        src_features = x[src]  # (num_edges, num_features)
        normalized_features = src_features * norm.unsqueeze(-1)
        
        # Use index_add_ for efficient aggregation
        out = torch.zeros_like(x)
        for i in range(num_edges):
            out[dst[i]] = out[dst[i]] + normalized_features[i]
        
        # Add edge features if present
        if edge_attr is not None and self.use_edge_features:
            edge_contribution = torch.matmul(edge_attr, self.edge_weight)  # (num_edges, num_features)
            for i in range(num_edges):
                out[dst[i]] = out[dst[i]] + edge_contribution[i]
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GCNNetwork(nn.Module):
    """Multi-layer GCN for agent coordination."""
    
    def __init__(self, config: GraphConfig):
        super(GCNNetwork, self).__init__()
        self.config = config
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(config.num_node_features if i == 0 else config.hidden_dim,
                    config.hidden_dim,
                    use_edge_features=config.use_edge_features,
                    edge_feature_dim=config.edge_feature_dim)
            for i in range(config.gcn_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
        self.value_layer = nn.Linear(config.hidden_dim, 1)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features (num_nodes, num_node_features)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_feature_dim)
            
        Returns:
            action_logits: (num_nodes, output_dim)
            values: (num_nodes, 1)
        """
        # GCN forward
        features = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            features = gcn_layer(features, edge_index, edge_attr)
            features = F.relu(features)
            features = self.dropout(features)
        
        # Output heads
        action_logits = self.output_layer(features)
        values = self.value_layer(features)
        
        return action_logits, values


class GCNAgent:
    """Wrapper for agent using GCN policy."""
    
    def __init__(self, agent_id: int, config: GraphConfig, device: str = 'cpu'):
        self.agent_id = agent_id
        self.config = config
        self.device = device
        
        self.policy_network = GCNNetwork(config).to(device)
        self.graph_constructor = GraphConstructor()
        
        self.observation_buffer = []
        self.state_history = []
    
    def update_observation(self, observation: np.ndarray):
        """Buffer observation for graph construction."""
        self.observation_buffer.append(observation)
        if len(self.observation_buffer) > 100:
            self.observation_buffer.pop(0)
    
    def compute_action(self, 
                      agent_positions: np.ndarray,
                      agent_features: torch.Tensor,
                      agent_ids: np.ndarray,
                      use_spatial_graph: bool = True) -> Tuple[int, float]:
        """
        Compute action using GCN policy.
        
        Args:
            agent_positions: Positions of all agents
            agent_features: Feature tensor for all agents
            agent_ids: Agent IDs
            use_spatial_graph: Whether to use spatial proximity graph
            
        Returns:
            action: Selected action (0-8 for landing/hold)
            action_prob: Probability of selected action
        """
        with torch.no_grad():
            # Construct graph
            if use_spatial_graph:
                edge_index, edge_attr = self.graph_constructor.construct_spatial_graph(
                    agent_positions, agent_ids
                )
            else:
                edge_index = np.array([[], []], dtype=np.int64)
                edge_attr = None
            
            # Convert to tensors
            edge_index_t = torch.from_numpy(edge_index).long().to(self.device)
            if edge_attr is not None:
                edge_attr_t = torch.from_numpy(edge_attr).float().to(self.device)
            else:
                edge_attr_t = None
            
            agent_features_t = agent_features.to(self.device) if not agent_features.is_cuda else agent_features
            
            # Forward pass through GCN
            action_logits, values = self.policy_network(
                agent_features_t, edge_index_t, edge_attr_t
            )
            
            # Select action for this agent
            my_idx = np.where(agent_ids == self.agent_id)[0][0]
            my_logits = action_logits[my_idx]
            
            action_probs = torch.softmax(my_logits, dim=-1)
            action = torch.argmax(action_probs).item()
            action_prob = action_probs[action].item()
            
        return action, action_prob


class MultiAgentGCNPolicy:
    """Multi-agent coordination policy using GCN."""
    
    def __init__(self, num_agents: int, config: GraphConfig, device: str = 'cpu'):
        self.num_agents = num_agents
        self.config = config
        self.device = device
        
        self.shared_network = GCNNetwork(config).to(device)
        self.graph_constructor = GraphConstructor()
        
        self.agents = {i: GCNAgent(i, config, device) for i in range(num_agents)}
    
    def compute_actions(self, 
                       agent_states: Dict[int, np.ndarray],
                       graph_type: str = 'spatial') -> Dict[int, Tuple[int, float]]:
        """
        Compute actions for all agents using shared GCN.
        
        Args:
            agent_states: Dict mapping agent_id to state array
            graph_type: 'spatial', 'communication', or 'priority'
            
        Returns:
            Dict mapping agent_id to (action, prob)
        """
        num_agents = len(agent_states)
        
        if num_agents == 0:
            return {}
        
        # Prepare data
        agent_ids = np.array(list(agent_states.keys()))
        agent_positions = np.vstack([agent_states[aid][:3] 
                                    for aid in agent_ids])  # x, y, z
        
        # Construct graph based on type
        if graph_type == 'spatial':
            edge_index, edge_attr = self.graph_constructor.construct_spatial_graph(
                agent_positions, agent_ids
            )
        else:
            edge_index = np.array([[], []], dtype=np.int64)
            edge_attr = None
        
        # Prepare features (flatten observations)
        features = []
        for aid in agent_ids:
            state = agent_states[aid]
            # Normalize and pad to fixed size
            state_norm = np.clip(state, 0, 1)
            if len(state_norm) < self.config.num_node_features:
                state_norm = np.pad(state_norm, 
                                   (0, self.config.num_node_features - len(state_norm)))
            features.append(state_norm[:self.config.num_node_features])
        
        features_tensor = torch.from_numpy(np.vstack(features)).float()
        
        # Convert graph to tensor
        edge_index_t = torch.from_numpy(edge_index).long().to(self.device)
        if edge_attr is not None:
            edge_attr_t = torch.from_numpy(edge_attr).float().to(self.device)
        else:
            edge_attr_t = None
        
        # Forward pass
        with torch.no_grad():
            features_tensor = features_tensor.to(self.device)
            action_logits, values = self.shared_network(
                features_tensor, edge_index_t, edge_attr_t
            )
            
            action_probs = torch.softmax(action_logits, dim=-1)
            actions = torch.argmax(action_probs, dim=-1)
        
        # Return actions for all agents
        results = {}
        for i, aid in enumerate(agent_ids):
            results[aid] = (actions[i].item(), action_probs[i, actions[i]].item())
        
        return results
    
    def get_network_parameters(self) -> Dict[str, Any]:
        """Get network architecture details."""
        return {
            'num_layers': self.config.gcn_layers,
            'hidden_dim': self.config.hidden_dim,
            'num_node_features': self.config.num_node_features,
            'num_agents': self.num_agents,
            'total_parameters': sum(p.numel() for p in self.shared_network.parameters())
        }
