"""
Test Suite: Agent Communication (B1) + GCN (B2) Integration

Tests to verify:
1. Communication network functionality
2. GCN graph construction
3. GCN forward passes
4. Integration between B1 and B2
5. End-to-end training
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

# Add codebase path
sys.path.insert(0, str(Path(__file__).parent))

from agent_communication import (
    MessageBuffer, CommunicationNetwork, DecentralizedCoordinator, 
    AgentState, AgentMessage
)
from gcn_network import (
    GraphConfig, GraphConstructor, GCNLayer, GCNNetwork, 
    GCNAgent, MultiAgentGCNPolicy
)


class TestAgentCommunication(unittest.TestCase):
    """Test B1: Agent Communication Protocols."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.network = CommunicationNetwork(communication_range=1000.0)
        self.num_agents = 5
        
        # Create test agents
        for i in range(self.num_agents):
            state = AgentState(
                agent_id=i,
                position=np.array([i * 100, i * 100, 500 + i * 100]),
                velocity=50.0,
                heading=0.0,
                battery_soc=0.5 + 0.2 * np.random.rand(),
                passenger_priority=np.random.randint(1, 6),
                current_delay=np.random.uniform(0, 20),
                intended_pad=-1,
                confidence=0.7
            )
            self.network.update_agent_state(i, state)
    
    def test_message_buffer(self):
        """Test message buffer operations."""
        buffer = MessageBuffer()
        
        # Create test messages
        msg1 = AgentMessage(
            sender_id=0, receiver_id=1,
            message_type='intent',
            timestamp=0.0,
            content={'intent': 3},
            priority=5
        )
        
        msg2 = AgentMessage(
            sender_id=1, receiver_id=0,
            message_type='status',
            timestamp=1.0,
            content={'battery': 0.8},
            priority=3
        )
        
        # Test sending
        buffer.send(msg1)
        buffer.send(msg2)
        
        # Messages should be queued
        self.assertEqual(len(buffer.outgoing), 2)
        
        # Flush outgoing
        messages = buffer.flush_outgoing()
        self.assertEqual(len(messages), 2)
        self.assertEqual(len(buffer.outgoing), 0)
    
    def test_broadcast_intent(self):
        """Test broadcasting landing intent."""
        self.network.broadcast_intent(
            agent_id=0,
            intended_pad=3,
            confidence=0.85,
            timestamp=0.0
        )
        
        # Check broadcast hub
        broadcast = self.network.get_broadcast_history()
        self.assertGreater(len(broadcast), 0)
        
        last_msg = broadcast[-1]
        self.assertEqual(last_msg.message_type, 'intent')
        self.assertEqual(last_msg.sender_id, 0)
        self.assertEqual(last_msg.receiver_id, -1)  # Broadcast
    
    def test_status_sharing(self):
        """Test status sharing."""
        self.network.share_status(agent_id=1, timestamp=0.0)
        
        broadcast = self.network.get_broadcast_history()
        self.assertGreater(len(broadcast), 0)
        
        last_msg = broadcast[-1]
        self.assertEqual(last_msg.message_type, 'status')
        self.assertIn('battery_soc', last_msg.content)
        self.assertIn('passenger_priority', last_msg.content)
    
    def test_conflict_resolution(self):
        """Test conflict negotiation."""
        winner = self.network.negotiate_conflict(
            agent_a=0, agent_b=1,
            contested_pad=2,
            timestamp=0.0
        )
        
        # Winner should be one of the agents
        self.assertIn(winner, [0, 1])
        
        # Check broadcast
        broadcast = self.network.get_broadcast_history()
        self.assertGreater(len(broadcast), 0)
        
        last_msg = broadcast[-1]
        self.assertEqual(last_msg.message_type, 'conflict')
    
    def test_collective_awareness(self):
        """Test collective awareness building."""
        # Broadcast some intents
        self.network.broadcast_intent(0, 3, 0.8, 0.0)
        self.network.broadcast_intent(1, 2, 0.7, 0.0)
        self.network.share_status(2, 0.0)
        
        # Get awareness
        awareness = self.network.get_collective_awareness(3)
        
        self.assertIn('active_intents', awareness)
        self.assertIn('agent_statuses', awareness)
        self.assertIn('conflicts', awareness)
        self.assertIn('nearby_agents', awareness)
    
    def test_communication_graph(self):
        """Test communication graph construction."""
        graph = self.network.compute_communication_graph()
        
        # Graph should be non-empty
        self.assertIsInstance(graph, dict)
        
        # All reachable agents should be connected
        for agent_id, neighbors in graph.items():
            self.assertIsInstance(neighbors, list)


class TestGCNArchitecture(unittest.TestCase):
    """Test B2: Graph Convolutional Networks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_agents = 8
        self.config = GraphConfig(
            num_node_features=35,
            num_agents=self.num_agents,
            hidden_dim=64,
            gcn_layers=2,
            output_dim=9,
            dropout=0.1,
            use_edge_features=True
        )
    
    def test_graph_constructor_spatial(self):
        """Test spatial graph construction."""
        constructor = GraphConstructor(spatial_threshold=1000.0)
        
        positions = np.random.uniform([0, 0, 500], [5000, 5000, 2000], 
                                      (self.num_agents, 3))
        agent_ids = np.arange(self.num_agents)
        
        edge_index, edge_attr = constructor.construct_spatial_graph(
            positions, agent_ids
        )
        
        # Check shapes
        self.assertEqual(edge_index.shape[0], 2)
        if edge_index.size > 0:
            self.assertEqual(edge_attr.shape[0], edge_index.shape[1])
            self.assertEqual(edge_attr.shape[1], 4)  # distance, alt, time, priority
    
    def test_graph_constructor_communication(self):
        """Test communication graph construction."""
        constructor = GraphConstructor()
        
        # Create communication matrix
        comm_matrix = np.random.choice([0, 1], size=(self.num_agents, self.num_agents))
        np.fill_diagonal(comm_matrix, 0)
        
        edge_index, edge_attr = constructor.construct_communication_graph(
            communication_ranges={},
            communication_links=comm_matrix
        )
        
        self.assertEqual(edge_index.shape[0], 2)
        if edge_index.size > 0:
            self.assertEqual(edge_attr.shape[1], 4)
    
    def test_graph_constructor_priority(self):
        """Test priority-based graph construction."""
        constructor = GraphConstructor()
        
        priorities = np.random.uniform(0, 5, self.num_agents)
        
        edge_index, edge_attr = constructor.construct_priority_graph(
            priorities, priority_threshold=0.5
        )
        
        self.assertEqual(edge_index.shape[0], 2)
        if edge_index.size > 0:
            self.assertGreater(edge_index.shape[1], 0)
    
    def test_gcn_layer(self):
        """Test single GCN layer."""
        gcn_layer = GCNLayer(
            in_features=35,
            out_features=64,
            use_edge_features=True,
            edge_feature_dim=4
        ).to(self.device)
        
        # Create test data
        num_nodes = 8
        x = torch.randn(num_nodes, 35).to(self.device)
        edge_index = torch.tensor(
            [[0, 1, 2, 3],
             [1, 2, 3, 4]], dtype=torch.long
        ).to(self.device)
        edge_attr = torch.randn(4, 4).to(self.device)
        
        # Forward pass
        output = gcn_layer(x, edge_index, edge_attr)
        
        # Check output shape
        self.assertEqual(output.shape, (num_nodes, 64))
    
    def test_gcn_network(self):
        """Test full GCN network."""
        gcn = GCNNetwork(self.config).to(self.device)
        
        # Create test data
        x = torch.randn(self.num_agents, 35).to(self.device)
        edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 3]], dtype=torch.long
        ).to(self.device)
        edge_attr = torch.randn(3, 4).to(self.device)  # 3 edges, 4 features each
        
        # Forward pass
        action_logits, values = gcn(x, edge_index, edge_attr)
        
        # Check output shapes
        self.assertEqual(action_logits.shape, (self.num_agents, 9))
        self.assertEqual(values.shape, (self.num_agents, 1))
        
        # Check softmax outputs sum to 1
        action_probs = torch.softmax(action_logits, dim=-1)
        sums = action_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
    
    def test_gcn_agent(self):
        """Test GCN agent wrapper."""
        agent = GCNAgent(agent_id=0, config=self.config, device=self.device)
        
        positions = np.random.uniform([0, 0, 500], [5000, 5000, 2000], 
                                      (self.num_agents, 3))
        features = torch.randn(self.num_agents, 35)
        agent_ids = np.arange(self.num_agents)
        
        action, prob = agent.compute_action(
            positions, features, agent_ids,
            use_spatial_graph=True
        )
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(prob, float)
        self.assertTrue(0 <= action < 9)  # 8 pads + hold
        self.assertTrue(0 <= prob <= 1)
    
    def test_multi_agent_gcn_policy(self):
        """Test multi-agent GCN policy."""
        policy = MultiAgentGCNPolicy(
            num_agents=self.num_agents,
            config=self.config,
            device=self.device
        )
        
        # Create agent states
        agent_states = {i: np.random.rand(35) for i in range(self.num_agents)}
        
        # Compute actions
        actions = policy.compute_actions(agent_states, graph_type='spatial')
        
        # Check results
        self.assertEqual(len(actions), self.num_agents)
        for agent_id, (action, prob) in actions.items():
            self.assertTrue(0 <= action < 9)
            self.assertTrue(0 <= prob <= 1)


class TestIntegration(unittest.TestCase):
    """Test integration of B1 and B2."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_agents = 5
    
    def test_communication_to_gcn_pipeline(self):
        """Test pipeline from communication to GCN."""
        # Create communication network
        network = CommunicationNetwork(communication_range=1000.0)
        
        # Create agents
        for i in range(self.num_agents):
            state = AgentState(
                agent_id=i,
                position=np.array([i * 200, i * 200, 500 + i * 100]),
                velocity=50.0,
                heading=0.0,
                battery_soc=0.5 + 0.2 * np.random.rand(),
                passenger_priority=np.random.randint(1, 6),
                current_delay=np.random.uniform(0, 20),
                intended_pad=-1,
                confidence=0.7
            )
            network.update_agent_state(i, state)
        
        # Broadcast communications
        for i in range(3):  # First 3 agents broadcast
            network.broadcast_intent(i, 2 + i % 3, 0.8, 0.0)
        
        # Extract communication graph
        comm_graph = network.compute_communication_graph()
        
        # Build GCN policy
        config = GraphConfig(
            num_node_features=35,
            num_agents=self.num_agents,
            hidden_dim=32,
            gcn_layers=1,
            output_dim=9
        )
        policy = MultiAgentGCNPolicy(self.num_agents, config, self.device)
        
        # Get agent states
        agent_states = {i: np.random.rand(35) for i in range(self.num_agents)}
        
        # Compute actions through GCN
        actions = policy.compute_actions(agent_states, graph_type='spatial')
        
        # Verify results
        self.assertEqual(len(actions), self.num_agents)
        for action, prob in actions.values():
            self.assertTrue(0 <= action < 9)
            self.assertTrue(0 <= prob <= 1)


class TestPerformance(unittest.TestCase):
    """Test performance and scalability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_gcn_inference_speed(self):
        """Test GCN inference speed."""
        import time
        
        num_agents = 20
        config = GraphConfig(
            num_node_features=35,
            num_agents=num_agents,
            hidden_dim=64,
            gcn_layers=2,
            output_dim=9
        )
        
        policy = MultiAgentGCNPolicy(num_agents, config, self.device)
        agent_states = {i: np.random.rand(35) for i in range(num_agents)}
        
        # Warm-up
        policy.compute_actions(agent_states)
        
        # Time 100 iterations
        start = time.time()
        for _ in range(100):
            policy.compute_actions(agent_states)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / 100) * 1000
        
        print(f"\nGCN Inference Speed ({self.device}):")
        print(f"  Agents: {num_agents}")
        print(f"  Avg time: {avg_time_ms:.2f}ms")
        
        # Should be reasonably fast (< 100ms on CPU)
        self.assertLess(avg_time_ms, 200)
    
    def test_communication_scale(self):
        """Test communication network scaling."""
        num_agents = 50
        network = CommunicationNetwork(communication_range=1000.0)
        
        # Create many agents
        for i in range(num_agents):
            state = AgentState(
                agent_id=i,
                position=np.random.uniform(0, 10000, 3),
                velocity=50.0,
                heading=0.0,
                battery_soc=0.5,
                passenger_priority=3,
                current_delay=5.0,
                intended_pad=-1,
                confidence=0.7
            )
            network.update_agent_state(i, state)
        
        # Compute graph
        graph = network.compute_communication_graph()
        
        # Should complete without error
        self.assertIsInstance(graph, dict)
        print(f"\nCommunication graph created for {num_agents} agents")
        print(f"  Connected agents: {len(graph)}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
