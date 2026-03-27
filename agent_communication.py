"""
Agent Communication Protocols for Multi-Agent RL

Implements communication mechanisms for agents to coordinate landing operations:
1. Intent Broadcasting: Agents announce landing intentions
2. Status Sharing: Share battery, priority, delay information
3. Conflict Resolution: Negotiate pad assignments
4. Collective Awareness: Maintain shared airspace picture

This enables decentralized coordination while maintaining safety constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import heapq
from collections import defaultdict


@dataclass
class AgentMessage:
    """Message structure for agent-to-agent communication."""
    sender_id: int
    receiver_id: int  # -1 for broadcast
    message_type: str  # 'intent', 'status', 'conflict', 'ack'
    timestamp: float
    content: Dict[str, Any]
    priority: int = 0
    
    def __lt__(self, other):
        """For priority queue ordering."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.timestamp < other.timestamp


@dataclass
class AgentState:
    """Internal state representation for communication."""
    agent_id: int
    position: np.ndarray  # [x, y, z]
    velocity: float
    heading: float
    battery_soc: float  # State of charge (0-1)
    passenger_priority: int  # 1-5
    current_delay: float  # minutes
    intended_pad: int  # -1 if no intent
    confidence: float  # 0-1, confidence in landing attempt


class MessageBuffer:
    """Manages incoming and outgoing messages with priority queuing."""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.incoming = []  # Priority queue
        self.outgoing = []  # Priority queue
        self.max_buffer_size = max_buffer_size
        self.delivery_history = defaultdict(list)
        
    def send(self, message: AgentMessage):
        """Queue an outgoing message."""
        if len(self.outgoing) < self.max_buffer_size:
            heapq.heappush(self.outgoing, message)
    
    def receive(self, message: AgentMessage):
        """Queue an incoming message."""
        if len(self.incoming) < self.max_buffer_size:
            heapq.heappush(self.incoming, message)
            self.delivery_history[message.receiver_id].append(message)
    
    def get_incoming(self, agent_id: int, num_messages: int = 5) -> List[AgentMessage]:
        """Retrieve top priority incoming messages for an agent."""
        messages = []
        temp = []
        
        while self.incoming and len(messages) < num_messages:
            msg = heapq.heappop(self.incoming)
            if msg.receiver_id == agent_id or msg.receiver_id == -1:
                messages.append(msg)
            else:
                temp.append(msg)
        
        # Put back messages not for this agent
        for msg in temp:
            heapq.heappush(self.incoming, msg)
        
        return messages
    
    def flush_outgoing(self) -> List[AgentMessage]:
        """Get all queued outgoing messages and clear buffer."""
        messages = []
        while self.outgoing:
            messages.append(heapq.heappop(self.outgoing))
        return messages


class CommunicationNetwork:
    """Network managing agent communication with spatial awareness."""
    
    def __init__(self, communication_range: float = 1000.0):
        """
        Initialize communication network.
        
        Args:
            communication_range: Maximum distance for direct communication (meters)
        """
        self.communication_range = communication_range
        self.agents_state = {}  # agent_id -> AgentState
        self.message_buffer = MessageBuffer()
        self.broadcast_hub = []  # For broadcasts via infrastructure
        
    def update_agent_state(self, agent_id: int, state: AgentState):
        """Update an agent's position and status."""
        self.agents_state[agent_id] = state
    
    def compute_communication_graph(self) -> Dict[int, List[int]]:
        """
        Compute which agents can communicate directly based on range.
        
        Returns:
            Dict mapping agent_id to list of reachable agent_ids
        """
        graph = defaultdict(list)
        agent_ids = list(self.agents_state.keys())
        
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i+1:]:
                state_a = self.agents_state[agent_a]
                state_b = self.agents_state[agent_b]
                
                distance = np.linalg.norm(state_a.position - state_b.position)
                
                if distance <= self.communication_range:
                    graph[agent_a].append(agent_b)
                    graph[agent_b].append(agent_a)
        
        return dict(graph)
    
    def broadcast_intent(self, agent_id: int, intended_pad: int, 
                         confidence: float, timestamp: float):
        """
        Agent broadcasts its landing intention.
        
        Args:
            agent_id: Broadcasting agent
            intended_pad: Pad the agent intends to land on (-1 for undecided)
            confidence: Confidence level (0-1)
            timestamp: Current simulation time
        """
        content = {
            'intended_pad': intended_pad,
            'confidence': confidence,
            'agent_state': self.agents_state.get(agent_id)
        }
        
        message = AgentMessage(
            sender_id=agent_id,
            receiver_id=-1,  # Broadcast
            message_type='intent',
            timestamp=timestamp,
            content=content,
            priority=2 if confidence > 0.8 else 1
        )
        
        # Add to broadcast hub first
        self.broadcast_hub.append(message)
        self.message_buffer.send(message)
    
    def share_status(self, agent_id: int, timestamp: float):
        """
        Agent shares its current status (battery, delay, constraints).
        
        Args:
            agent_id: Sharing agent
            timestamp: Current simulation time
        """
        state = self.agents_state.get(agent_id)
        if state is None:
            return
        
        content = {
            'battery_soc': state.battery_soc,
            'current_delay': state.current_delay,
            'passenger_priority': state.passenger_priority,
            'position': state.position.tolist()
        }
        
        message = AgentMessage(
            sender_id=agent_id,
            receiver_id=-1,  # Broadcast
            message_type='status',
            timestamp=timestamp,
            content=content,
            priority=3 if state.battery_soc < 0.2 else 1
        )
        
        self.broadcast_hub.append(message)
        self.message_buffer.send(message)
    
    def negotiate_conflict(self, agent_a: int, agent_b: int, 
                          contested_pad: int, timestamp: float):
        """
        Resolve conflict when multiple agents want same pad.
        
        Conflict resolution: Higher priority wins
        Priority = battery_urgency * 100 + passenger_priority * 10 - age_penalty
        
        Args:
            agent_a, agent_b: Conflicting agents
            contested_pad: Padid causing conflict
            timestamp: Current simulation time
            
        Returns:
            winner_id: Agent with higher priority
        """
        state_a = self.agents_state.get(agent_a)
        state_b = self.agents_state.get(agent_b)
        
        if state_a is None or state_b is None:
            return None
        
        # Calculate priorities
        priority_a = self._calculate_priority(state_a, timestamp)
        priority_b = self._calculate_priority(state_b, timestamp)
        
        winner_id = agent_a if priority_a >= priority_b else agent_b
        loser_id = agent_b if winner_id == agent_a else agent_a
        
        # Send conflict resolution message
        content = {
            'contested_pad': contested_pad,
            'winner_id': winner_id,
            'winner_priority': max(priority_a, priority_b),
            'loser_priority': min(priority_a, priority_b)
        }
        
        message = AgentMessage(
            sender_id=0,  # From coordinator
            receiver_id=-1,
            message_type='conflict',
            timestamp=timestamp,
            content=content,
            priority=5  # High priority
        )
        
        self.broadcast_hub.append(message)
        self.message_buffer.send(message)
        
        return winner_id
    
    def _calculate_priority(self, state: AgentState, timestamp: float) -> float:
        """
        Calculate agent priority for conflict resolution.
        
        Priority = battery_urgency * 100 
                 + passenger_priority * 10 
                 - delay_age * 0.1
        """
        battery_urgency = max(0, (0.2 - state.battery_soc) / 0.2)  # 0-1
        delay_age = max(0, state.current_delay)  # minutes
        
        priority = (battery_urgency * 100 + 
                   state.passenger_priority * 10 - 
                   delay_age * 0.1)
        
        return priority
    
    def get_collective_awareness(self, agent_id: int) -> Dict[str, Any]:
        """
        Build shared airspace picture from received messages.
        
        Args:
            agent_id: Requesting agent
            
        Returns:
            Dict with collective awareness: active intents, status updates, conflicts
        """
        messages = self.message_buffer.get_incoming(agent_id, num_messages=10)
        
        awareness = {
            'active_intents': [],
            'agent_statuses': {},
            'conflicts': [],
            'nearby_agents': []
        }
        
        comm_graph = self.compute_communication_graph()
        nearby = comm_graph.get(agent_id, [])
        
        for msg in messages:
            if msg.message_type == 'intent':
                awareness['active_intents'].append({
                    'from_agent': msg.sender_id,
                    'intended_pad': msg.content['intended_pad'],
                    'confidence': msg.content['confidence']
                })
            elif msg.message_type == 'status':
                awareness['agent_statuses'][msg.sender_id] = msg.content
            elif msg.message_type == 'conflict':
                awareness['conflicts'].append(msg.content)
        
        awareness['nearby_agents'] = nearby
        
        return awareness
    
    def get_communication_messages(self, agent_id: int) -> List[AgentMessage]:
        """Get all messages for an agent."""
        return self.message_buffer.get_incoming(agent_id)
    
    def get_broadcast_history(self, num_recent: int = 10) -> List[AgentMessage]:
        """Get recent broadcast messages for debugging."""
        return self.broadcast_hub[-num_recent:]


class DecentralizedCoordinator:
    """
    Decentralized coordinator using local agent communication.
    
    Enables distributed decision-making where agents coordinate through messages
    rather than centralized authority.
    """
    
    def __init__(self, communication_range: float = 1000.0, 
                 max_message_hops: int = 3):
        self.network = CommunicationNetwork(communication_range)
        self.max_message_hops = max_message_hops
        self.coordination_history = []
        
    def step(self, timestamp: float) -> Dict[str, Any]:
        """
        Execute one coordination step.
        
        Returns:
            Coordination results with resolved conflicts and shared information
        """
        results = {
            'messages_processed': 0,
            'conflicts_resolved': 0,
            'intents_broadcast': 0,
            'status_updates': 0
        }
        
        # Process all queued messages
        messages = self.network.message_buffer.flush_outgoing()
        results['messages_processed'] = len(messages)
        
        for msg in messages:
            if msg.message_type == 'conflict':
                results['conflicts_resolved'] += 1
            elif msg.message_type == 'intent':
                results['intents_broadcast'] += 1
            elif msg.message_type == 'status':
                results['status_updates'] += 1
        
        self.coordination_history.append(results)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication and coordination statistics."""
        stats = {
            'total_messages': len(self.network.broadcast_hub),
            'avg_messages_per_step': 0,
            'total_conflicts_resolved': 0,
            'communication_graph_connected': False
        }
        
        if self.coordination_history:
            total_conflicts = sum(h['conflicts_resolved'] 
                                for h in self.coordination_history)
            stats['total_conflicts_resolved'] = total_conflicts
            
            if len(self.coordination_history) > 0:
                stats['avg_messages_per_step'] = (
                    sum(h['messages_processed'] 
                        for h in self.coordination_history) / 
                    len(self.coordination_history)
                )
        
        # Check connectivity
        comm_graph = self.network.compute_communication_graph()
        if comm_graph and len(comm_graph) > 0:
            stats['communication_graph_connected'] = self._is_graph_connected(comm_graph)
        
        return stats
    
    def _is_graph_connected(self, graph: Dict[int, List[int]]) -> bool:
        """Check if communication graph is connected."""
        if not graph:
            return False
        
        visited = set()
        start_node = next(iter(graph.keys()))
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(graph.get(node, []))
        
        return len(visited) == len(graph)
