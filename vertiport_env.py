"""
VertiportEnv: Multi-Agent eVTOL Vertiport Landing Scheduling Environment
Author: MARL Vertiport Scheduling Project
Date: February 2026

This environment simulates a vertiport with multiple landing pads and eVTOL aircraft
competing for landing slots while maintaining safety separation constraints.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class AircraftType(IntEnum):
    """Different eVTOL aircraft types with varying characteristics"""
    LIGHT = 0      # 25 min endurance
    MEDIUM = 1     # 40 min endurance  
    HEAVY = 2      # 60 min endurance


@dataclass
class Aircraft:
    """Represents a single eVTOL aircraft"""
    id: int
    aircraft_type: AircraftType
    position: np.ndarray  # [x, y, altitude] in meters
    velocity: float  # m/s
    heading: float  # radians
    battery_soc: float  # State of Charge, 0-100%
    passenger_priority: int  # 1-5, higher is more urgent
    eta_minutes: float  # Estimated time to arrival
    target_pad: Optional[int] = None
    landing_time: Optional[float] = None
    
    def __post_init__(self):
        """Set endurance based on aircraft type"""
        self.max_endurance = {
            AircraftType.LIGHT: 25,
            AircraftType.MEDIUM: 40,
            AircraftType.HEAVY: 60
        }[self.aircraft_type]


@dataclass
class LandingPad:
    """Represents a landing pad (TLOF)"""
    id: int
    position: np.ndarray  # [x, y] in meters
    occupied: bool = False
    cooldown_remaining: float = 0.0  # minutes
    cooldown_period: float = 3.0  # minutes between uses
    compatible_types: List[AircraftType] = None
    
    def __post_init__(self):
        if self.compatible_types is None:
            self.compatible_types = list(AircraftType)  # All types by default


class VertiportEnv:
    """
    Multi-Agent Vertiport Scheduling Environment
    
    State Space:
    - Vertiport graph with 8 landing pads
    - 10-50 aircraft at various approach altitudes
    - Separation constraints and pad availability
    
    Action Space (per agent):
    - Select target pad (0-7) or hold
    - Descent rate (slow/normal)
    
    Reward:
    - Throughput bonus for successful landings
    - Delay penalty
    - Safety penalty for violations
    - Energy efficiency bonus
    """
    
    def __init__(
        self,
        num_pads: int = 8,
        arrival_rate: float = 20.0,  # aircraft per hour
        max_aircraft: int = 50,
        dt: float = 0.1,  # timestep in minutes (6 seconds)
        separation_distance: float = 500.0,  # meters
        render_mode: str = None
    ):
        self.num_pads = num_pads
        self.arrival_rate = arrival_rate
        self.max_aircraft = max_aircraft
        self.dt = dt
        self.separation_distance = separation_distance
        self.render_mode = render_mode
        
        # Approach altitude rings (meters)
        self.approach_altitudes = [1500, 1000, 500, 0]
        
        # Initialize pads in 2x4 grid, 150m apart
        self.pads = self._create_pads()
        
        # State tracking
        self.aircraft: List[Aircraft] = []
        self.current_time = 0.0  # minutes
        self.next_arrival_time = 0.0
        self.total_landings = 0
        self.total_delay = 0.0
        self.separation_violations = 0
        
        # Metrics
        self.episode_metrics = {
            'landings': [],
            'delays': [],
            'violations': [],
            'throughput': [],
            'pad_utilization': []
        }
        
    def _create_pads(self) -> List[LandingPad]:
        """Create landing pads in 2x4 grid configuration"""
        pads = []
        rows, cols = 2, 4
        spacing = 150.0  # meters
        
        for i in range(self.num_pads):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            
            # Vary cooldown periods slightly for heterogeneity
            cooldown = 3.0 + np.random.uniform(-0.5, 0.5)
            
            pad = LandingPad(
                id=i,
                position=np.array([x, y]),
                cooldown_period=cooldown
            )
            pads.append(pad)
            
        return pads
    
    def _generate_arrival(self) -> Aircraft:
        """Generate a new aircraft arrival using Poisson process"""
        # Determine arrival position (random bearing at 1500m altitude)
        bearing = np.random.uniform(0, 2 * np.pi)
        distance = 2000.0  # meters from center
        x = distance * np.cos(bearing)
        y = distance * np.sin(bearing)
        
        # Random aircraft type
        aircraft_type = np.random.choice(list(AircraftType))
        
        # Battery state: most have plenty, some are critical
        if np.random.random() < 0.15:  # 15% are low on battery
            battery_soc = np.random.uniform(10, 25)
        else:
            battery_soc = np.random.uniform(40, 90)
        
        # Passenger priority
        priority = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.07, 0.03])
        
        aircraft = Aircraft(
            id=len(self.aircraft) + self.total_landings,
            aircraft_type=aircraft_type,
            position=np.array([x, y, self.approach_altitudes[0]]),
            velocity=30.0,  # m/s typical cruise
            heading=bearing + np.pi,  # heading toward center
            battery_soc=battery_soc,
            passenger_priority=priority,
            eta_minutes=distance / (30.0 * 60)  # rough estimate
        )
        
        return aircraft
    
    def reset(self) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        self.aircraft = []
        self.current_time = 0.0
        self.next_arrival_time = np.random.exponential(60.0 / self.arrival_rate)
        self.total_landings = 0
        self.total_delay = 0.0
        self.separation_violations = 0
        
        # Reset pads
        for pad in self.pads:
            pad.occupied = False
            pad.cooldown_remaining = 0.0
        
        # Generate initial aircraft
        initial_aircraft = max(3, int(self.arrival_rate / 6))
        for _ in range(initial_aircraft):
            self.aircraft.append(self._generate_arrival())
        
        return self._get_observations(), self._get_info()
    
    def _get_observations(self) -> Dict:
        """Get observations for all agents"""
        observations = {}
        
        for aircraft in self.aircraft:
            obs = self._get_aircraft_observation(aircraft)
            observations[aircraft.id] = obs
            
        return observations
    
    def _get_aircraft_observation(self, aircraft: Aircraft) -> np.ndarray:
        """
        Get observation for a single aircraft
        
        Observation includes:
        - Own state: position (3), velocity (1), heading (1), battery (1), priority (1)
        - Pad states: for each pad: available (1), cooldown (1), distance (1)
        - Other aircraft: count in each altitude ring
        
        Total dimension: 7 + 3*num_pads + 4 = 35 for 8 pads
        """
        obs = []
        
        # Own state
        obs.extend(aircraft.position / 2000.0)  # normalized position
        obs.append(aircraft.velocity / 50.0)  # normalized velocity
        obs.append(aircraft.heading / (2 * np.pi))  # normalized heading
        obs.append(aircraft.battery_soc / 100.0)  # normalized battery
        obs.append(aircraft.passenger_priority / 5.0)  # normalized priority
        
        # Pad states
        for pad in self.pads:
            available = 0.0 if (pad.occupied or pad.cooldown_remaining > 0) else 1.0
            cooldown_norm = pad.cooldown_remaining / pad.cooldown_period
            distance = np.linalg.norm(aircraft.position[:2] - pad.position)
            distance_norm = min(distance / 3000.0, 1.0)
            
            obs.extend([available, cooldown_norm, distance_norm])
        
        # Other aircraft in altitude rings
        for altitude in self.approach_altitudes:
            count = sum(1 for a in self.aircraft 
                       if abs(a.position[2] - altitude) < 100 and a.id != aircraft.id)
            obs.append(min(count / 10.0, 1.0))  # normalized count
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get environment info"""
        pad_utilization = sum(1 for pad in self.pads if pad.occupied) / self.num_pads
        
        return {
            'current_time': self.current_time,
            'num_aircraft': len(self.aircraft),
            'total_landings': self.total_landings,
            'avg_delay': self.total_delay / max(self.total_landings, 1),
            'violations': self.separation_violations,
            'pad_utilization': pad_utilization
        }
    
    def get_action_mask(self, aircraft: Aircraft) -> np.ndarray:
        """
        Get valid action mask for an aircraft
        
        Actions: 0-7 = land on pad 0-7, 8 = hold/wait
        """
        mask = np.zeros(self.num_pads + 1, dtype=bool)
        
        # Check each pad
        for i, pad in enumerate(self.pads):
            # Pad must be available
            if pad.occupied or pad.cooldown_remaining > 0:
                mask[i] = False
            # Must be at appropriate altitude (within 600m to land)
            elif aircraft.position[2] > 600:  # Too high to land
                mask[i] = False
            # Check separation from other aircraft targeting this pad
            elif self._check_separation_violation(aircraft, pad):
                mask[i] = False
            else:
                mask[i] = True
        
        # Hold action always available
        mask[-1] = True
        
        return mask
    
    def _check_separation_violation(self, aircraft: Aircraft, pad: LandingPad) -> bool:
        """Check if landing would violate separation with other aircraft"""
        for other in self.aircraft:
            if other.id == aircraft.id:
                continue
            
            # Check horizontal separation
            distance = np.linalg.norm(aircraft.position[:2] - other.position[:2])
            if distance < self.separation_distance:
                # Check if other aircraft is near the same pad
                other_pad_dist = np.linalg.norm(other.position[:2] - pad.position)
                if other_pad_dist < self.separation_distance:
                    return True
        
        return False
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one timestep
        
        Args:
            actions: Dict mapping aircraft_id to action (pad_id or hold=8)
            
        Returns:
            observations, rewards, terminated, truncated, info
        """
        rewards = {aid: 0.0 for aid in actions.keys()}
        terminated = {aid: False for aid in actions.keys()}
        truncated = {aid: False for aid in actions.keys()}
        
        # Update pad cooldowns
        for pad in self.pads:
            if pad.cooldown_remaining > 0:
                pad.cooldown_remaining = max(0, pad.cooldown_remaining - self.dt)
                if pad.cooldown_remaining == 0:
                    pad.occupied = False
        
        # Process actions
        landed_aircraft = []
        for aircraft in self.aircraft:
            if aircraft.id not in actions:
                continue
                
            action = actions[aircraft.id]
            
            # Hold action
            if action == self.num_pads:
                # Penalize holding, especially for low battery
                hold_penalty = -0.5
                if aircraft.battery_soc < 20:
                    hold_penalty *= 3
                rewards[aircraft.id] += hold_penalty
                
                # Descend to approach altitude if high
                if aircraft.position[2] > 100:
                    descent_rate = 200 * self.dt  # Faster descent
                    aircraft.position[2] = max(0, aircraft.position[2] - descent_rate)
                
                # Consume battery
                aircraft.battery_soc -= 0.2 * self.dt
                
            # Land on pad
            elif action < self.num_pads:
                pad = self.pads[action]
                
                # Check if action is valid
                if pad.occupied or pad.cooldown_remaining > 0:
                    rewards[aircraft.id] -= 50  # Invalid action penalty
                    continue
                
                # Execute landing
                pad.occupied = True
                pad.cooldown_remaining = pad.cooldown_period
                aircraft.landing_time = self.current_time
                landed_aircraft.append(aircraft)
                
                # Calculate delay (time from first appearance)
                delay = self.current_time - (aircraft.eta_minutes * 0.8)
                self.total_delay += max(0, delay)
                
                # Reward structure
                landing_reward = 100  # Base throughput reward
                
                # Delay penalty
                delay_penalty = -2 * max(0, delay)
                
                # Battery priority bonus
                battery_bonus = 0
                if aircraft.battery_soc < 20:
                    battery_bonus = 50
                elif aircraft.battery_soc < 30:
                    battery_bonus = 20
                
                # Passenger priority bonus
                priority_bonus = aircraft.passenger_priority * 5
                
                total_reward = landing_reward + delay_penalty + battery_bonus + priority_bonus
                rewards[aircraft.id] = total_reward
                terminated[aircraft.id] = True
                
                self.total_landings += 1
        
        # Remove landed aircraft
        self.aircraft = [a for a in self.aircraft if a not in landed_aircraft]
        
        # Generate new arrivals (Poisson process)
        while self.current_time >= self.next_arrival_time and len(self.aircraft) < self.max_aircraft:
            self.aircraft.append(self._generate_arrival())
            self.next_arrival_time += np.random.exponential(60.0 / self.arrival_rate)
        
        # Advance time
        self.current_time += self.dt
        
        # Get new observations
        observations = self._get_observations()
        info = self._get_info()
        
        # Update episode metrics
        if self.total_landings > 0:
            self.episode_metrics['avg_delay'] = self.total_delay / self.total_landings
        
        return observations, rewards, terminated, truncated, info
    
    def render(self):
        """Render the current state (placeholder for visualization)"""
        if self.render_mode is None:
            return
        
        print(f"\n=== Time: {self.current_time:.1f} min ===")
        print(f"Aircraft in airspace: {len(self.aircraft)}")
        print(f"Total landings: {self.total_landings}")
        print(f"Avg delay: {self.total_delay / max(self.total_landings, 1):.2f} min")
        print(f"Separation violations: {self.separation_violations}")
        
        # Pad status
        print("\nPad Status:")
        for pad in self.pads:
            status = "OCCUPIED" if pad.occupied else f"COOLDOWN {pad.cooldown_remaining:.1f}" if pad.cooldown_remaining > 0 else "AVAILABLE"
            print(f"  Pad {pad.id}: {status}")
        
        # Aircraft summary
        if self.aircraft:
            print("\nAircraft:")
            for aircraft in self.aircraft[:5]:  # Show first 5
                print(f"  ID {aircraft.id}: Alt {aircraft.position[2]:.0f}m, Battery {aircraft.battery_soc:.1f}%, Priority {aircraft.passenger_priority}")


def run_test():
    """Test the environment"""
    print("Testing VertiportEnv...")
    
    env = VertiportEnv(num_pads=8, arrival_rate=20, render_mode='human')
    obs, info = env.reset()
    
    print(f"\nInitial state:")
    print(f"  Number of aircraft: {info['num_aircraft']}")
    print(f"  Observation shape: {list(obs.values())[0].shape}")
    
    # Run for 50 steps with random valid actions
    for step in range(50):
        actions = {}
        for aircraft in env.aircraft:
            mask = env.get_action_mask(aircraft)
            valid_actions = np.where(mask)[0]
            actions[aircraft.id] = np.random.choice(valid_actions)
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 10 == 0:
            env.render()
    
    print(f"\n=== Final Statistics ===")
    print(f"Total landings: {env.total_landings}")
    print(f"Average delay: {env.total_delay / max(env.total_landings, 1):.2f} minutes")
    print(f"Separation violations: {env.separation_violations}")
    print(f"Final pad utilization: {info['pad_utilization']:.1%}")


if __name__ == "__main__":
    run_test()
