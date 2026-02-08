# MARL eVTOL Vertiport Scheduling 

**Multi-Agent Reinforcement Learning for Safe eVTOL Vertiport Landing Scheduling**

## Project Overview

This project implements a Multi-Agent Reinforcement Learning system for optimizing eVTOL (electric Vertical Take-Off and Landing) aircraft landing scheduling at urban vertiports. The system addresses the critical bottleneck in Urban Air Mobility (UAM) networks where limited landing pads (4-16 per facility) must serve high volumes of aircraft while maintaining strict safety constraints.

### Key Objectives
- **Reduce landing delays** from 15-45 min (FCFS) to <5 min target
- **Improve throughput** by 25-35% over baseline methods
- **Maintain safety** through hard constraint enforcement
- **Optimize multi-objective** tradeoffs (delay, energy, safety)



## What's in This Prototype

### Core Components

1. **`vertiport_env.py`** - Multi-Agent Vertiport Environment
   - Graph-based state representation
   - 8 landing pads in 2×4 grid configuration
   - Aircraft with heterogeneous properties (battery, type, priority)
   - Action masking for safety constraints
   - Separation enforcement (500m zones)
   - Poisson arrival process simulation

2. **`baselines.py`** - Baseline Schedulers
   - **FCFS (First-Come-First-Served)**: Traditional queuing
   - **Greedy**: Smart heuristic with battery/priority awareness
   - Benchmark comparison framework

3. **`visualization.py`** - Dashboard Visualization
   - Real-time top-down vertiport view
   - Aircraft positions colored by altitude
   - Pad status indicators
   - Metrics tracking (delay, throughput, utilization)

4. **`demo.py`** - Comprehensive Demonstration
   - Environment walkthrough
   - Baseline comparison
   - Visualization generation



## Architecture

### State Space (per aircraft)
```
Observation (35D for 8 pads):
├── Own state (7D)
│   ├── Position (x, y, altitude) - normalized
│   ├── Velocity, heading
│   ├── Battery SoC (%)
│   └── Passenger priority (1-5)
├── Pad states (24D = 8 pads × 3 features)
│   ├── Available (0/1)
│   ├── Cooldown remaining
│   └── Distance to pad
└── Airspace density (4D)
    └── Aircraft count per altitude ring
```

### Action Space
```
Actions (9 discrete):
├── 0-7: Land on pad 0-7
└── 8: Hold at current altitude
```

### Reward Structure
```python
reward = throughput_bonus          # +100 for successful landing
       + delay_penalty             # -2 × delay_minutes
       + battery_priority_bonus    # +50 if battery < 20%
       + passenger_priority_bonus  # +5 × priority_level
       + energy_penalty            # -0.1 × hold_energy
```



## Quick Start

### Run the Demo
```bash
cd marl
python3 demo.py
```

This will:
1. Demonstrate environment basics
2. Compare FCFS vs Greedy schedulers
3. Generate visualization dashboard
4. Output performance metrics

### Individual Components

**Test environment only:**
```bash
python vertiport_env.py
```

**Compare baselines:**
```bash
python baselines.py
```

**Generate visualization:**
```bash
python visualization.py
```


## Expected Results

### Baseline Performance (20 aircraft/hour)

| Metric | FCFS | Greedy | Target (MARL) |
|--------|------|--------|---------------|
| Avg Landing Delay | 15-20 min | 10-15 min | **<5 min** |
| Throughput | 35-40 ac/hr | 40-45 ac/hr | **45-50 ac/hr** |
| Pad Utilization | 65-70% | 70-75% | **80-85%** |
| Separation Violations | 1-2 per 100 | 0-1 per 100 | **0 (hard constraint)** |

### Greedy vs FCFS Improvements
- **~25% delay reduction** through battery-aware prioritization
- **~12% throughput increase** from smarter pad assignment
- **~8% utilization gain** via nearest-pad selection



## Technical Details

### Environment Features

**Aircraft Heterogeneity:**
- 3 types: Light (25 min), Medium (40 min), Heavy (60 min) endurance
- Battery state: 10-90% SoC, with 15% critical (<20%)
- Passenger priority: 1-5 scale (higher = more urgent)

**Safety Constraints:**
- 500m horizontal separation zones around each pad
- Action masking prevents invalid actions (occupied pads, collisions)
- Pad cooldown: 3-5 min between departures

**Demand Modeling:**
- Poisson arrival process (λ = 5-50 aircraft/hour)
- Configurable density for curriculum learning

**Vertiport Configuration:**
- 8 TLOF (Touchdown and Lift-Off) pads
- 2×4 grid layout, 150m spacing
- Approach altitude rings: 1500m, 1000m, 500m

### Baseline Algorithms

**FCFS (First-Come-First-Served):**
```python
1. Sort aircraft by arrival time
2. Prioritize critical battery (<20%)
3. Assign first available pad in order
4. Hold if no pad available
```

**Greedy Heuristic:**
```python
priority_score = battery_urgency × 100
               + passenger_priority × 10  
               - distance_to_nearest_pad

1. Calculate priority for all aircraft
2. Sort by priority (descending)
3. Assign to nearest available pad
4. Hold if no valid assignment
```



## Next Steps (Week 3-4)

### Independent PPO Training
- [ ] Implement PPO agents using stable baselines
- [ ] Train on low-density scenario (10 arrivals/hour)
- [ ] Validate learning curves
- [ ] Measure convergence time

### Ray RLlib Integration  
- [ ] Set up distributed training
- [ ] Configure multi-GPU if available
- [ ] Implement experiment logging (Weights & Biases)

### Performance Validation
- [ ] Compare PPO vs Greedy vs FCFS
- [ ] Measure sample efficiency
- [ ] Test on 10, 15, 20 arrivals/hour



## Requirements

### Core Dependencies
```
Python 3.8+
numpy >= 1.21
matplotlib >= 3.5
```

### Future Dependencies (Week 3+)
```
gymnasium >= 0.29
pettingzoo >= 1.24
ray[rllib] >= 2.20
torch >= 2.0
torch-geometric >= 2.4
```



## File Structure

```
marl_vertiport/
├── vertiport_env.py      # Core environment (350 lines)
├── baselines.py          # FCFS and Greedy schedulers (200 lines)
├── visualization.py      # Dashboard visualization (250 lines)
├── demo.py              # Comprehensive demo (200 lines)
├── README.md            # This file
└── outputs/
    ├── vertiport_dashboard.png
    └── demo_visualization.png
```



## Educational Value

### Skills Demonstrated
1. **Multi-Agent Systems**: Decentralized decision-making with coordination
2. **Reinforcement Learning**: MDP formulation, reward shaping, action masking
3. **Safety-Critical AI**: Hard constraint enforcement, separation zones
4. **Simulation Engineering**: Custom Gym environment, Poisson processes
5. **Domain Modeling**: Aviation operations, vertiport constraints

### Key Concepts Implemented
- Graph-based state representation
- Action masking for safety
- Heterogeneous agent properties
- Multi-objective reward balancing
- Curriculum learning readiness (density scaling)



## References

Based on research proposal aligning with:
- FAA AC 150/5390-2D (Vertiport Design Standards)
- ASTM F3423-22 (Operational Standards)
- NASA STRIVES eVTOL datasets
- NMCAD graph-based control research

Academic foundations:
- Krishnan et al. (2023): Graph Learning for UAM Take-Off/Landing
- Govindan et al. (2024): Throughput Maximizing eVTOL Scheduling
- Lowe et al. (2017): Multi-Agent Actor-Critic (QMIX foundations)



## Week 1-2 Deliverables - COMPLETE

- [x] VertiportEnv implementation with graph-based state
- [x] FCFS baseline scheduler
- [x] Greedy heuristic baseline
- [x] Benchmark comparison framework
- [x] Visualization dashboard
- [x] Comprehensive documentation
- [x] Reproducible demo script



## Success Criteria Met

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Environment functional | Completed | Multi-agent environment with 8 pads, Poisson arrivals |
| Baselines implemented | Completed | FCFS and Greedy schedulers operational |
| Metrics tracked | Completed | Delay, throughput, utilization, violations |
| Visualization working | Completed | Real-time dashboard with 4 panels |
| Documentation complete | Completed | README, inline comments, docstrings |
| Reproducible | Completed | Single `demo.py` runs all components |



## Design Highlights

### Why This Architecture?

1. **Graph-Based State**: Natural representation for vertiport topology
   - Pads as nodes, separation as edges
   - Enables GCN integration in Week 5-6

2. **Action Masking**: Safety-first approach
   - Hard constraints prevent invalid actions
   - No learning from unsafe experiences needed

3. **Curriculum Ready**: Scalable from 10→50 arrivals/hour
   - Environment parameterized by arrival_rate
   - Baseline tested at multiple densities

4. **Multi-Objective**: Real-world tradeoffs
   - Not just throughput - also safety, energy, passenger experience
   - Reward components can be reweighted

5. **Modular Design**: Easy to extend
   - Schedulers inherit from base class
   - Environment agnostic to scheduler type
   - Visualization decoupled from training


## Known Limitations

1. **No wind dynamics yet** - Will add in Week 7 robustness testing
2. **Simplified kinematics** - Constant velocity, no complex flight dynamics  
3. **Perfect information** - Full observability (will add partial obs later)
4. **No communication delays** - Added in perturbation testing
5. **Deterministic separation** - Future: probabilistic collision modeling


