"""
Visualization Module for Vertiport Environment
Creates 2D top-down view of vertiport operations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation
from typing import List, Optional
from vertiport_env import VertiportEnv, Aircraft, LandingPad


class VertiportVisualizer:
    """
    Visualize vertiport operations in real-time
    """
    
    def __init__(self, env: VertiportEnv, figsize=(12, 10)):
        self.env = env
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Vertiport Operations Dashboard', fontsize=16, fontweight='bold')
        
        # Main view axis
        self.ax_main = self.axes[0, 0]
        self.ax_metrics = self.axes[0, 1]
        self.ax_delay = self.axes[1, 0]
        self.ax_utilization = self.axes[1, 1]
        
        # Data tracking
        self.time_history = []
        self.delay_history = []
        self.throughput_history = []
        self.utilization_history = []
        
    def render_frame(self, save_path: Optional[str] = None):
        """Render current state of the environment"""
        
        # Clear all axes
        for ax in [self.ax_main, self.ax_metrics, self.ax_delay, self.ax_utilization]:
            ax.clear()
        
        # === Main View: Top-down vertiport ===
        self._render_main_view()
        
        # === Metrics Panel ===
        self._render_metrics_panel()
        
        # === Delay Over Time ===
        self._render_delay_plot()
        
        # === Pad Utilization ===
        self._render_utilization_plot()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.pause(0.01)
    
    def _render_main_view(self):
        """Render top-down view of vertiport"""
        ax = self.ax_main
        ax.set_title('Vertiport Top-Down View', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw landing pads
        for pad in self.env.pads:
            if pad.occupied:
                color = 'red'
                alpha = 0.8
            elif pad.cooldown_remaining > 0:
                color = 'orange'
                alpha = 0.6
            else:
                color = 'green'
                alpha = 0.4
            
            # Pad as circle
            circle = Circle(
                pad.position, 
                radius=30, 
                color=color, 
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(circle)
            
            # Pad number
            ax.text(
                pad.position[0], 
                pad.position[1], 
                f'P{pad.id}',
                ha='center', 
                va='center',
                fontsize=10,
                fontweight='bold',
                color='white' if pad.occupied else 'black',
                zorder=3
            )
            
            # Separation zone
            separation_circle = Circle(
                pad.position,
                radius=self.env.separation_distance,
                color='gray',
                fill=False,
                linestyle='--',
                alpha=0.2,
                zorder=1
            )
            ax.add_patch(separation_circle)
        
        # Draw aircraft
        for aircraft in self.env.aircraft:
            # Color by altitude
            if aircraft.position[2] > 1000:
                color = 'blue'
                marker = '^'
                size = 100
            elif aircraft.position[2] > 500:
                color = 'cyan'
                marker = '^'
                size = 80
            else:
                color = 'purple'
                marker = 'v'
                size = 120
            
            # Color intensity by battery
            alpha = 0.4 + 0.6 * (aircraft.battery_soc / 100.0)
            
            ax.scatter(
                aircraft.position[0],
                aircraft.position[1],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors='black',
                linewidth=1.5,
                zorder=4
            )
            
            # Show critical battery aircraft
            if aircraft.battery_soc < 20:
                ax.scatter(
                    aircraft.position[0],
                    aircraft.position[1],
                    c='none',
                    marker='o',
                    s=size * 2,
                    edgecolors='red',
                    linewidth=3,
                    zorder=4
                )
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Pad: Available', alpha=0.6),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                      markersize=10, label='Pad: Cooldown', alpha=0.6),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Pad: Occupied', alpha=0.8),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
                      markersize=10, label='Aircraft: High Alt', alpha=0.8),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='cyan',
                      markersize=10, label='Aircraft: Med Alt', alpha=0.8),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='purple',
                      markersize=10, label='Aircraft: Low Alt', alpha=0.8),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Set limits
        ax.set_xlim(-500, 700)
        ax.set_ylim(-500, 500)
    
    def _render_metrics_panel(self):
        """Render key metrics"""
        ax = self.ax_metrics
        ax.set_title('Real-Time Metrics', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Calculate current metrics
        avg_delay = self.env.total_delay / max(self.env.total_landings, 1)
        throughput = self.env.total_landings / max(self.env.current_time / 60.0, 0.1)
        pad_util = sum(1 for pad in self.env.pads if pad.occupied) / self.env.num_pads
        
        # Count aircraft by battery level
        critical_battery = sum(1 for a in self.env.aircraft if a.battery_soc < 20)
        low_battery = sum(1 for a in self.env.aircraft if 20 <= a.battery_soc < 40)
        
        metrics_text = f"""
Time: {self.env.current_time:.1f} min

Aircraft in Airspace: {len(self.env.aircraft)}
  • Critical Battery (<20%): {critical_battery}
  • Low Battery (20-40%): {low_battery}

Landing Statistics:
  • Total Landings: {self.env.total_landings}
  • Avg Delay: {avg_delay:.2f} min
  • Throughput: {throughput:.1f} aircraft/hr
  • Separation Violations: {self.env.separation_violations}

Pad Status:
  • Utilization: {pad_util:.1%}
  • Available: {sum(1 for p in self.env.pads if not p.occupied and p.cooldown_remaining == 0)}
  • Cooldown: {sum(1 for p in self.env.pads if p.cooldown_remaining > 0)}
  • Occupied: {sum(1 for p in self.env.pads if p.occupied)}
        """
        
        ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Update history
        self.time_history.append(self.env.current_time)
        self.delay_history.append(avg_delay)
        self.throughput_history.append(throughput)
        self.utilization_history.append(pad_util)
    
    def _render_delay_plot(self):
        """Render delay over time"""
        ax = self.ax_delay
        ax.set_title('Average Landing Delay', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Avg Delay (min)')
        ax.grid(True, alpha=0.3)
        
        if len(self.time_history) > 1:
            ax.plot(self.time_history, self.delay_history, 'b-', linewidth=2)
            ax.fill_between(self.time_history, 0, self.delay_history, alpha=0.3)
            
            # Target line
            ax.axhline(y=5, color='green', linestyle='--', label='Target (<5 min)', linewidth=2)
            ax.axhline(y=15, color='orange', linestyle='--', label='FCFS Baseline (~15 min)', linewidth=2)
            ax.legend(fontsize=8)
    
    def _render_utilization_plot(self):
        """Render pad utilization"""
        ax = self.ax_utilization
        ax.set_title('Pad Utilization Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Utilization (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        if len(self.time_history) > 1:
            ax.plot(self.time_history, self.utilization_history, 'g-', linewidth=2)
            ax.fill_between(self.time_history, 0, self.utilization_history, 
                          alpha=0.3, color='green')
            
            # Target lines
            ax.axhline(y=0.80, color='green', linestyle='--', 
                      label='Target (80%)', linewidth=2)
            ax.axhline(y=0.65, color='orange', linestyle='--',
                      label='FCFS Baseline (65%)', linewidth=2)
            ax.legend(fontsize=8)


def create_demo_visualization():
    """Create a demo visualization"""
    print("Creating demonstration visualization...")
    
    from baselines import GreedyScheduler
    
    env = VertiportEnv(num_pads=8, arrival_rate=25, render_mode=None)
    scheduler = GreedyScheduler(env)
    visualizer = VertiportVisualizer(env)
    
    obs, info = env.reset()
    scheduler.reset()
    
    print("Running simulation for 100 steps...")
    for step in range(100):
        actions = scheduler.select_actions(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Render every 10 steps
        if step % 10 == 0:
            visualizer.render_frame()
            print(f"  Step {step}: {len(env.aircraft)} aircraft, "
                  f"{env.total_landings} landings, "
                  f"{info['avg_delay']:.2f} min avg delay")
    
    # Save final frame
    visualizer.render_frame(save_path='demo_visualization.png')
    print("\nVisualization saved to demo_visualization.png")
    
    print("\n=== Final Statistics ===")
    print(f"Total landings: {env.total_landings}")
    print(f"Average delay: {env.total_delay / max(env.total_landings, 1):.2f} minutes")
    print(f"Throughput: {env.total_landings / (env.current_time / 60):.1f} aircraft/hour")
    print(f"Pad utilization: {info['pad_utilization']:.1%}")
    
    return visualizer


if __name__ == "__main__":
    visualizer = create_demo_visualization()
    plt.show()
