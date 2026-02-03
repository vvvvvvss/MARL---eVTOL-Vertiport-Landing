import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from simple_evtol_env import SimpleEVTOLEnv
from stable_baselines3 import PPO

def analyze_performance(model_path, num_trials=20):
    """
    Comprehensive performance analysis
    """
    env = SimpleEVTOLEnv()
    model = PPO.load(model_path)
    
    metrics = {
        'success': [],
        'steps': [],
        'battery_used': [],
        'final_distance': [],
        'avg_speed': [],
        'collisions': []
    }
    
    print(f"Running {num_trials} test episodes...")
    
    for trial in range(num_trials):
        obs, info = env.reset()
        
        episode_data = {
            'positions': [env.position.copy()],
            'velocities': [],
            'collision': False
        }
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data['positions'].append(env.position.copy())
            episode_data['velocities'].append(np.linalg.norm(env.velocity))
            
            # Check collision
            for obstacle in env.obstacles:
                dist = np.linalg.norm(env.position - obstacle['pos'])
                if dist < obstacle['radius']:
                    episode_data['collision'] = True
            
            if done or truncated:
                break
        
        # Calculate metrics
        final_dist = np.linalg.norm(env.goal - env.position)
        success = final_dist < 5.0
        battery_used = 100 - env.battery
        avg_speed = np.mean(episode_data['velocities']) if episode_data['velocities'] else 0
        
        metrics['success'].append(1 if success else 0)
        metrics['steps'].append(step + 1)
        metrics['battery_used'].append(battery_used)
        metrics['final_distance'].append(final_dist)
        metrics['avg_speed'].append(avg_speed)
        metrics['collisions'].append(1 if episode_data['collision'] else 0)
        
        if (trial + 1) % 5 == 0:
            print(f"  Completed {trial + 1}/{num_trials} trials")
    
    return metrics


def create_performance_dashboard(metrics):
    """
    Create interactive dashboard with all performance metrics
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Success Rate',
            'Episode Length Distribution',
            'Battery Usage',
            'Final Distance to Goal',
            'Average Speed',
            'Collision Rate'
        ),
        specs=[
            [{'type': 'indicator'}, {'type': 'histogram'}],
            [{'type': 'box'}, {'type': 'box'}],
            [{'type': 'box'}, {'type': 'indicator'}]
        ]
    )
    
    # 1. Success Rate (Gauge)
    success_rate = np.mean(metrics['success']) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=success_rate,
            title={'text': "Success Rate (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # 2. Episode Length Distribution
    fig.add_trace(
        go.Histogram(
            x=metrics['steps'],
            nbinsx=20,
            marker_color='rgb(55, 83, 109)',
            name='Steps'
        ),
        row=1, col=2
    )
    
    # 3. Battery Usage Box Plot
    fig.add_trace(
        go.Box(
            y=metrics['battery_used'],
            name='Battery Used (%)',
            marker_color='rgb(255, 127, 14)',
            boxmean='sd'
        ),
        row=2, col=1
    )
    
    # 4. Final Distance Box Plot
    fig.add_trace(
        go.Box(
            y=metrics['final_distance'],
            name='Final Distance (m)',
            marker_color='rgb(44, 160, 44)',
            boxmean='sd'
        ),
        row=2, col=2
    )
    
    # 5. Average Speed Box Plot
    fig.add_trace(
        go.Box(
            y=metrics['avg_speed'],
            name='Avg Speed (m/s)',
            marker_color='rgb(214, 39, 40)',
            boxmean='sd'
        ),
        row=3, col=1
    )
    
    # 6. Collision Rate (Gauge)
    collision_rate = np.mean(metrics['collisions']) * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=collision_rate,
            title={'text': "Collision Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 30], 'color': "yellow"},
                    {'range': [30, 100], 'color': "lightcoral"}
                ]
            }
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=False,
        title_text="eVTOL Agent Performance Dashboard",
        title_font_size=20
    )
    
    # Add summary statistics annotation
    summary_text = f"""
    <b>Performance Summary ({len(metrics['success'])} trials):</b><br>
    ✅ Success Rate: {success_rate:.1f}%<br>
    📊 Avg Steps: {np.mean(metrics['steps']):.1f} ± {np.std(metrics['steps']):.1f}<br>
    🔋 Avg Battery Used: {np.mean(metrics['battery_used']):.1f}% ± {np.std(metrics['battery_used']):.1f}%<br>
    🎯 Avg Final Distance: {np.mean(metrics['final_distance']):.2f}m<br>
    💨 Avg Speed: {np.mean(metrics['avg_speed']):.2f}m/s<br>
    💥 Collision Rate: {collision_rate:.1f}%
    """
    
    fig.add_annotation(
        x=0.5, y=-0.05,
        xref='paper', yref='paper',
        text=summary_text,
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=2,
        align='left',
        font=dict(size=12)
    )
    
    return fig


def create_comparison_table(trained_metrics, random_metrics):
    """
    Create comparison table between trained and random agent
    """
    trained_avg = {
        'Success Rate': f"{np.mean(trained_metrics['success'])*100:.1f}%",
        'Avg Steps': f"{np.mean(trained_metrics['steps']):.1f}",
        'Avg Battery Used': f"{np.mean(trained_metrics['battery_used']):.1f}%",
        'Avg Final Distance': f"{np.mean(trained_metrics['final_distance']):.2f}m",
        'Collision Rate': f"{np.mean(trained_metrics['collisions'])*100:.1f}%"
    }
    
    random_avg = {
        'Success Rate': f"{np.mean(random_metrics['success'])*100:.1f}%",
        'Avg Steps': f"{np.mean(random_metrics['steps']):.1f}",
        'Avg Battery Used': f"{np.mean(random_metrics['battery_used']):.1f}%",
        'Avg Final Distance': f"{np.mean(random_metrics['final_distance']):.2f}m",
        'Collision Rate': f"{np.mean(random_metrics['collisions'])*100:.1f}%"
    }
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Trained Agent</b>', '<b>Random Agent</b>', '<b>Improvement</b>'],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                list(trained_avg.keys()),
                list(trained_avg.values()),
                list(random_avg.values()),
                [
                    f"+{(np.mean(trained_metrics['success']) - np.mean(random_metrics['success']))*100:.1f}%",
                    f"{np.mean(trained_metrics['steps']) - np.mean(random_metrics['steps']):.1f}",
                    f"{np.mean(trained_metrics['battery_used']) - np.mean(random_metrics['battery_used']):.1f}%",
                    f"{np.mean(random_metrics['final_distance']) - np.mean(trained_metrics['final_distance']):.2f}m better",
                    f"{(np.mean(random_metrics['collisions']) - np.mean(trained_metrics['collisions']))*100:.1f}% fewer"
                ]
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title='Performance Comparison: Trained vs Random Agent',
        height=400
    )
    
    return fig


if __name__ == "__main__":
    print("📊 Analyzing trained agent performance...\n")
    
    # Analyze trained agent
    trained_metrics = analyze_performance("evtol_trained", num_trials=20)
    
    print("\n📊 Creating performance dashboard...")
    dashboard = create_performance_dashboard(trained_metrics)
    dashboard.write_html("performance_dashboard.html")
    dashboard.show()
    
    print("\n✅ Dashboard saved to 'performance_dashboard.html'")
    
    # Optional: Compare with random agent
    print("\n🎲 Analyzing random agent for comparison...")
    env = SimpleEVTOLEnv()
    random_metrics = {
        'success': [],
        'steps': [],
        'battery_used': [],
        'final_distance': [],
        'avg_speed': [],
        'collisions': []
    }
    
    for trial in range(20):
        obs, info = env.reset()
        velocities = []
        collision = False
        
        for step in range(500):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            velocities.append(np.linalg.norm(env.velocity))
            
            for obstacle in env.obstacles:
                dist = np.linalg.norm(env.position - obstacle['pos'])
                if dist < obstacle['radius']:
                    collision = True
            
            if done or truncated:
                break
        
        final_dist = np.linalg.norm(env.goal - env.position)
        random_metrics['success'].append(1 if final_dist < 5.0 else 0)
        random_metrics['steps'].append(step + 1)
        random_metrics['battery_used'].append(100 - env.battery)
        random_metrics['final_distance'].append(final_dist)
        random_metrics['avg_speed'].append(np.mean(velocities) if velocities else 0)
        random_metrics['collisions'].append(1 if collision else 0)
    
    print("\n📋 Creating comparison table...")
    comparison = create_comparison_table(trained_metrics, random_metrics)
    comparison.write_html("comparison_table.html")
    comparison.show()
    
    print("\n✅ All visualizations created successfully!")
    print("   Files generated:")
    print("   - performance_dashboard.html")
    print("   - comparison_table.html")