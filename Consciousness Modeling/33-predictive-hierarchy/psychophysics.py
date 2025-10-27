"""
Psychophysical experiments: change blindness, binocular rivalry, attentional blink.
"""
import numpy as np


def change_blindness_experiment(hierarchy, workspace, attention, n_trials=10):
    """Simulate change blindness: changes go unnoticed without attention.
    
    Returns: detection rates with/without attention
    """
    results = {'attended': [], 'unattended': []}
    
    for trial in range(n_trials):
        # Create stimulus with change
        stimulus_before = np.random.randn(hierarchy.levels[0].output_dim)
        stimulus_after = stimulus_before.copy()
        
        # Change a subset of features
        change_idx = np.random.choice(len(stimulus_after), size=3, replace=False)
        stimulus_after[change_idx] += 2.0
        
        # Attended condition
        attention.focus_attention(level_idx=0, feature_indices=change_idx, strength=3.0)
        hierarchy.forward(stimulus_before, n_iterations=5)
        workspace.update()
        before_state = workspace.workspace.copy()
        
        hierarchy.forward(stimulus_after, n_iterations=5)
        workspace.update()
        after_state = workspace.workspace.copy()
        
        # Detect change (workspace difference)
        change_detected = np.linalg.norm(after_state - before_state) > 0.5
        results['attended'].append(change_detected)
        
        # Unattended condition
        attention.diffuse_attention()
        hierarchy.forward(stimulus_before, n_iterations=5)
        workspace.update()
        before_state = workspace.workspace.copy()
        
        hierarchy.forward(stimulus_after, n_iterations=5)
        workspace.update()
        after_state = workspace.workspace.copy()
        
        change_detected = np.linalg.norm(after_state - before_state) > 0.5
        results['unattended'].append(change_detected)
    
    detection_attended = np.mean(results['attended'])
    detection_unattended = np.mean(results['unattended'])
    
    return {
        'detection_attended': float(detection_attended),
        'detection_unattended': float(detection_unattended),
        'attention_effect': float(detection_attended - detection_unattended),
    }


def binocular_rivalry_experiment(hierarchy, workspace, n_steps=100):
    """Simulate binocular rivalry: alternating percepts.
    
    Returns: time series of dominant percept
    """
    # Two competing stimuli
    stimulus_A = np.ones(hierarchy.levels[0].output_dim) * 1.0
    stimulus_B = np.ones(hierarchy.levels[0].output_dim) * -1.0
    
    # Mix with noise
    dominance_history = []
    
    for step in range(n_steps):
        # Combine stimuli with noise
        noise = np.random.randn(hierarchy.levels[0].output_dim) * 0.1
        mixed = 0.5 * stimulus_A + 0.5 * stimulus_B + noise
        
        # Process
        outputs = hierarchy.forward(mixed, n_iterations=5)
        workspace.update()
        
        # Determine dominant percept (which stimulus is better predicted)
        pred = outputs['predictions'][0]
        error_A = np.sum((pred - stimulus_A) ** 2)
        error_B = np.sum((pred - stimulus_B) ** 2)
        
        dominant = 'A' if error_A < error_B else 'B'
        dominance_history.append(dominant)
    
    # Count switches
    switches = sum(1 for i in range(1, len(dominance_history)) if dominance_history[i] != dominance_history[i-1])
    
    return {
        'dominance_history': dominance_history,
        'n_switches': int(switches),
        'switch_rate': float(switches / n_steps),
    }


def attentional_blink_experiment(hierarchy, workspace, attention, n_trials=20):
    """Simulate attentional blink: second target missed if too close to first.
    
    Returns: detection rates by lag
    """
    results = {lag: [] for lag in [1, 2, 3, 5, 8]}
    
    for trial in range(n_trials):
        for lag in results.keys():
            # Stimulus stream
            stream_length = 15
            stream = [np.random.randn(hierarchy.levels[0].output_dim) * 0.5 for _ in range(stream_length)]
            
            # Insert targets
            T1_pos = 5
            T2_pos = T1_pos + lag
            
            if T2_pos >= stream_length:
                continue
            
            # Targets are stronger signals
            stream[T1_pos] = np.ones(hierarchy.levels[0].output_dim) * 2.0
            stream[T2_pos] = np.ones(hierarchy.levels[0].output_dim) * -2.0
            
            # Process stream
            T2_detected = False
            
            for t, stimulus in enumerate(stream):
                outputs = hierarchy.forward(stimulus, n_iterations=3)
                workspace.update()
                
                # Check if T2 is in workspace (conscious)
                if t == T2_pos:
                    broadcast_strength = workspace.get_broadcast_strength()
                    T2_detected = broadcast_strength > 1.0
            
            results[lag].append(T2_detected)
    
    # Compute detection rates
    detection_by_lag = {lag: float(np.mean(detections)) for lag, detections in results.items() if len(detections) > 0}
    
    return {
        'detection_by_lag': detection_by_lag,
        'blink_present': detection_by_lag.get(2, 0) < detection_by_lag.get(8, 1),
    }
