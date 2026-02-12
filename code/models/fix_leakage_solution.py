import pandas as pd
import numpy as np

def create_lagged_features_fixed(episode_df):
    """
    Create lagged features for an episode WITHOUT target leakage.
    
    KEY FIX: 
    1. We separate the 'target' pass (last row) from 'history' (previous passes).
    2. All statistics (mean, std) are calculated ONLY on 'history'.
    3. Lagged features are pulled from 'history'.
    """
    # Ensure sorted by action_id just in case
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)
    n = len(episode_df)
    
    # 1. Identify Target (Last Pass)
    # The last row is what we want to predict. 
    # In Train: contains valid end_x, end_y, dx, dy (TARGETS)
    # In Test: contains NaN end_x, end_y (to be predicted)
    last = episode_df.iloc[-1]
    
    # 2. Identify History (Past Passes)
    # These are all passes BEFORE the target pass.
    # We must use ONLY these for feature engineering to avoid leakage.
    history = episode_df.iloc[:-1] # Exclude the last row
    n_history = len(history)
    
    features = {
        # Metadata
        'game_episode': last['game_episode'],
        'game_id': last['game_id'],
        
        # Features available at inference time for the target pass
        # (We know where the pass starts, just not where it ends)
        'start_x': last['start_x'],
        'start_y': last['start_y'],
        'zone_x': last['zone_x'],
        'zone_y': last['zone_y'],
        'goal_distance': last['goal_distance'],
        'goal_angle': last['goal_angle'],
        'dist_to_goal_line': last['dist_to_goal_line'],
        'dist_to_center_y': last['dist_to_center_y'],
        # If result is known before pass completes (e.g. failure?), include it. 
        # Usually result is known AFTER pass, so be careful. 
        # Assuming result_encoded is NOT known for the target pass in test.
        # If it IS known (e.g. "we know this pass succeeded, where did it go?"), keep it.
        # Based on competition: usually we predict outcome OR location. 
        # If strictly predicting location, maybe we don't know result yet.
        # SAFEST: Don't use target's result unless sure.
        # 'result_encoded': last['result_encoded'], 
    }
    
    # Add Targets (only for training) - OK to keep here as they are labels, not features
    features['target_dx'] = last['dx']
    features['target_dy'] = last['dy']

    # 3. Lagged Features (from History)
    # lag=1 means the immediate previous pass (last row of history)
    for lag in [1, 2, 3]:
        # History index: -1 is lag 1, -2 is lag 2...
        hist_idx = -lag 
        
        if n_history >= lag:
            prev = history.iloc[hist_idx]
            features[f'prev{lag}_dx'] = prev['dx']
            features[f'prev{lag}_dy'] = prev['dy']
            features[f'prev{lag}_start_x'] = prev['start_x']
            features[f'prev{lag}_start_y'] = prev['start_y']
            features[f'prev{lag}_end_x'] = prev['end_x']
            features[f'prev{lag}_end_y'] = prev['end_y']
            features[f'prev{lag}_result'] = prev['result_encoded']
            features[f'prev{lag}_goal_dist'] = prev['goal_distance']
            
            # Team change indicator (compared to current kicker's team)
            features[f'prev{lag}_same_team'] = 1 if prev['team_id'] == last['team_id'] else 0
        else:
            # Pad with zeros or appropriate defaults
            features[f'prev{lag}_dx'] = 0
            features[f'prev{lag}_dy'] = 0
            features[f'prev{lag}_start_x'] = last['start_x'] # Fallback to current start
            features[f'prev{lag}_start_y'] = last['start_y']
            features[f'prev{lag}_end_x'] = last['start_x']
            features[f'prev{lag}_end_y'] = last['start_y']
            features[f'prev{lag}_result'] = 2 # Unknown/Other
            features[f'prev{lag}_goal_dist'] = last['goal_distance']
            features[f'prev{lag}_same_team'] = 1 # Assume same team if no history
            
    # 4. Episode Statistics (Calculated on HISTORY only)
    if n_history > 0:
        features['avg_dx'] = history['dx'].mean()
        features['avg_dy'] = history['dy'].mean()
        features['std_dx'] = history['dx'].std()
        features['std_dy'] = history['dy'].std()
        features['total_distance'] = np.sqrt(history['dx']**2 + history['dy']**2).sum()
        features['avg_goal_dist'] = history['goal_distance'].mean()
        
        # Direction changes within history
        angles = np.arctan2(history['dy'].values, history['dx'].values)
        if len(angles) > 1:
            angle_diffs = np.abs(np.diff(angles))
            features['avg_angle_change'] = np.mean(angle_diffs)
        else:
            features['avg_angle_change'] = 0
            
        features['episode_success_rate'] = (history['result_encoded'] == 0).mean()
        
        # Team possession in history
        last_team = last['team_id']
        features['team_possession_ratio'] = (history['team_id'] == last_team).mean()
        
    else:
        # First pass of the episode (no history)
        features['avg_dx'] = 0
        features['avg_dy'] = 0
        features['std_dx'] = 0
        features['std_dy'] = 0
        features['total_distance'] = 0
        features['avg_goal_dist'] = last['goal_distance']
        features['avg_angle_change'] = 0
        features['episode_success_rate'] = 0.5 # Default
        features['team_possession_ratio'] = 1.0 # Only current team exists
        
    return features
