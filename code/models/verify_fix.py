import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from fix_leakage_solution import create_lagged_features_fixed

def test_leakage_fix():
    print("Verifying Leakage Fix...")
    
    # 1. Create a mock episode (TRAIN version - has target values)
    # 3 passes total. Last one is target.
    train_data = {
        'game_episode': [1, 1, 1],
        'game_id': [1, 1, 1],
        'action_id': [0, 1, 2],
        'team_id': [1, 1, 1],
        'start_x': [10, 20, 30],
        'start_y': [10, 20, 30],
        'end_x': [20, 30, 40], # Target end_x is 40
        'end_y': [20, 30, 40], # Target end_y is 40
        'dx': [10, 10, 10],    # Target dx is 10
        'dy': [10, 10, 10],    # Target dy is 10
        'zone_x': [1, 2, 3],
        'zone_y': [1, 2, 3],
        'goal_distance': [50, 40, 30],
        'goal_angle': [0, 0, 0],
        'dist_to_goal_line': [90, 80, 70],
        'dist_to_center_y': [24, 14, 4],
        'result_encoded': [0, 0, 0]
    }
    train_df = pd.DataFrame(train_data)
    
    # 2. Create mock episode (TEST version - target is NaN)
    test_df = train_df.copy()
    test_df.loc[2, 'end_x'] = np.nan
    test_df.loc[2, 'end_y'] = np.nan
    test_df.loc[2, 'dx'] = np.nan
    test_df.loc[2, 'dy'] = np.nan
    
    # 3. Generate Features
    print("Generating Train Features...")
    feats_train = create_lagged_features_fixed(train_df)
    
    print("Generating Test Features...")
    feats_test = create_lagged_features_fixed(test_df)
    
    # 4. Check for Equality of FEATURES (ignoring targets)
    feature_keys = [k for k in feats_train.keys() if 'target' not in k]
    
    mismatches = []
    for k in feature_keys:
        v_train = feats_train[k]
        v_test = feats_test[k]
        
        # Handle NaN equality
        if np.isnan(v_train) and np.isnan(v_test):
            continue
            
        if v_train != v_test:
            mismatches.append(f"{k}: Train={v_train}, Test={v_test}")
            
    if mismatches:
        print("FAIL: Mismatches found between Train and Test features!")
        for m in mismatches:
            print("  " + m)
    else:
        print("PASS: Train and Test features are identical!")
        
    # 5. Check if Target was excluded from stats
    # History dx are [10, 10]. Mean is 10.
    # If target (10) was included, mean is 10.
    # Let's change target dx to 100 in Train and see if avg_dx changes.
    
    train_df_leak = train_df.copy()
    train_df_leak.loc[2, 'dx'] = 100 # Huge outlier target
    
    feats_leak = create_lagged_features_fixed(train_df_leak)
    
    print(f"\nChecking Aggregation Logic:")
    print(f"  History dx: [10, 10]")
    print(f"  Target dx: 100")
    print(f"  Calculated avg_dx: {feats_leak['avg_dx']}")
    
    if feats_leak['avg_dx'] == 10:
        print("PASS: avg_dx matches history only (10). Target (100) was excluded.")
    else:
        print(f"FAIL: avg_dx is {feats_leak['avg_dx']}. It likely included the target.")

if __name__ == "__main__":
    test_leakage_fix()