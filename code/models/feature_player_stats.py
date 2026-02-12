import pandas as pd
import numpy as np

def get_player_stats(df):
    """
    Calculates aggregated statistics for each player from the provided DataFrame.
    Returns a DataFrame indexed by player_id.
    """
    # Create copies of columns needed to avoid modifying original
    temp_df = df.copy()
    
    # Calculate derived metrics if not present
    if 'dx' not in temp_df.columns:
        temp_df['dx'] = temp_df['end_x'] - temp_df['start_x']
    if 'dy' not in temp_df.columns:
        temp_df['dy'] = temp_df['end_y'] - temp_df['start_y']
        
    temp_df['dist'] = np.sqrt(temp_df['dx']**2 + temp_df['dy']**2)
    # Check strict equality for success, handle potential case variations
    temp_df['is_success'] = temp_df['result_name'].apply(lambda x: 1 if x == 'Successful' else 0)
    
    # Aggregations
    stats = temp_df.groupby('player_id').agg({
        'dx': ['sum', 'count', 'mean', 'std'],
        'dy': ['sum', 'mean', 'std'],
        'dist': ['mean'],
        'is_success': ['mean', 'sum'] # mean is rate, sum is total successes
    })
    
    # Flatten MultiIndex columns
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    # Rename for clarity
    stats = stats.rename(columns={
        'dx_mean': 'player_avg_dx',
        'dx_std': 'player_std_dx',
        'dy_mean': 'player_avg_dy',
        'dy_std': 'player_std_dy',
        'dist_mean': 'player_avg_dist',
        'is_success_mean': 'player_success_rate',
        'dx_count': 'player_pass_count'
    })
    
    # Calculate preferred angle (mean of angles)
    # Using arctan2 of the mean vector is better than mean of angles
    stats['player_preferred_angle'] = np.arctan2(stats['player_avg_dy'], stats['player_avg_dx'])
    
    return stats

def add_player_features_train(train_df):
    """
    Adds player features to the TRAINING set using Leave-One-Out (LOO) encoding
    to prevent target leakage.
    """
    print("Generating player features for Training set (LOO method)...")
    df = train_df.copy()
    
    # Calculate global stats (sum and count) for LOO
    if 'dx' not in df.columns:
        df['dx'] = df['end_x'] - df['start_x']
    if 'dy' not in df.columns:
        df['dy'] = df['end_y'] - df['start_y']
        
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['is_success'] = (df['result_name'] == 'Successful').astype(int)
    
    # 1. Global Aggregation
    # We need sum and count to perform LOO: (Sum - Current) / (Count - 1)
    grp = df.groupby('player_id')[['dx', 'dy', 'dist', 'is_success']]
    sums = grp.transform('sum')
    counts = grp.transform('count')
    
    # 2. LOO Calculation
    # For Mean: (TotalSum - CurrentVal) / (TotalCount - 1)
    
    # Avoid division by zero for single-observation players
    # We will fill these with the global average later
    counts_safe = counts.replace(1, np.nan) 
    
    df['player_avg_dx'] = (sums['dx'] - df['dx']) / (counts_safe['dx'] - 1)
    df['player_avg_dy'] = (sums['dy'] - df['dy']) / (counts_safe['dy'] - 1)
    df['player_avg_dist'] = (sums['dist'] - df['dist']) / (counts_safe['dist'] - 1)
    df['player_success_rate'] = (sums['is_success'] - df['is_success']) / (counts_safe['is_success'] - 1)
    
    df['player_pass_count'] = counts['dx'] # Total count is fine (or count-1)
    
    # 3. Handle Missing/Single-Observation Players
    # Fill NaNs with Global Population Means
    global_means = {
        'player_avg_dx': df['dx'].mean(),
        'player_avg_dy': df['dy'].mean(),
        'player_avg_dist': df['dist'].mean(),
        'player_success_rate': df['is_success'].mean()
    }
    
    df.fillna(value=global_means, inplace=True)
    
    # Preferred Angle (re-calculate from LOO dx/dy)
    df['player_preferred_angle'] = np.arctan2(df['player_avg_dy'], df['player_avg_dx'])
    
    return df

def add_player_features_test(test_df, train_df):
    """
    Adds player features to the TEST set using statistics from the TRAIN set.
    """
    print("Generating player features for Test set (Mapping from Train)...")
    test_df = test_df.copy()
    
    # 1. Get stats from Train
    player_stats = get_player_stats(train_df)
    
    # Columns to keep
    cols_to_use = [
        'player_avg_dx', 'player_avg_dy', 'player_avg_dist', 
        'player_success_rate', 'player_pass_count', 'player_preferred_angle'
    ]
    
    # 2. Merge
    test_df = test_df.merge(player_stats[cols_to_use], on='player_id', how='left')
    
    # 3. Fill Missing Players (New in Test)
    # Calculate global means from TRAIN
    if 'dx' not in train_df.columns:
        train_df['dx'] = train_df['end_x'] - train_df['start_x']
    if 'dy' not in train_df.columns:
        train_df['dy'] = train_df['end_y'] - train_df['start_y']
        
    global_dist = np.sqrt(train_df['dx']**2 + train_df['dy']**2).mean()
    global_success = (train_df['result_name'] == 'Successful').mean()
    
    fill_values = {
        'player_avg_dx': train_df['dx'].mean(),
        'player_avg_dy': train_df['dy'].mean(),
        'player_avg_dist': global_dist,
        'player_success_rate': global_success,
        'player_pass_count': 0,
        'player_preferred_angle': 0 # Neutral
    }
    
    test_df.fillna(value=fill_values, inplace=True)
    
    return test_df
