"""
exp_088: LSTM + Delta Prediction + Domain Knowledge

이전 LSTM 실패 원인:
- 절대 좌표 (end_x, end_y) 예측 → CV-LB Gap 큼

새로운 접근:
- Delta prediction (dx, dy) 예측 → start_x/y + dx/dy = end_x/y
- 도메인 지식 기반 features (zone, goal_distance, goal_angle 등)
- GroupKFold by game_id
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"
SUBMISSION_DIR = BASE / "submissions"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==============================================================================
# LSTM Model
# ==============================================================================
class PassLSTMDelta(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # dx, dy
        )

    def forward(self, x, lengths):
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.lstm(packed)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden)
        out = self.fc(last_hidden)
        return out

# ==============================================================================
# Dataset
# ==============================================================================
class EpisodeDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

def collate_fn(batch):
    X, y, lengths = zip(*batch)
    X = torch.stack(X)
    y = torch.stack(y)
    lengths = torch.stack(lengths)
    return X, y, lengths

# ==============================================================================
# Feature Engineering
# ==============================================================================
def create_features(df):
    """Create features for each pass"""
    df = df.copy()

    # Basic delta
    df['dx'] = df['end_x'] - df['start_x']
    df['dy'] = df['end_y'] - df['start_y']
    df['dist'] = np.sqrt(df['dx']**2 + df['dy']**2)

    # Zone (6x6)
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)

    # Goal-related
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.arctan2(34 - df['start_y'], 105 - df['start_x'])
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)

    # Result encoding
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)

    return df

def process_episode(episode_df, max_len=50):
    """Process single episode into feature sequence"""
    episode_df = episode_df.sort_values('action_id').reset_index(drop=True)

    # CRITICAL: Separate history from target to prevent leakage!
    # History = all passes EXCEPT the last one (which we're predicting)
    # Target = last pass delta
    last_pass = episode_df.iloc[-1]
    history_df = episode_df.iloc[:-1]  # Exclude last pass from features!

    # Feature columns for sequence (NO dx, dy, dist - these leak target info!)
    feature_cols = [
        'start_x', 'start_y',
        'zone_x', 'zone_y', 'goal_distance', 'goal_angle',
        'dist_to_goal_line', 'dist_to_center_y', 'result_encoded'
    ]

    # If no history (single pass episode), use last pass start position only
    if len(history_df) == 0:
        features = np.zeros((1, len(feature_cols)), dtype=np.float32)
        features[0, 0] = last_pass['start_x'] / 105
        features[0, 1] = last_pass['start_y'] / 68
        features[0, 2] = last_pass['zone_x'] / 5
        features[0, 3] = last_pass['zone_y'] / 5
        features[0, 4] = last_pass['goal_distance'] / 120
        features[0, 5] = last_pass['goal_angle'] / np.pi
        features[0, 6] = last_pass['dist_to_goal_line'] / 105
        features[0, 7] = last_pass['dist_to_center_y'] / 34
        features[0, 8] = 0  # No result for pending pass
    else:
        # Use history features (previous passes, not the target pass)
        features = history_df[feature_cols].values.astype(np.float32)

        # Normalize (simple min-max scaling)
        features[:, 0] /= 105  # start_x
        features[:, 1] /= 68   # start_y
        features[:, 2] /= 5    # zone_x
        features[:, 3] /= 5    # zone_y
        features[:, 4] /= 120  # goal_distance
        features[:, 5] /= np.pi  # goal_angle
        features[:, 6] /= 105  # dist_to_goal_line
        features[:, 7] /= 34  # dist_to_center_y

    # Replace inf/nan
    features = np.nan_to_num(features, nan=0, posinf=1, neginf=-1)
    target = np.array([last_pass['dx'], last_pass['dy']], dtype=np.float32)
    start_xy = np.array([last_pass['start_x'], last_pass['start_y']], dtype=np.float32)
    end_xy = np.array([last_pass['end_x'], last_pass['end_y']], dtype=np.float32)

    # Pad sequence
    seq_len = min(len(features), max_len)
    padded = np.zeros((max_len, len(feature_cols)), dtype=np.float32)
    padded[:seq_len] = features[-seq_len:]  # Take last seq_len passes

    return padded, target, seq_len, start_xy, end_xy

def prepare_data(train_df, max_len=50):
    """Prepare all episodes"""
    print("Processing episodes...")

    X_list, y_list, length_list = [], [], []
    start_list, end_list, game_ids = [], [], []

    for game_ep, ep_df in train_df.groupby('game_episode'):
        X, y, length, start, end = process_episode(ep_df, max_len)
        X_list.append(X)
        y_list.append(y)
        length_list.append(length)
        start_list.append(start)
        end_list.append(end)
        game_ids.append(ep_df['game_id'].iloc[0])

    X = np.stack(X_list)
    y = np.stack(y_list)
    lengths = np.array(length_list)
    starts = np.stack(start_list)
    ends = np.stack(end_list)
    game_ids = np.array(game_ids)

    print(f"  Episodes: {len(X)}")
    print(f"  X shape: {X.shape}")
    print(f"  Unique games: {len(np.unique(game_ids))}")

    return X, y, lengths, starts, ends, game_ids

# ==============================================================================
# Training
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y, lengths in loader:
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()
        pred = model(X, lengths)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)

def eval_epoch(model, loader, starts, device):
    """Evaluate with Euclidean distance on absolute coordinates"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, y, lengths in loader:
            X, lengths = X.to(device), lengths.to(device)
            pred = model(X, lengths)
            all_preds.append(pred.cpu().numpy())

    pred_delta = np.vstack(all_preds)
    pred_abs = starts + pred_delta

    return pred_abs

def run_cv(X, y, lengths, starts, ends, game_ids, n_splits=5, epochs=50, patience=10):
    """Run cross-validation"""
    input_size = X.shape[2]
    gkf = GroupKFold(n_splits=n_splits)

    oof_pred = np.zeros((len(X), 2))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, game_ids), 1):
        print(f"\n  Fold {fold}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        len_train, len_val = lengths[train_idx], lengths[val_idx]
        start_val = starts[val_idx]
        end_val = ends[val_idx]

        train_ds = EpisodeDataset(X_train, y_train, len_train)
        val_ds = EpisodeDataset(X_val, y_val, len_val)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = PassLSTMDelta(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.5)
        model = model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_score = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # Evaluate
            pred_abs = eval_epoch(model, val_loader, start_val, device)
            dist = np.sqrt((pred_abs[:, 0] - end_val[:, 0])**2 + (pred_abs[:, 1] - end_val[:, 1])**2)
            val_score = dist.mean()

            scheduler.step(val_score)

            if val_score < best_val_score:
                best_val_score = val_score
                best_pred = pred_abs.copy()
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or no_improve == 0:
                print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_dist={val_score:.4f}")

            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        oof_pred[val_idx] = best_pred
        fold_scores.append(best_val_score)
        print(f"    Fold {fold} best: {best_val_score:.4f}")

        del model, optimizer, train_loader, val_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    cv_score = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    return cv_score, cv_std, fold_scores, oof_pred

# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("exp_088: LSTM + Delta Prediction")
    print("=" * 70)

    print("\n[1] Loading data...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    print(f"  Raw: {len(train_raw)} rows")

    print("\n[2] Creating features...")
    train_df = create_features(train_raw)

    print("\n[3] Preparing sequences...")
    X, y, lengths, starts, ends, game_ids = prepare_data(train_df, max_len=30)

    del train_raw, train_df
    gc.collect()

    print("\n[4] Running CV (5-fold)...")
    cv, cv_std, fold_scores, oof_pred = run_cv(
        X, y, lengths, starts, ends, game_ids,
        n_splits=5, epochs=100, patience=15
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"  CV: {cv:.4f} +/- {cv_std:.4f}")

    baseline_cv = 13.5435  # exp_083 best
    diff = cv - baseline_cv
    print(f"\n  Baseline (exp_083): {baseline_cv:.4f}")
    print(f"  Difference: {diff:+.4f}")

    if diff < 0:
        print("  *** IMPROVEMENT! ***")

    print("=" * 70)

if __name__ == "__main__":
    main()
