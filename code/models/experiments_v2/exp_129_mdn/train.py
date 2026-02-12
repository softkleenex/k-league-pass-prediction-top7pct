"""
exp_129: Mixture Density Network (MDN)
- Instead of predicting single (dx, dy), predict Gaussian mixture
- Handles multimodal pass destinations (player might pass left OR right)
- Uses Negative Log-Likelihood loss
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
DATA_DIR = BASE / "data"

def create_features(df):
    df['zone_x'] = (df['start_x'] / (105/6)).astype(int).clip(0, 5)
    df['zone_y'] = (df['start_y'] / (68/6)).astype(int).clip(0, 5)
    df['goal_distance'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2)
    df['goal_angle'] = np.degrees(np.arctan2(34 - df['start_y'], 105 - df['start_x']))
    df['dx'] = df['end_x'] - df['start_x'] if 'end_x' in df.columns else 0
    df['dy'] = df['end_y'] - df['start_y'] if 'end_y' in df.columns else 0
    df['prev_dx'] = df.groupby('game_episode')['dx'].shift(1).fillna(0) if 'dx' in df.columns else 0
    df['prev_dy'] = df.groupby('game_episode')['dy'].shift(1).fillna(0) if 'dy' in df.columns else 0
    result_map = {'Successful': 0, 'Unsuccessful': 1}
    df['result_encoded'] = df['result_name'].map(result_map).fillna(2).astype(int)
    ema_span = 2
    df['ema_start_x'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_x'])
    df['ema_start_y'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(df['start_y'])
    result_map2 = {'Successful': 1, 'Unsuccessful': 0}
    df['is_successful'] = df['result_name'].map(result_map2).fillna(0)
    df['ema_success_rate'] = df.groupby('game_episode')['is_successful'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['final_team_id'] = df.groupby('game_episode')['team_id'].transform('last')
    df['is_final_team'] = (df['team_id'] == df['final_team_id']).astype(int)
    df['ema_possession'] = df.groupby('game_episode')['is_final_team'].transform(
        lambda x: x.ewm(span=ema_span, adjust=False).mean().shift(1)).fillna(0.5)
    df['dist_to_goal_line'] = 105 - df['start_x']
    df['dist_to_center_y'] = np.abs(df['start_y'] - 34)
    df['diff_x'] = df.groupby('game_episode')['start_x'].diff().fillna(0)
    df['prev_start_x'] = df.groupby('game_episode')['start_x'].shift(1).fillna(df['start_x'])
    df['prev_start_y'] = df.groupby('game_episode')['start_y'].shift(1).fillna(df['start_y'])
    df['velocity'] = np.sqrt((df['start_x'] - df['prev_start_x'])**2 + (df['start_y'] - df['prev_start_y'])**2)
    df['ema_momentum_y'] = df['ema_start_y'] - df['start_y']
    return df

def load_test_data():
    test_index = pd.read_csv(DATA_DIR / 'test.csv')
    dfs = []
    for _, row in test_index.iterrows():
        path = DATA_DIR / row['path'].replace('./', '')
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

FEATURES = ['goal_angle', 'zone_y', 'goal_distance', 'dist_to_goal_line',
            'dist_to_center_y', 'prev_dx', 'prev_dy', 'ema_start_x',
            'ema_start_y', 'ema_success_rate', 'ema_possession',
            'zone_x', 'result_encoded', 'diff_x', 'velocity', 'ema_momentum_y']


class MDN(nn.Module):
    """Mixture Density Network for 2D output"""
    def __init__(self, input_dim, hidden_dim=128, n_components=3):
        super(MDN, self).__init__()
        self.n_components = n_components

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Output: pi (mixing coeffs), mu_x, mu_y, sigma_x, sigma_y for each component
        # Total: n_components * 5
        self.pi = nn.Linear(hidden_dim, n_components)  # mixing coefficients
        self.mu_x = nn.Linear(hidden_dim, n_components)
        self.mu_y = nn.Linear(hidden_dim, n_components)
        self.sigma_x = nn.Linear(hidden_dim, n_components)
        self.sigma_y = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.shared(x)
        pi = torch.softmax(self.pi(h), dim=-1)
        mu_x = self.mu_x(h)
        mu_y = self.mu_y(h)
        sigma_x = torch.exp(self.sigma_x(h)) + 1e-4  # Ensure positive
        sigma_y = torch.exp(self.sigma_y(h)) + 1e-4
        return pi, mu_x, mu_y, sigma_x, sigma_y


def mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, target_x, target_y):
    """Negative Log-Likelihood loss for MDN"""
    # Gaussian PDF for each component
    target_x = target_x.unsqueeze(-1)  # [batch, 1]
    target_y = target_y.unsqueeze(-1)

    # Compute log probability for each component
    log_prob_x = -0.5 * ((target_x - mu_x) / sigma_x)**2 - torch.log(sigma_x) - 0.5 * np.log(2 * np.pi)
    log_prob_y = -0.5 * ((target_y - mu_y) / sigma_y)**2 - torch.log(sigma_y) - 0.5 * np.log(2 * np.pi)
    log_prob = log_prob_x + log_prob_y  # Independent x, y

    # Weighted sum (logsumexp for numerical stability)
    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_pi + log_prob
    nll = -torch.logsumexp(log_prob_weighted, dim=-1)

    return nll.mean()


def predict_mdn(model, X, device, use_mode=True):
    """Get prediction from MDN"""
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        pi, mu_x, mu_y, sigma_x, sigma_y = model(X)
        if use_mode:
            # Use the mean of the highest probability component
            best_idx = pi.argmax(dim=-1)
            pred_x = mu_x.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
            pred_y = mu_y.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
        else:
            # Weighted average
            pred_x = (pi * mu_x).sum(dim=-1)
            pred_y = (pi * mu_y).sum(dim=-1)
    return pred_x.cpu().numpy(), pred_y.cpu().numpy()


def main():
    print("="*60)
    print("exp_129: Mixture Density Network (MDN)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    train_df = create_features(train_df)
    train_last = train_df.groupby('game_episode').last().reset_index()

    print(f"Train episodes: {len(train_last)}")

    X = train_last[FEATURES].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
    y_dx = train_last['dx'].values.astype(np.float32)
    y_dy = train_last['dy'].values.astype(np.float32)
    groups = train_last['game_id'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gkf = GroupKFold(n_splits=11)
    folds = list(gkf.split(X, y_dx, groups))

    # Test different configurations
    configs = [
        ('3 components', {'n_components': 3, 'hidden_dim': 128}),
        ('5 components', {'n_components': 5, 'hidden_dim': 128}),
        ('3 comp, 256 hidden', {'n_components': 3, 'hidden_dim': 256}),
    ]

    results = {}
    for name, cfg in configs:
        print(f"\n{name}...")
        scores = []

        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                X_train_fold = torch.FloatTensor(X_scaled[train_idx]).to(device)
                X_val_fold = torch.FloatTensor(X_scaled[val_idx]).to(device)
                y_dx_train = torch.FloatTensor(y_dx[train_idx]).to(device)
                y_dy_train = torch.FloatTensor(y_dy[train_idx]).to(device)
                y_dx_val = y_dx[val_idx]
                y_dy_val = y_dy[val_idx]

                model = MDN(input_dim=len(FEATURES), **cfg).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # Training
                dataset = TensorDataset(X_train_fold, y_dx_train, y_dy_train)
                loader = DataLoader(dataset, batch_size=256, shuffle=True)

                best_val_loss = float('inf')
                patience = 20
                patience_counter = 0

                for epoch in range(200):
                    model.train()
                    for batch_X, batch_dx, batch_dy in loader:
                        optimizer.zero_grad()
                        pi, mu_x, mu_y, sigma_x, sigma_y = model(batch_X)
                        loss = mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, batch_dx, batch_dy)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        pi, mu_x, mu_y, sigma_x, sigma_y = model(X_val_fold)
                        val_loss = mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y,
                                           torch.FloatTensor(y_dx_val).to(device),
                                           torch.FloatTensor(y_dy_val).to(device))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            break

                # Load best model and evaluate
                model.load_state_dict(best_state)
                pred_dx, pred_dy = predict_mdn(model, torch.FloatTensor(X_scaled[val_idx]), device)
                dist = np.sqrt((pred_dx - y_dx_val)**2 + (pred_dy - y_dy_val)**2)
                fold_scores.append(dist.mean())

            scores.append(np.mean(fold_scores))
            print(f"  Seed {seed}: {np.mean(fold_scores):.4f}")

        cv = np.mean(scores)
        std = np.std(scores)
        results[name] = (cv, std)
        print(f"  {name}: CV {cv:.4f} (+/- {std:.4f})")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    best = min(results, key=lambda k: results[k][0])
    for k in sorted(results.keys(), key=lambda k: results[k][0]):
        cv, std = results[k]
        m = " <-- BEST" if k == best else ""
        print(f"  {k}: CV {cv:.4f} (+/- {std:.4f}){m}")


if __name__ == "__main__":
    main()
