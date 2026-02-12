"""
exp_050: Transformer 시퀀스 모델 + Goal-oriented 피처
목표: 1위 (12점대) 달성

핵심 전략:
1. 전체 시퀀스를 Transformer로 학습
2. Goal-oriented 피처 (골대 거리/각도)
3. 시퀀스 패턴 학습
"""
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE = Path("/mnt/c/LSJ/dacon/dacon/kleague-algorithm")
TRAIN_CSV = BASE / "data/train.csv"

# 축구장 크기 (K리그 기준)
FIELD_LENGTH = 105  # x: 0-105
FIELD_WIDTH = 68    # y: 0-68
GOAL_X = 105        # 공격 골대 x좌표

class SoccerDataset(Dataset):
    """시퀀스 데이터셋"""
    def __init__(self, sequences, targets=None, max_len=50):
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # 패딩/트런케이션
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]  # 마지막 부분 유지

        # 피처 추출 (각 액션별)
        features = []
        for i, action in enumerate(seq):
            feat = self._extract_features(action, i, len(seq))
            features.append(feat)

        # 패딩
        n_features = len(features[0]) if features else 12
        while len(features) < self.max_len:
            features.insert(0, [0.0] * n_features)  # 앞에 패딩

        features = torch.tensor(features, dtype=torch.float32)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return features, target
        return features

    def _extract_features(self, action, pos, total):
        """액션별 피처 추출"""
        start_x, start_y = action['start_x'], action['start_y']
        end_x, end_y = action['end_x'], action['end_y']

        # 기본 피처
        dx = end_x - start_x
        dy = end_y - start_y
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        # Goal-oriented 피처 (핵심!)
        dist_to_goal = np.sqrt((GOAL_X - start_x)**2 + (FIELD_WIDTH/2 - start_y)**2)
        angle_to_goal = np.arctan2(FIELD_WIDTH/2 - start_y, GOAL_X - start_x)

        # 정규화된 위치
        norm_x = start_x / FIELD_LENGTH
        norm_y = start_y / FIELD_WIDTH

        # 시퀀스 위치 정보
        rel_pos = pos / max(total - 1, 1)
        is_last = 1.0 if pos == total - 1 else 0.0

        return [
            norm_x, norm_y,           # 정규화 위치
            dx / 50, dy / 50,         # 정규화 이동
            dist / 50,                # 정규화 거리
            angle / np.pi,            # 정규화 각도
            dist_to_goal / 100,       # 골대까지 거리
            angle_to_goal / np.pi,    # 골대 방향
            action.get('is_home', 1), # 홈/원정
            rel_pos,                  # 시퀀스 내 상대 위치
            is_last,                  # 마지막 액션 여부
            1.0                       # 패딩 마스크
        ]


class TransformerModel(nn.Module):
    """Transformer 기반 시퀀스 모델"""
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # end_x, end_y
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 마지막 시점
        x = self.fc(x)
        return x


def load_sequences(csv_path):
    """시퀀스 데이터 로드"""
    df = pd.read_csv(csv_path)

    sequences = []
    targets = []
    game_ids = []

    for game_ep, ep_df in tqdm(df.groupby('game_episode'), desc="Loading"):
        ep_df = ep_df.sort_values('time_seconds').reset_index(drop=True)

        seq = []
        for _, row in ep_df.iterrows():
            seq.append({
                'start_x': row['start_x'],
                'start_y': row['start_y'],
                'end_x': row['end_x'],
                'end_y': row['end_y'],
                'is_home': 1 if row['is_home'] else 0
            })

        sequences.append(seq)
        targets.append([ep_df.iloc[-1]['end_x'], ep_df.iloc[-1]['end_y']])
        game_ids.append(int(game_ep.split('_')[0]))

    return sequences, np.array(targets), np.array(game_ids)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            preds.append(pred.cpu().numpy())
            targets.append(batch_y.numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # Euclidean distance
    dist = np.sqrt(np.sum((preds - targets)**2, axis=1))
    return np.mean(dist)


def main():
    print("="*70)
    print("exp_050: Transformer 시퀀스 모델")
    print("목표: 12점대 (1위)")
    print("="*70)

    # 데이터 로드
    print("\n[1] 데이터 로드...")
    sequences, targets, game_ids = load_sequences(TRAIN_CSV)
    print(f"  에피소드: {len(sequences)}")
    print(f"  타겟 범위: x={targets[:,0].min():.1f}-{targets[:,0].max():.1f}, y={targets[:,1].min():.1f}-{targets[:,1].max():.1f}")

    # Cross-validation
    print("\n[2] 3-Fold CV 학습...")
    gkf = GroupKFold(n_splits=3)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(sequences, groups=game_ids)):
        print(f"\n--- Fold {fold+1} ---")

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]

        # Dataset & Loader
        train_dataset = SoccerDataset(train_seqs, train_targets, max_len=50)
        val_dataset = SoccerDataset(val_seqs, val_targets, max_len=50)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Model
        model = TransformerModel(
            input_dim=12,
            d_model=64,
            nhead=4,
            num_layers=3,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        criterion = nn.MSELoss()

        # Training
        best_score = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_score = evaluate(model, val_loader, device)
            scheduler.step()

            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(model.state_dict(), f'model_fold{fold}.pt')
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_score={val_score:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        cv_scores.append(best_score)
        print(f"  Fold {fold+1} Best: {best_score:.4f}")

    mean_cv = np.mean(cv_scores)
    print(f"\n" + "="*70)
    print(f"CV Score: {mean_cv:.4f}")
    print(f"Individual: {cv_scores}")
    print("="*70)

    return mean_cv


if __name__ == "__main__":
    main()
