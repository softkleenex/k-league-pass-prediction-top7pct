"""
K리그 패스 좌표 예측 - LSTM v2 (메모리 효율적)
On-the-fly 시퀀스 생성으로 OOM 문제 해결

목표: CV 13-15점대 (XGBoost 15.73 대비 동등 또는 향상)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 설정
# ==========================================

DATA_DIR = ''  # 루트 디렉토리
SEQ_LEN = 3  # 이전 3개 패스 사용 (메모리 절약)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256  # 64 → 256 (4배 증가, 시간 단축: 20시간 → 5-8시간)
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Sequence Length: {SEQ_LEN}")
print(f"Batch Size: {BATCH_SIZE}")
print()

# ==========================================
# 데이터 로드
# ==========================================

print("[1] 데이터 로드...")
train = pd.read_csv(f'{DATA_DIR}train.csv')
test = pd.read_csv(f'{DATA_DIR}test.csv')

# 시간 차이 사전 계산
train = train.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
train['time_diff'] = train.groupby('game_episode')['time_seconds'].diff().fillna(0)

# test는 time_seconds 없을 수 있음
if 'time_seconds' in test.columns:
    test = test.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
    test['time_diff'] = test.groupby('game_episode')['time_seconds'].diff().fillna(0)
else:
    test['time_diff'] = 0  # 기본값

print(f"  Train episodes: {train['game_episode'].nunique():,}")
print(f"  Train samples: {len(train):,}")
print(f"  Test episodes: {test['game_episode'].nunique():,}")
print(f"  Test samples: {len(test):,}")
print()

# ==========================================
# 메모리 효율적 Dataset (Lazy Loading)
# ==========================================

class PassSequenceDataset(Dataset):
    """
    시퀀스를 사전 생성하지 않고, 필요할 때만 생성 (on-the-fly)
    메모리 사용량: O(N) → O(1) per batch
    최적화: groupby로 episode별 미리 저장 → __init__ 18-36배, __getitem__ 100배+ 빠름
    """
    def __init__(self, df, seq_len=3):
        self.seq_len = seq_len

        # 미리 정렬 및 episode별로 그룹화 (O(N) 한 번만!)
        df_sorted = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
        self.episode_groups = dict(list(df_sorted.groupby('game_episode')))

        # 전체 df도 저장 (인덱스 접근용)
        self.df = df_sorted

        # 시퀀스 생성 가능한 인덱스만 저장
        self.valid_indices = []

        for episode_id, episode_df in self.episode_groups.items():
            if len(episode_df) <= seq_len:
                continue

            # 시퀀스 길이 이후부터 유효
            for i in range(seq_len, len(episode_df)):
                self.valid_indices.append(episode_df.index[i])

        print(f"    Valid sequences: {len(self.valid_indices):,}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        필요할 때만 시퀀스 생성 (lazy loading)
        최적화: 미리 저장된 episode_groups에서 O(1) 접근
        """
        target_idx = self.valid_indices[idx]
        episode_id = self.df.loc[target_idx, 'game_episode']

        # 미리 저장된 그룹에서 O(1) 접근! (이미 정렬됨)
        episode_df = self.episode_groups[episode_id]

        # 타겟 위치 찾기
        target_pos = episode_df.index.get_loc(target_idx)

        # 이전 seq_len개 패스
        seq_indices = episode_df.index[target_pos-self.seq_len:target_pos]

        # 시퀀스 생성
        features = []
        for seq_idx in seq_indices:
            row = self.df.loc[seq_idx]
            feat = [
                row['start_x'] / 105.0,
                row['start_y'] / 68.0,
                row.get('prev_dx', 0) / 105.0,
                row.get('prev_dy', 0) / 68.0,
                np.log1p(row['time_diff']) / 10.0
            ]
            features.append(feat)

        # Target
        target_row = self.df.loc[target_idx]
        target = [
            target_row['end_x'] / 105.0,
            target_row['end_y'] / 68.0
        ]

        return torch.FloatTensor(features), torch.FloatTensor(target)

# ==========================================
# LSTM 모델
# ==========================================

class PassPredictionLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # FC layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]

        # FC layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# ==========================================
# 학습 함수
# ==========================================

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss

# ==========================================
# 평가 함수
# ==========================================

def evaluate_model(model, val_loader):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Denormalize
    preds[:, 0] *= 105
    preds[:, 1] *= 68
    targets[:, 0] *= 105
    targets[:, 1] *= 68

    # Euclidean distance
    distances = np.sqrt(np.sum((preds - targets) ** 2, axis=1))
    mean_distance = distances.mean()

    return mean_distance

# ==========================================
# 메인 실행
# ==========================================

print("[2] GroupKFold 교차 검증 (Fold 1-3)...")
gkf = GroupKFold(n_splits=5)
groups = train['game_episode']

cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
    if fold >= 3:  # Fold 1-3만
        break

    print(f"\nFold {fold+1}:")

    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    # Dataset 생성 (메모리 효율적!)
    train_dataset = PassSequenceDataset(train_fold, seq_len=SEQ_LEN)
    val_dataset = PassSequenceDataset(val_fold, seq_len=SEQ_LEN)

    print("    [DEBUG] Dataset 생성 완료")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("    [DEBUG] DataLoader 생성 완료")

    # 모델 생성
    model = PassPredictionLSTM(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

    print("    [DEBUG] 모델 생성 및 GPU 이동 완료")

    # 학습
    model, best_loss = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # 평가
    score = evaluate_model(model, val_loader)
    cv_scores.append(score)
    models.append(model)

    print(f"  Fold {fold+1} Score: {score:.4f}")

print()
print("=" * 80)
print("교차 검증 결과")
print("=" * 80)
print(f"Fold 1-3 평균: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print()

# ==========================================
# Test 예측
# ==========================================

print("[3] Test 예측...")
test_dataset = PassSequenceDataset(test, seq_len=SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 앙상블 예측
all_test_preds = []

for model in models:
    model.eval()
    fold_preds = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            fold_preds.append(outputs.cpu().numpy())

    fold_preds = np.vstack(fold_preds)
    all_test_preds.append(fold_preds)

ensemble_preds = np.mean(all_test_preds, axis=0)

# Denormalize
ensemble_preds[:, 0] *= 105
ensemble_preds[:, 1] *= 68

print(f"  예측 완료: {len(ensemble_preds):,} samples")
print()

# ==========================================
# 제출 파일 생성
# ==========================================

print("[4] 제출 파일 생성...")

submission = test[['id']].copy()
submission['end_x'] = 0.0
submission['end_y'] = 0.0

# 예측값 할당 (시퀀스 생성된 인덱스만)
test_indices = test_dataset.valid_indices
submission.loc[test_indices, 'end_x'] = ensemble_preds[:, 0]
submission.loc[test_indices, 'end_y'] = ensemble_preds[:, 1]

# 시퀀스 미생성 샘플: 간단한 휴리스틱
mask = (submission['end_x'] == 0) & (submission['end_y'] == 0)
if mask.sum() > 0:
    submission.loc[mask, 'end_x'] = test.loc[mask, 'start_x'] + test.loc[mask, 'prev_dx']
    submission.loc[mask, 'end_y'] = test.loc[mask, 'start_y'] + test.loc[mask, 'prev_dy']
    submission.loc[mask, 'end_x'] = submission.loc[mask, 'end_x'].clip(0, 105)
    submission.loc[mask, 'end_y'] = submission.loc[mask, 'end_y'].clip(0, 68)

submission.to_csv('submission_lstm_v2.csv', index=False)
print(f"  submission_lstm_v2.csv 저장 완료")
print()

print("=" * 80)
print("최종 요약")
print("=" * 80)
print(f"[성능]")
print(f"  Fold 1-3 CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"  XGBoost (비교): 15.73")
print()
print(f"[모델]")
print(f"  LSTM v2 (메모리 효율적)")
print(f"  Lazy loading (on-the-fly 시퀀스 생성)")
print()
print(f"[제출 파일]")
print(f"  submission_lstm_v2.csv")
print()
print("=" * 80)
print("완료!")
print("=" * 80)
