"""
K리그 패스 좌표 예측 - LSTM 시퀀스 모델 v1
상위권 진입을 위한 딥러닝 접근

목표: Public 13-14점대 (Zone 통계 16.35 대비 2-3점 향상)
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
BATCH_SIZE = 64  # 메모리 절약
EPOCHS = 30  # 빠른 테스트
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print(f"Sequence Length: {SEQ_LEN}")
print(f"Hidden Size: {HIDDEN_SIZE}")
print()

# ==========================================
# 데이터 로드
# ==========================================

print("[1] 데이터 로드...")
train = pd.read_csv(f'{DATA_DIR}train.csv')
test = pd.read_csv(f'{DATA_DIR}test.csv')

print(f"  Train episodes: {train['game_episode'].nunique():,}")
print(f"  Train samples: {len(train):,}")
print(f"  Test episodes: {test['game_episode'].nunique():,}")
print(f"  Test samples: {len(test):,}")
print()

# ==========================================
# 데이터 전처리 함수
# ==========================================

def create_sequences(df, seq_len=5, is_train=True):
    """
    에피소드별로 시퀀스 생성

    Args:
        df: DataFrame
        seq_len: 시퀀스 길이
        is_train: True면 target 포함, False면 target 없음

    Returns:
        X: [num_samples, seq_len, 5]
        y: [num_samples, 2] (is_train=True일 때만)
        indices: 원본 DataFrame의 인덱스 (is_train=False일 때)
    """
    X_list = []
    y_list = []
    indices_list = []

    for episode_id in df['game_episode'].unique():
        episode = df[df['game_episode'] == episode_id].sort_values('time_seconds').reset_index(drop=True)

        # 시퀀스 길이보다 짧으면 skip
        if len(episode) <= seq_len:
            continue

        # 시간 차이 계산
        episode['time_diff'] = episode['time_seconds'].diff().fillna(0)

        # 에피소드 내에서 시퀀스 생성
        for i in range(seq_len, len(episode)):
            # Input: 이전 seq_len개 패스
            seq = episode.iloc[i-seq_len:i]

            features = []
            for _, row in seq.iterrows():
                # 5 features: start_x, start_y, prev_dx, prev_dy, time_diff
                feat = [
                    row['start_x'] / 105.0,  # Normalize to [0, 1]
                    row['start_y'] / 68.0,
                    row.get('prev_dx', 0) / 105.0,
                    row.get('prev_dy', 0) / 68.0,
                    np.log1p(row['time_diff']) / 10.0  # Log scale
                ]
                features.append(feat)

            X_list.append(features)

            if is_train:
                # Target: 현재 패스의 종료 좌표
                target_row = episode.iloc[i]
                y_list.append([
                    target_row['end_x'] / 105.0,
                    target_row['end_y'] / 68.0
                ])
            else:
                # Test의 경우 원본 인덱스 저장
                indices_list.append(episode.index[i])

    X = np.array(X_list, dtype=np.float32)

    if is_train:
        y = np.array(y_list, dtype=np.float32)
        return X, y
    else:
        return X, np.array(indices_list)

# ==========================================
# PyTorch Dataset
# ==========================================

class PassDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  # Output: [end_x, end_y]

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_size]

        Returns:
            out: [batch, 2]
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]

        # FC layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [batch, 2]

        return x

# ==========================================
# 학습 함수
# ==========================================

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    모델 학습
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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

            # Forward
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward
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

        # Scheduler
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if patience_counter >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, best_val_loss

# ==========================================
# 평가 함수
# ==========================================

def evaluate_model(model, X_val, y_val):
    """
    모델 평가 (Euclidean distance)
    """
    model.eval()

    dataset = PassDataset(X_val, y_val)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
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

print("[2] 시퀀스 생성 (샘플링 20%)...")
# 메모리 절약: 전체 데이터의 20%만 사용
train_sample = train.sample(frac=0.2, random_state=42)
X_all, y_all = create_sequences(train_sample, seq_len=SEQ_LEN, is_train=True)
print(f"  Total sequences: {len(X_all):,}")
print(f"  Input shape: {X_all.shape}")
print(f"  Target shape: {y_all.shape}")
print()

print("[3] GroupKFold 교차 검증 (Fold 1-3만)...")
gkf = GroupKFold(n_splits=5)
groups = train['game_episode']

cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train, groups=groups)):
    if fold >= 3:  # Fold 1-3만 사용 (기존 전략 유지)
        break

    print(f"\nFold {fold+1}:")

    # Split data
    train_fold = train.iloc[train_idx]
    val_fold = train.iloc[val_idx]

    # Create sequences
    X_train, y_train = create_sequences(train_fold, seq_len=SEQ_LEN, is_train=True)
    X_val, y_val = create_sequences(val_fold, seq_len=SEQ_LEN, is_train=True)

    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Val sequences: {len(X_val):,}")

    # Create DataLoaders
    train_dataset = PassDataset(X_train, y_train)
    val_dataset = PassDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train model
    model = PassPredictionLSTM(input_size=5, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    model, best_loss = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    # Evaluate
    score = evaluate_model(model, X_val, y_val)
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

print("[4] Test 예측...")
X_test, test_indices = create_sequences(test, seq_len=SEQ_LEN, is_train=False)
print(f"  Test sequences: {len(X_test):,}")

# 앙상블 예측 (Fold 1-3 평균)
all_test_preds = []

for fold_idx, model in enumerate(models):
    model.eval()

    test_dataset = PassDataset(X_test, np.zeros((len(X_test), 2)))  # Dummy targets
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    fold_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(DEVICE)
            outputs = model(batch_X)
            fold_preds.append(outputs.cpu().numpy())

    fold_preds = np.vstack(fold_preds)
    all_test_preds.append(fold_preds)

# 평균
ensemble_preds = np.mean(all_test_preds, axis=0)

# Denormalize
ensemble_preds[:, 0] *= 105
ensemble_preds[:, 1] *= 68

print(f"  앙상블 예측 완료")
print()

# ==========================================
# 제출 파일 생성
# ==========================================

print("[5] 제출 파일 생성...")

# test의 모든 행에 대해 예측값 할당
submission = test[['id']].copy()
submission['end_x'] = 0.0
submission['end_y'] = 0.0

# 시퀀스로 예측된 인덱스에 대해서만 값 할당
submission.loc[test_indices, 'end_x'] = ensemble_preds[:, 0]
submission.loc[test_indices, 'end_y'] = ensemble_preds[:, 1]

# 시퀀스 길이 미만인 샘플들은 간단한 휴리스틱 사용 (start + prev_d)
mask = (submission['end_x'] == 0) & (submission['end_y'] == 0)
if mask.sum() > 0:
    submission.loc[mask, 'end_x'] = test.loc[mask, 'start_x'] + test.loc[mask, 'prev_dx']
    submission.loc[mask, 'end_y'] = test.loc[mask, 'start_y'] + test.loc[mask, 'prev_dy']

    # Clip
    submission.loc[mask, 'end_x'] = submission.loc[mask, 'end_x'].clip(0, 105)
    submission.loc[mask, 'end_y'] = submission.loc[mask, 'end_y'].clip(0, 68)

submission.to_csv('submission_lstm_v1.csv', index=False)
print(f"  submission_lstm_v1.csv 저장 완료")
print()

print("=" * 80)
print("최종 요약")
print("=" * 80)
print(f"[성능]")
print(f"  Fold 1-3 CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"  예상 Public: {np.mean(cv_scores) + 0.03:.4f} (Gap +0.03 가정)")
print()
print(f"[모델]")
print(f"  LSTM(hidden={HIDDEN_SIZE}, layers={NUM_LAYERS}, seq_len={SEQ_LEN})")
print(f"  Training samples: {len(X_all):,}")
print()
print(f"[제출 파일]")
print(f"  submission_lstm_v1.csv")
print()
print(f"[목표]")
print(f"  Public 13-14점대")
print(f"  상위 25등 진입")
print()
print("=" * 80)
print("완료!")
print("=" * 80)
