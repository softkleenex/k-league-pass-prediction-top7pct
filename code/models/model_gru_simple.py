"""
K리그 패스 좌표 예측 - Simple GRU 시퀀스 모델
강한 정규화 + 단순 아키텍처
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# GPU/CPU 설정 - Triton 호환성 문제로 CPU 사용
device = torch.device('cpu')
print(f"Using device: {device} (CPU forced due to Triton compatibility)")

DATA_DIR = Path(".")

print("=" * 70)
print("K리그 패스 좌표 예측 - Simple GRU 시퀀스 모델")
print("=" * 70)

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 테스트 에피소드 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train episodes: {train_df['game_episode'].nunique():,}")
print(f"Test episodes: {test_all['game_episode'].nunique():,}")

# =============================================================================
# 2. 시퀀스 피처 준비
# =============================================================================
print("\n[2] 시퀀스 피처 준비...")

def prepare_sequence_features(df):
    """각 액션에 대한 피처 생성"""
    df = df.copy()

    # 기본 위치 피처 (0-1 정규화)
    df['start_x_norm'] = df['start_x'] / 105
    df['start_y_norm'] = df['start_y'] / 68

    # 이동량 (있는 경우)
    df['dx'] = (df['end_x'] - df['start_x']).fillna(0) / 105
    df['dy'] = (df['end_y'] - df['start_y']).fillna(0) / 68

    # 골문 방향
    df['dist_to_goal'] = np.sqrt((105 - df['start_x'])**2 + (34 - df['start_y'])**2) / 120
    df['angle_to_goal'] = np.arctan2(34 - df['start_y'], 105 - df['start_x']) / np.pi

    # 경기 시간 정규화
    df['time_norm'] = df['time_seconds'] / 3000  # 약 50분

    # 액션 타입 인코딩 (주요 3가지)
    df['is_pass'] = (df['type_name'] == 'Pass').astype(float)
    df['is_carry'] = (df['type_name'] == 'Carry').astype(float)
    df['is_other'] = (~df['type_name'].isin(['Pass', 'Carry'])).astype(float)

    # 전반/후반
    df['is_second_half'] = (df['period_id'] == 2).astype(float)

    return df

train_df = prepare_sequence_features(train_df)
test_all = prepare_sequence_features(test_all)

# 사용할 피처
seq_features = [
    'start_x_norm', 'start_y_norm',
    'dx', 'dy',
    'dist_to_goal', 'angle_to_goal',
    'time_norm',
    'is_pass', 'is_carry', 'is_other',
    'is_second_half'
]

print(f"시퀀스 피처 수: {len(seq_features)}")

# =============================================================================
# 3. 시퀀스 데이터 생성
# =============================================================================
print("\n[3] 시퀀스 데이터 생성...")

MAX_SEQ_LEN = 50  # 최대 시퀀스 길이

def create_sequences(df, is_train=True):
    """에피소드별 시퀀스 생성"""
    episodes = df.groupby('game_episode')
    sequences = []
    targets = []
    game_ids = []
    episode_ids = []

    for ep_id, ep_df in episodes:
        ep_df = ep_df.sort_values('action_id')

        # 시퀀스 피처 추출
        seq = ep_df[seq_features].values

        # 패딩/자르기
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]  # 마지막 N개만
        elif len(seq) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(seq)
            seq = np.pad(seq, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)

        sequences.append(seq)

        # 타겟 (마지막 액션의 end_x, end_y)
        if is_train:
            last_row = ep_df.iloc[-1]
            targets.append([last_row['end_x'], last_row['end_y']])
        else:
            last_row = ep_df.iloc[-1]
            targets.append([last_row['start_x'], last_row['start_y']])  # placeholder

        game_ids.append(ep_df['game_id'].iloc[0])
        episode_ids.append(ep_id)

    return np.array(sequences), np.array(targets), np.array(game_ids), episode_ids

print("  Train 시퀀스 생성 중...")
X_train_seq, y_train, train_game_ids, train_ep_ids = create_sequences(train_df, is_train=True)

print("  Test 시퀀스 생성 중...")
X_test_seq, _, _, test_ep_ids = create_sequences(test_all, is_train=False)

print(f"Train sequences shape: {X_train_seq.shape}")
print(f"Test sequences shape: {X_test_seq.shape}")
print(f"Target shape: {y_train.shape}")

# =============================================================================
# 4. PyTorch Dataset & Model
# =============================================================================
print("\n[4] 모델 정의...")

class PassDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class SimpleGRU(nn.Module):
    """단순한 GRU 모델 - 강한 정규화"""
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.5):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        # 출력층 (매우 단순)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)  # end_x, end_y
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden)
        out = h_n[-1]  # 마지막 레이어의 hidden state
        out = self.dropout(out)
        out = self.fc(out)
        return out

# =============================================================================
# 5. 학습 함수
# =============================================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for seqs, targets in loader:
        seqs, targets = seqs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(seqs)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # 유클리드 거리
    dist = np.sqrt((predictions[:, 0] - actuals[:, 0])**2 + (predictions[:, 1] - actuals[:, 1])**2)
    return dist.mean(), predictions

# =============================================================================
# 6. GroupKFold 교차 검증
# =============================================================================
print("\n[5] GroupKFold 교차 검증...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_preds = np.zeros((len(X_train_seq), 2))
test_preds = np.zeros((len(X_test_seq), 2))
fold_scores = []

# 하이퍼파라미터
HIDDEN_DIM = 32
NUM_LAYERS = 1
DROPOUT = 0.5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01  # L2 정규화
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_seq, groups=train_game_ids)):
    print(f"\n  Fold {fold+1}/{n_splits}")

    X_tr, X_val = X_train_seq[train_idx], X_train_seq[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # DataLoader
    train_dataset = PassDataset(X_tr, y_tr)
    val_dataset = PassDataset(X_val, y_val)
    test_dataset = PassDataset(X_test_seq, np.zeros((len(X_test_seq), 2)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 초기화
    model = SimpleGRU(
        input_dim=len(seq_features),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_score = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_score, _ = evaluate(model, val_loader)

        scheduler.step(val_score)

        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Score = {val_score:.4f}")

    # Best 모델 로드
    model.load_state_dict(best_model_state)

    # OOF 예측
    _, val_preds = evaluate(model, val_loader)
    oof_preds[val_idx] = val_preds

    # 테스트 예측
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for seqs, _ in test_loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            fold_test_preds.append(outputs.cpu().numpy())
    fold_test_preds = np.vstack(fold_test_preds)
    test_preds += fold_test_preds / n_splits

    fold_score = best_val_score
    fold_scores.append(fold_score)
    print(f"    Fold {fold+1} Best Score: {fold_score:.4f}")

# 전체 OOF 점수
oof_dist = np.sqrt((oof_preds[:, 0] - y_train[:, 0])**2 + (oof_preds[:, 1] - y_train[:, 1])**2)
gru_score = oof_dist.mean()

print("\n" + "=" * 70)
print(f"GRU CV Score: {gru_score:.4f}")
print(f"Fold Scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Std: {np.std(fold_scores):.4f}")
print("=" * 70)

# =============================================================================
# 7. Zone Baseline과 앙상블
# =============================================================================
print("\n[6] Zone Baseline과 앙상블...")

# Zone Baseline 준비
def get_zone_6x6(x, y):
    x_zone = min(5, int(x / (105 / 6)))
    y_zone = min(5, int(y / (68 / 6)))
    return x_zone * 6 + y_zone

# Train 마지막 액션 추출
train_last = train_df.groupby('game_episode').last().reset_index()
train_last = train_last.dropna(subset=['end_x', 'end_y'])
train_last['zone'] = train_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)
train_last['delta_x'] = train_last['end_x'] - train_last['start_x']
train_last['delta_y'] = train_last['end_y'] - train_last['start_y']

zone_stats = train_last.groupby('zone').agg({
    'delta_x': 'median',
    'delta_y': 'median'
}).to_dict()

# Test Zone 예측
test_last = test_all.groupby('game_episode').last().reset_index()
test_last['zone'] = test_last.apply(lambda r: get_zone_6x6(r['start_x'], r['start_y']), axis=1)

zone_pred_x_test = []
zone_pred_y_test = []
for _, row in test_last.iterrows():
    zone = get_zone_6x6(row['start_x'], row['start_y'])
    dx = zone_stats['delta_x'].get(zone, 0)
    dy = zone_stats['delta_y'].get(zone, 0)
    zone_pred_x_test.append(np.clip(row['start_x'] + dx, 0, 105))
    zone_pred_y_test.append(np.clip(row['start_y'] + dy, 0, 68))

zone_pred_x_test = np.array(zone_pred_x_test)
zone_pred_y_test = np.array(zone_pred_y_test)

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n[7] 제출 파일 생성...")

# 1. Pure GRU
test_preds_clipped = np.clip(test_preds, [[0, 0]], [[105, 68]])

# episode_id 순서 맞추기
test_ep_to_pred = dict(zip(test_ep_ids, test_preds_clipped))

submission_gru = []
for ep_id in sample_sub['game_episode']:
    pred = test_ep_to_pred.get(ep_id, [52.5, 34])
    submission_gru.append({'game_episode': ep_id, 'end_x': pred[0], 'end_y': pred[1]})

submission_gru = pd.DataFrame(submission_gru)
submission_gru.to_csv('submission_gru_simple.csv', index=False)
print(f"  1. submission_gru_simple.csv 저장 (CV: {gru_score:.4f})")

# 2. Zone + GRU 앙상블
for alpha in [0.6, 0.7, 0.8]:
    ensemble_x = alpha * zone_pred_x_test + (1 - alpha) * test_preds_clipped[:, 0]
    ensemble_y = alpha * zone_pred_y_test + (1 - alpha) * test_preds_clipped[:, 1]

    ensemble_x = np.clip(ensemble_x, 0, 105)
    ensemble_y = np.clip(ensemble_y, 0, 68)

    # episode_id 순서 맞추기
    test_ep_to_ens = dict(zip(test_ep_ids, zip(ensemble_x, ensemble_y)))

    submission_ens = []
    for ep_id in sample_sub['game_episode']:
        pred = test_ep_to_ens.get(ep_id, (52.5, 34))
        submission_ens.append({'game_episode': ep_id, 'end_x': pred[0], 'end_y': pred[1]})

    submission_ens = pd.DataFrame(submission_ens)
    filename = f'submission_gru_zone{int(alpha*100)}.csv'
    submission_ens.to_csv(filename, index=False)
    print(f"  2. {filename} 저장")

# =============================================================================
# 9. 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)

print(f"\n[모델 비교]")
print(f"  Zone Baseline (6x6 median): CV = 16.68 → Public = 16.85")
print(f"  Simple GRU:                 CV = {gru_score:.4f}")

print(f"\n[생성된 파일]")
print(f"  1. submission_gru_simple.csv")
print(f"  2. submission_gru_zone60.csv")
print(f"  3. submission_gru_zone70.csv")
print(f"  4. submission_gru_zone80.csv")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
