"""
K리그 패스 좌표 예측 - LSTM 모델
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings('ignore')

# CPU 강제 사용 (GPU 호환성 문제 방지)
device = torch.device('cpu')
print(f"Device: {device}")

DATA_DIR = Path(".")

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n[1] 데이터 로드...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

# 테스트 데이터 로드
test_episodes = []
for _, row in test_df.iterrows():
    ep_df = pd.read_csv(DATA_DIR / row['path'])
    test_episodes.append(ep_df)
test_all = pd.concat(test_episodes, ignore_index=True)

print(f"Train: {len(train_df):,} rows")
print(f"Test: {len(test_all):,} rows")

# =============================================================================
# 2. 전처리
# =============================================================================
print("\n[2] 전처리...")

# type_name 인코딩
type_encoder = LabelEncoder()
all_types = pd.concat([train_df['type_name'], test_all['type_name']])
type_encoder.fit(all_types)
train_df['type_encoded'] = type_encoder.transform(train_df['type_name'])
test_all['type_encoded'] = type_encoder.transform(test_all['type_name'])

# result_name 인코딩
result_encoder = LabelEncoder()
train_df['result_name'] = train_df['result_name'].fillna('None')
test_all['result_name'] = test_all['result_name'].fillna('None')
all_results = pd.concat([train_df['result_name'], test_all['result_name']])
result_encoder.fit(all_results)
train_df['result_encoded'] = result_encoder.transform(train_df['result_name'])
test_all['result_encoded'] = result_encoder.transform(test_all['result_name'])

n_types = len(type_encoder.classes_)
n_results = len(result_encoder.classes_)
print(f"Type classes: {n_types}")
print(f"Result classes: {n_results}")

# =============================================================================
# 3. 시퀀스 데이터 생성
# =============================================================================
print("\n[3] 시퀀스 데이터 생성...")

def create_sequences(df, max_len=50, is_train=True):
    """에피소드별 시퀀스 데이터 생성"""
    sequences = []
    targets = []
    game_episodes = []
    game_ids = []

    for game_ep, group in df.groupby('game_episode'):
        group = group.sort_values('action_id')

        # 시퀀스 피처 (최대 max_len개)
        seq = group.tail(max_len)

        # 수치형 피처
        numeric_feats = seq[['start_x', 'start_y', 'time_seconds', 'period_id', 'is_home']].copy()
        numeric_feats['start_x'] = numeric_feats['start_x'] / 105.0  # 정규화
        numeric_feats['start_y'] = numeric_feats['start_y'] / 68.0
        numeric_feats['time_seconds'] = numeric_feats['time_seconds'] / 5400.0  # 90분
        numeric_feats['is_home'] = numeric_feats['is_home'].astype(float)

        # end_x, end_y도 추가 (마지막 제외)
        end_x = seq['end_x'].fillna(seq['start_x']).values / 105.0
        end_y = seq['end_y'].fillna(seq['start_y']).values / 68.0

        # 범주형 피처
        type_enc = seq['type_encoded'].values
        result_enc = seq['result_encoded'].values

        # 시퀀스 결합
        seq_array = np.column_stack([
            numeric_feats.values,
            end_x,
            end_y,
            type_enc,
            result_enc
        ])

        sequences.append(torch.FloatTensor(seq_array))
        game_episodes.append(game_ep)
        game_ids.append(group['game_id'].iloc[0])

        # 타겟 (Train만)
        if is_train:
            last_row = group.iloc[-1]
            target_x = last_row['end_x'] / 105.0  # 정규화
            target_y = last_row['end_y'] / 68.0
            targets.append([target_x, target_y])

    return sequences, targets, game_episodes, game_ids

train_seqs, train_targets, train_eps, train_game_ids = create_sequences(train_df, is_train=True)
test_seqs, _, test_eps, _ = create_sequences(test_all, is_train=False)

print(f"Train sequences: {len(train_seqs)}")
print(f"Test sequences: {len(test_seqs)}")

# 결측치 제거
valid_idx = [i for i, t in enumerate(train_targets) if not np.isnan(t).any()]
train_seqs = [train_seqs[i] for i in valid_idx]
train_targets = [train_targets[i] for i in valid_idx]
train_eps = [train_eps[i] for i in valid_idx]
train_game_ids = [train_game_ids[i] for i in valid_idx]

print(f"Valid train sequences: {len(train_seqs)}")

# =============================================================================
# 4. Dataset 정의
# =============================================================================

class PassDataset(Dataset):
    def __init__(self, sequences, targets=None):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.targets is not None:
            target = torch.FloatTensor(self.targets[idx])
            return seq, target
        return seq

def collate_fn(batch):
    if len(batch[0]) == 2:  # with targets
        seqs, targets = zip(*batch)
        lengths = torch.LongTensor([len(s) for s in seqs])
        seqs_padded = pad_sequence(seqs, batch_first=True)
        targets = torch.stack(targets)
        return seqs_padded, lengths, targets
    else:  # without targets
        seqs = batch
        lengths = torch.LongTensor([len(s) for s in seqs])
        seqs_padded = pad_sequence(seqs, batch_first=True)
        return seqs_padded, lengths

# =============================================================================
# 5. 모델 정의
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # end_x, end_y
        )

    def forward(self, x, lengths):
        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n, c_n) = self.lstm(packed)

        # 마지막 hidden state 사용 (bidirectional)
        h_forward = h_n[-2, :, :]  # 마지막 forward
        h_backward = h_n[-1, :, :]  # 마지막 backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        out = self.fc(h_combined)
        return out

# =============================================================================
# 6. 학습
# =============================================================================
print("\n[4] 모델 학습 (5-Fold CV)...")

N_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.001

gkf = GroupKFold(n_splits=N_FOLDS)
game_ids_array = np.array(train_game_ids)

oof_preds = np.zeros((len(train_seqs), 2))
test_preds = np.zeros((len(test_seqs), 2))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(train_seqs, groups=game_ids_array)):
    print(f"\n  Fold {fold+1}/{N_FOLDS}")

    # 데이터 준비
    train_seqs_fold = [train_seqs[i] for i in train_idx]
    train_targets_fold = [train_targets[i] for i in train_idx]
    val_seqs_fold = [train_seqs[i] for i in val_idx]
    val_targets_fold = [train_targets[i] for i in val_idx]

    train_dataset = PassDataset(train_seqs_fold, train_targets_fold)
    val_dataset = PassDataset(val_seqs_fold, val_targets_fold)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 모델 초기화
    model = LSTMModel(input_dim=9, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            seqs, lengths, targets = batch
            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(seqs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch in val_loader:
                seqs, lengths, targets = batch
                seqs, targets = seqs.to(device), targets.to(device)

                outputs = model(seqs, lengths)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                val_preds.append(outputs.cpu().numpy())
                val_trues.append(targets.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 유클리드 거리 계산
        val_preds_np = np.vstack(val_preds)
        val_trues_np = np.vstack(val_trues)
        # 역정규화
        val_preds_np[:, 0] *= 105
        val_preds_np[:, 1] *= 68
        val_trues_np[:, 0] *= 105
        val_trues_np[:, 1] *= 68
        euclidean = np.sqrt(((val_preds_np - val_trues_np) ** 2).sum(axis=1)).mean()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, euclidean={euclidean:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Best 모델 로드
    model.load_state_dict(best_model_state)
    model.eval()

    # OOF 예측
    with torch.no_grad():
        val_dataset_full = PassDataset(val_seqs_fold, val_targets_fold)
        val_loader_full = DataLoader(val_dataset_full, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        fold_preds = []
        for batch in val_loader_full:
            seqs, lengths, _ = batch
            seqs = seqs.to(device)
            outputs = model(seqs, lengths)
            fold_preds.append(outputs.cpu().numpy())

        fold_preds = np.vstack(fold_preds)
        fold_preds[:, 0] *= 105
        fold_preds[:, 1] *= 68
        oof_preds[val_idx] = fold_preds

    # 테스트 예측
    test_dataset = PassDataset(test_seqs)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        test_fold_preds = []
        for batch in test_loader:
            seqs, lengths = batch
            seqs = seqs.to(device)
            outputs = model(seqs, lengths)
            test_fold_preds.append(outputs.cpu().numpy())

        test_fold_preds = np.vstack(test_fold_preds)
        test_fold_preds[:, 0] *= 105
        test_fold_preds[:, 1] *= 68
        test_preds += test_fold_preds / N_FOLDS

    # Fold 스코어
    fold_trues = np.array([train_targets[i] for i in val_idx])
    fold_trues[:, 0] *= 105
    fold_trues[:, 1] *= 68
    fold_dist = np.sqrt(((oof_preds[val_idx] - fold_trues) ** 2).sum(axis=1)).mean()
    fold_scores.append(fold_dist)
    print(f"    Fold {fold+1} Score: {fold_dist:.4f}")

# CV 스코어
train_targets_np = np.array(train_targets)
train_targets_np[:, 0] *= 105
train_targets_np[:, 1] *= 68
cv_dist = np.sqrt(((oof_preds - train_targets_np) ** 2).sum(axis=1)).mean()
print(f"\n  CV Score (Mean): {cv_dist:.4f}")
print(f"  CV Score (Std): {np.std(fold_scores):.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n[5] 제출 파일 생성...")

# 좌표 클리핑
test_preds[:, 0] = np.clip(test_preds[:, 0], 0, 105)
test_preds[:, 1] = np.clip(test_preds[:, 1], 0, 68)

submission = pd.DataFrame({
    'game_episode': test_eps,
    'end_x': test_preds[:, 0],
    'end_y': test_preds[:, 1]
})

submission = sample_sub[['game_episode']].merge(submission, on='game_episode', how='left')
submission.to_csv('submission_lstm.csv', index=False)

print(f"제출 파일 저장: submission_lstm.csv")
print(submission.head(10))

print("\n완료!")
