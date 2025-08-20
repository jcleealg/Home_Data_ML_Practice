import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import time
import random
import optuna
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'

# ===== 隨機種子設定 =====
SEED = 42
# =======================

# ===== 超參數設定區 =====
N_TRIALS = 50 # Optuna 要嘗試的次數
EPOCHS = 1000 # 每次試驗的最大 Epochs
BATCH_SIZE = 50
EARLY_STOPPING_PATIENCE = 30
# ===================================

# ===== 裝置自動選擇 =====
USE_CPU_ONLY = True
if USE_CPU_ONLY:
    DEVICE = torch.device('cpu')
else:
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
print(f"使用裝置：{DEVICE}")
# =======================

# 設定隨機種子
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 讀取資料
try:
    df = pd.read_csv(PROCESSED_DATA_DIR / 'train_preprocessed_v2.csv')
except FileNotFoundError:
    print(f"錯誤：找不到資料檔案。請確認 train_preprocessed_v2.csv 位於 {PROCESSED_DATA_DIR}")
    exit()

drop_cols = ['SalePrice']
if 'Id' in df.columns:
    drop_cols.append('Id')
X = df.drop(drop_cols, axis=1).values.astype(np.float32)
y = df['SalePrice'].values.astype(np.float32)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=SEED)

X_train_tensor = torch.tensor(X_train).to(DEVICE)
y_train_tensor = torch.tensor(y_train).view(-1, 1).to(DEVICE)
X_val_tensor = torch.tensor(X_val).to(DEVICE)
y_val_tensor = torch.tensor(y_val).view(-1, 1).to(DEVICE)
X_test_tensor = torch.tensor(X_test).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 定義動態 DNN 模型
class DNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_units, dropout_p):
        super().__init__()
        layers = []
        in_features = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p[i]))
            in_features = n_units[i]
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Optuna 的 Objective Function
def objective(trial):
    # 定義超參數搜索空間
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int('n_layers', 2, 5)
    dropout_rates = [trial.suggest_float(f'dropout_l{i}', 0.1, 0.5) for i in range(n_layers)]
    n_units = [trial.suggest_int(f'n_units_l{i}', 64, 512, log=True) for i in range(n_layers)]
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    
    model = DNN(X_train.shape[1], n_layers, n_units, dropout_rates).to(DEVICE)
    criterion = nn.MSELoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 訓練迴圈
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break
            
        # Optuna Pruning: 提早中止沒有希望的試驗
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.sqrt(best_val_loss) # 返回驗證集的 RMSE

# --- Optuna Study --- #
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)

# --- 結果分析與儲存 --- #
print("===== Optuna 優化完成 =====")
print(f"嘗試次數: {len(study.trials)}")
print(f"最佳試驗: Trial {study.best_trial.number}")
print("  - 最佳驗證集 RMSE: {:.4f}".format(study.best_value))
print("  - 最佳超參數:")
for key, value in study.best_params.items():
    print(f"    - {key}: {value}")

# --- 使用最佳超參數重新訓練最終模型 (加入 Early Stopping) ---
print("\n使用最佳超參數重新訓練最終模型...")
best_params = study.best_params

# 從 best_params 中提取超參數
lr = best_params['lr']
n_layers = best_params['n_layers']
dropout_rates = [best_params[f'dropout_l{i}'] for i in range(n_layers)]
n_units = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
optimizer_name = best_params['optimizer']

final_model = DNN(X_train.shape[1], n_layers, n_units, dropout_rates).to(DEVICE)
criterion = nn.MSELoss()

if optimizer_name == 'Adam':
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
else:
    optimizer = optim.AdamW(final_model.parameters(), lr=lr)

start_train = time.time()
best_val_loss = float('inf')
epochs_no_improve = 0
best_final_model_state = None
final_epochs = EPOCHS * 2 # 給予更長的訓練時間以確保收斂

for epoch in range(final_epochs):
    final_model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = final_model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    final_model.eval()
    with torch.no_grad():
        val_pred = final_model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_final_model_state = final_model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}/{final_epochs}, Val RMSE: {np.sqrt(val_loss):.4f}")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"  在第 {epoch+1} 次 epoch 提早結束最終訓練。")
        break

# 載入表現最好的模型狀態
if best_final_model_state:
    final_model.load_state_dict(best_final_model_state)

end_train = time.time()
print(f"最終模型訓練時間：{end_train - start_train:.2f} 秒")

# 在測試集上評估最終模型
final_model.eval()
with torch.no_grad():
    y_pred_log = final_model(X_test_tensor).cpu().numpy().flatten()

y_pred_orig = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test)

final_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
final_r2 = r2_score(y_test_orig, y_pred_orig)

print("===== 最終模型評估結果 (測試集) =====")
print(f"  - 測試集 RMSE: {final_rmse:.2f}")
print(f"  - 測試集 R2 Score: {final_r2:.4f}")

# 儲存最佳模型與其設定
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = SAVED_MODELS_DIR / 'best_model.pth'
config_path = SAVED_MODELS_DIR / 'best_model_config.json'

print(f"\n正在儲存 RMSE 最低的模型至 {model_path}...")
torch.save(final_model.state_dict(), model_path)

best_model_config = study.best_params
best_model_config['test_rmse'] = final_rmse
best_model_config['test_r2'] = final_r2

print(f"正在儲存最佳模型的設定至 {config_path}...")
with open(config_path, 'w') as f:
    json.dump(best_model_config, f, indent=4)

print("儲存完成。")
