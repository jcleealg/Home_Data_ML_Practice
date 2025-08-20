import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' #  <-- 補上這行
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

# 裝置自動選擇（與訓練一致）
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

# 載入模型結構參數
config_path = SAVED_MODELS_DIR / 'best_model_config.json'
model_path = SAVED_MODELS_DIR / 'best_model.pth'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"錯誤：找不到模型設定檔 {config_path}。請先執行 dnn.py 或 dnn_v2.py 進行訓練。")
    exit()

class DNN(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        # 根據設定檔是來自 dnn.py 還是 dnn_v2.py (optuna) 來建立模型
        if 'n_layers' in config: # Optuna-based config
            layers = []
            in_features = input_dim
            for i in range(config['n_layers']):
                layers.append(nn.Linear(in_features, config[f'n_units_l{i}']))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config[f'dropout_l{i}']))
                in_features = config[f'n_units_l{i}']
            layers.append(nn.Linear(in_features, 1))
            self.net = nn.Sequential(*layers)
        else: # Manual config
            self.net = nn.Sequential(
                nn.Linear(input_dim, config['HIDDEN1']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_P']),
                nn.Linear(config['HIDDEN1'], config['HIDDEN2']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_P']),
                nn.Linear(config['HIDDEN2'], config['HIDDEN3']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_P']),
                nn.Linear(config['HIDDEN3'], config['HIDDEN4']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_P']),
                nn.Linear(config['HIDDEN4'], config['HIDDEN5']),
                nn.ReLU(),
                nn.Dropout(config['DROPOUT_P']),
                nn.Linear(config['HIDDEN5'], 1)
            )

    def forward(self, x):
        return self.net(x)

# 載入 test 資料
try:
    X_test = pd.read_csv(PROCESSED_DATA_DIR / 'test_preprocessed_v2.csv')
    train_cols_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_preprocessed_v2.csv', nrows=1)
    original_test_df = pd.read_csv(RAW_DATA_DIR / 'test.csv')
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請檢查 v2 版本的預處理資料和原始 test.csv 是否存在。")
    exit()

ids = original_test_df['Id']

# 讀取 train 的特徵欄位順序
train_cols = train_cols_df.columns.tolist()
feature_cols = [col for col in train_cols if col not in ['Id', 'SalePrice']]

# 只保留與 train 相同的特徵欄位，並依照順序排列
X = X_test[feature_cols]

X_tensor = torch.tensor(X.values.astype(np.float32)).to(DEVICE)

# 載入模型
input_dim = X.shape[1]
model = DNN(input_dim, config).to(DEVICE)

try:
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
except FileNotFoundError:
    print(f"錯誤：找不到模型檔案 {model_path}。請先執行 dnn.py 或 dnn_v2.py 進行訓練。")
    exit()

model.eval()

# 預測
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy().flatten()

# 檢查預測是否為 log scale (來自 dnn_v2.py)
if 'test_rmse' in config: # Heuristic: if it has test_rmse, it's likely from dnn_v2
    print("偵測到模型可能來自 v2 (log-transformed)，將預測結果進行 expm1 轉換。")
    preds = np.expm1(preds)

# 儲存結果
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
submission_path = SUBMISSIONS_DIR / 'submission.csv'
submission = pd.DataFrame({'Id': ids, 'SalePrice': preds})
submission.to_csv(submission_path, index=False, float_format='%.10f')
print(f'已儲存 submission.csv 至 {submission_path}')