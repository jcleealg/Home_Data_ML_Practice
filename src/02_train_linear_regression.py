import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_DIR = DATA_DIR / 'raw'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

# --- 1. 讀取資料 ---
print("讀取前處理後的資料 (v1)...")
try:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_preprocessed_v1.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_preprocessed_v1.csv')
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請確認檔案位於 {PROCESSED_DATA_DIR}")
    exit()

# --- 2. 準備資料 ---
y_train = train_df['SalePrice']
X_train = train_df.drop(['SalePrice', 'Id'], axis=1)
X_test = test_df.drop(['Id'], axis=1)

# 從原始 test.csv 讀取 Id
try:
    test_ids = pd.read_csv(RAW_DATA_DIR / 'test.csv')['Id']
except FileNotFoundError:
    print(f"錯誤：找不到原始 test.csv。請確認檔案位於 {RAW_DATA_DIR}")
    exit()


print(f"訓練集維度: {X_train.shape}")
print(f"測試集維度: {X_test.shape}")

# --- 3. 訓練多元線性迴歸模型 ---
print("\n建立並訓練 LinearRegression 模型...")
model = LinearRegression()
model.fit(X_train, y_train)
print("模型訓練完成。")

# --- 4. 預測測試集並產生提交檔案 ---
print("\n開始預測測試集...")
predictions = model.predict(X_test)

# 建立提交 DataFrame
submission_df = pd.DataFrame({
    'Id': test_ids.astype(int),
    'SalePrice': predictions
})

# 確保 submissions 目錄存在
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
submission_filename = SUBMISSIONS_DIR / 'submission_mlr.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\n提交檔案 '{submission_filename}' 已成功產生。")
print(submission_df.head())
