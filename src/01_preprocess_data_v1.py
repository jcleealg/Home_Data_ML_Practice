import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

print("===== 開始執行 data_preprocess.py (v1) =====")

# --- 1. 讀取資料 ---
try:
    train_df = pd.read_csv(RAW_DATA_DIR / 'train.csv')
    test_df = pd.read_csv(RAW_DATA_DIR / 'test.csv')
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請確認 train.csv 與 test.csv 是否位於 {RAW_DATA_DIR}")
    exit()

# --- 2. 合併資料以便統一處理 ---
# 儲存 ID 和目標變數以備後用
train_ids = train_df['Id']
test_ids = test_df['Id']
sale_price = train_df['SalePrice']

# 在合併前移除 ID 和 SalePrice
train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
test_df = test_df.drop('Id', axis=1)

all_data = pd.concat([train_df, test_df], ignore_index=True)
print(f"合併後的資料維度: {all_data.shape}")

# --- 3. 處理缺失值 ---
print("\n開始填補缺失值...")
# 數值型特徵用中位數填補
for col in all_data.select_dtypes(include=np.number).columns:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# 類別型特徵用 'None' 填補 (因為有些NA代表"沒有"的意思)
for col in all_data.select_dtypes(include=['object']).columns:
    all_data[col] = all_data[col].fillna('None')

print("所有缺失值填補完成。")

# --- 4. 特徵編碼 ---
print("\n開始進行特徵編碼...")
# 4.1 順序編碼
ordinal_maps = {
    'ExterQual':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'ExterCond':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtQual':     {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtCond':     {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtExposure': {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0},
    'BsmtFinType1': {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0},
    'BsmtFinType2': {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0},
    'HeatingQC':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'KitchenQual':  {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'FireplaceQu':  {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'GarageFinish': {'Fin':3, 'RFn':2, 'Unf':1, 'None':0},
    'GarageQual':   {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'GarageCond':   {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'PoolQC':       {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'None':0}
}

for col, mapping in ordinal_maps.items():
    all_data[col] = all_data[col].map(mapping)

print("順序編碼完成。")

# 4.2 One-Hot 編碼
all_data = pd.get_dummies(all_data, drop_first=True, dtype=int)
print(f"One-Hot 編碼後的資料維度: {all_data.shape}")

# --- 5. 分離並儲存資料 ---
print("\n分離並儲存處理好的資料...")

# 分離
X_train = all_data.iloc[:len(train_df)]
X_test = all_data.iloc[len(train_df):]

# 加回 ID 和 SalePrice
X_train['Id'] = train_ids
X_train['SalePrice'] = sale_price
X_test['Id'] = test_ids

# 確保 processed 目錄存在
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 儲存
train_output_path = PROCESSED_DATA_DIR / 'train_preprocessed_v1.csv'
test_output_path = PROCESSED_DATA_DIR / 'test_preprocessed_v1.csv'
X_train.to_csv(train_output_path, index=False)
X_test.to_csv(test_output_path, index=False)

print(f"\n===== 處理完成 (v1) =====")
print(f"已儲存處理後的訓練集: {train_output_path} (維度: {X_train.shape})")
print(f"已儲存處理後的測試集: {test_output_path} (維度: {X_test.shape})")


# --- 1. 讀取資料 ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請確認 train.csv 與 test.csv 是否存在。")
    exit()

# --- 2. 合併資料以便統一處理 ---
# 儲存 ID 和目標變數以備後用
train_ids = train_df['Id']
test_ids = test_df['Id']
sale_price = train_df['SalePrice']

# 在合併前移除 ID 和 SalePrice
train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
test_df = test_df.drop('Id', axis=1)

all_data = pd.concat([train_df, test_df], ignore_index=True)
print(f"合併後的資料維度: {all_data.shape}")

# --- 3. 處理缺失值 ---
print("\n開始填補缺失值...")
# 數值型特徵用中位數填補
for col in all_data.select_dtypes(include=np.number).columns:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# 類別型特徵用 'None' 填補 (因為有些NA代表"沒有"的意思)
for col in all_data.select_dtypes(include=['object']).columns:
    all_data[col] = all_data[col].fillna('None')

print("所有缺失值填補完成。")

# --- 4. 特徵編碼 ---
print("\n開始進行特徵編碼...")
# 4.1 順序編碼
ordinal_maps = {
    'ExterQual':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'ExterCond':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtQual':     {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtCond':     {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'BsmtExposure': {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'None':0},
    'BsmtFinType1': {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0},
    'BsmtFinType2': {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'None':0},
    'HeatingQC':    {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'KitchenQual':  {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'FireplaceQu':  {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'GarageFinish': {'Fin':3, 'RFn':2, 'Unf':1, 'None':0},
    'GarageQual':   {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'GarageCond':   {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0},
    'PoolQC':       {'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'None':0}
}

for col, mapping in ordinal_maps.items():
    all_data[col] = all_data[col].map(mapping)

print("順序編碼完成。")

# 4.2 One-Hot 編碼
all_data = pd.get_dummies(all_data, drop_first=True, dtype=int)
print(f"One-Hot 編碼後的資料維度: {all_data.shape}")

# --- 5. 分離並儲存資料 ---
print("\n分離並儲存處理好的資料...")

# 分離
X_train = all_data.iloc[:len(train_df)]
X_test = all_data.iloc[len(train_df):]

# 加回 ID 和 SalePrice
X_train['Id'] = train_ids
X_train['SalePrice'] = sale_price
X_test['Id'] = test_ids

# 儲存
train_output_path = 'train_preprocessed_v1.csv'
test_output_path = 'test_preprocessed_v1.csv'
X_train.to_csv(train_output_path, index=False)
X_test.to_csv(test_output_path, index=False)

print(f"\n===== 處理完成 (v1) =====")
print(f"已儲存處理後的訓練集: {train_output_path} (維度: {X_train.shape})")
print(f"已儲存處理後的測試集: {test_output_path} (維度: {X_test.shape})")