
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# ===== 控制參數 =====
# 是否對偏斜特徵及目標變數進行 log1p 轉換
DO_LOG_TRANSFORM = True
# 是否對所有數值特徵進行標準化
DO_STANDARDIZE = True
# 偏度閾值，絕對值大於此值的數值特徵會被轉換
SKEWNESS_THRESHOLD = 0.75
# ====================

print("===== 開始執行 data_preprocess_v2.py =====")

# 1. 讀取原始資料
try:
    train_df = pd.read_csv(RAW_DATA_DIR / 'train.csv')
    test_df = pd.read_csv(RAW_DATA_DIR / 'test.csv')
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請確認 train.csv 與 test.csv 是否位於 {RAW_DATA_DIR}")
    exit()

# 儲存 Test集的 Id
train_ids = train_df['Id']
test_ids = test_df['Id']

# 2. 目標變數轉換 (Log Transform)
if DO_LOG_TRANSFORM:
    print("對目標變數 SalePrice 進行 log1p 轉換...")
    train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# 3. 合併 train 和 test 以便進行一致的處理
#    在合併前先移除 train 的 SalePrice 和 Id
all_data = pd.concat((train_df.drop(['Id', 'SalePrice'], axis=1),
                       test_df.drop('Id', axis=1)),
                      ignore_index=True)

print(f"合併後的資料維度: {all_data.shape}")

# 4. 特徵工程 (Feature Engineering)
print("開始進行特徵工程...")
# 4.1 創造新的組合特徵
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
print("已建立新特徵: TotalSF, TotalBath, HouseAge, RemodAge")

# 4.2 處理偏斜的數值特徵
if DO_LOG_TRANSFORM:
    print(f"對偏度大於 {SKEWNESS_THRESHOLD} 的數值特徵進行 log1p 轉換...")
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: x.skew())
    skewed_feats = skewed_feats[abs(skewed_feats) > SKEWNESS_THRESHOLD].index
    
    if not skewed_feats.empty:
        print(f"找到 {len(skewed_feats)} 個偏斜特徵: {skewed_feats.tolist()}")
        all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    else:
        print("未找到符合條件的偏斜特徵。")

# 5. 缺失值填補 (Missing Value Imputation)
print("開始填補缺失值...")
# 類別型特徵用 'None' 填補
for col in all_data.select_dtypes(include=['object']).columns:
    all_data[col] = all_data[col].fillna('None')

# 數值型特徵用中位數填補
for col in all_data.select_dtypes(include=np.number).columns:
    all_data[col] = all_data[col].fillna(all_data[col].median())
print("缺失值填補完成。")

# 6. 類別特徵編碼 (One-Hot Encoding)
print("開始對類別特徵進行 One-Hot 編碼...")
all_data = pd.get_dummies(all_data, drop_first=True, dtype=int)
print(f"One-Hot 編碼後的資料維度: {all_data.shape}")

# 7. 數據標準化 (Standardization)
if DO_STANDARDIZE:
    print("開始對所有數值特徵進行標準化...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data)
    all_data = pd.DataFrame(scaled_data, index=all_data.index, columns=all_data.columns)
    print("標準化完成。")

# 8. 分離 train 和 test
print("分離處理好的 train 和 test 資料集...")
X_train = all_data.iloc[:len(train_df)]
X_test = all_data.iloc[len(train_df):]

# 將 Id 和 SalePrice 加回訓練集
X_train['Id'] = train_ids
X_train['SalePrice'] = train_df['SalePrice']

# 將 Id 加回測試集
X_test['Id'] = test_ids

# 9. 儲存結果
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
train_output_path = PROCESSED_DATA_DIR / 'train_preprocessed_v2.csv'
test_output_path = PROCESSED_DATA_DIR / 'test_preprocessed_v2.csv'
X_train.to_csv(train_output_path, index=False)
X_test.to_csv(test_output_path, index=False)

print("\n===== 處理完成 =====")
print(f"已儲存處理後的訓練集: {train_output_path} (維度: {X_train.shape})")
print(f"已儲存處理後的測試集: {test_output_path} (維度: {X_test.shape})")
print(f"參數設定: DO_LOG_TRANSFORM={DO_LOG_TRANSFORM}, DO_STANDARDIZE={DO_STANDARDIZE}")
