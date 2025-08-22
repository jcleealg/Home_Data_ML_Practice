import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
import time
import joblib
import json
from pathlib import Path

# --- 0. 設定路徑 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

# ===== 隨機種子設定 =====
SEED = 43
# ========================

# ===== Optuna 設定 =====
N_TRIALS = 100     # 嘗試優化的次數
N_SPLITS = 5        # K-Fold 交叉驗證的折數
# =======================

print("===== 開始執行 lightgbm_optuna.py =====")

# --- 1. 資料讀取與準備 ---
print(f"從 {PROCESSED_DATA_DIR} 和 {RAW_DATA_DIR} 讀取資料...")
try:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / 'train_preprocessed_v2.csv')
    test_df = pd.read_csv(PROCESSED_DATA_DIR / 'test_preprocessed_v2.csv')
    original_test_df = pd.read_csv(RAW_DATA_DIR / 'test.csv') # 讀取原始測試文件以獲取 ID
except FileNotFoundError as e:
    print(f"錯誤：找不到資料檔案 {e.filename}。請先執行 data_preprocess_v2.py。")
    exit()

# 準備訓練資料
X = train_df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
y = train_df['SalePrice'] # y 是 log1p 轉換後的值

# 準備測試資料
test_ids = original_test_df['Id'] # 從原始文件中獲取 ID
if 'Id' in test_df.columns:
    X_submission = test_df.drop('Id', axis=1)
else:
    X_submission = test_df.copy()

# 確保訓練集和測試集的欄位順序一致
X_submission = X_submission[X.columns]

# --- 2. Optuna Objective Function ---
def objective(trial):
    # 定義超參數搜索空間
    param = {
        'objective': 'regression_l1', # MAE, 對異常值較不敏感
        'metric': 'rmse', 
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_rmse = []

    # K-Fold 交叉驗證
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        oof_rmse.append(rmse)

    return np.mean(oof_rmse)

# --- 3. 執行 Optuna 優化 ---
print(f"開始執行 Optuna 優化，共 {N_TRIALS} 次試驗...")
start_optuna = time.time()
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
end_optuna = time.time()

print(f"Optuna 優化完成，耗時：{end_optuna - start_optuna:.2f} 秒")
print("最佳試驗:")
print(f"  - 平均 RMSE (log scale): {study.best_value:.5f}")
print("  - 最佳超參數:")
for key, value in study.best_params.items():
    print(f"    - {key}: {value}")

# --- 4. 使用最佳參數訓練最終模型 ---
print("\n使用最佳超參數在完整訓練集上訓練最終模型...")
best_params = study.best_params
best_params['random_state'] = SEED
best_params['n_jobs'] = -1
best_params['verbose'] = -1
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X, y)

# --- 5. 儲存模型與設定 ---
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = SAVED_MODELS_DIR / 'lightgbm_best_model.pkl'
config_path = SAVED_MODELS_DIR / 'lightgbm_best_config.json'

print(f"儲存最佳模型至 {model_path}...")
joblib.dump(final_model, model_path)

config_to_save = {
    'best_params': study.best_params,
    'best_value_log_rmse': study.best_value
}

print(f"儲存最佳設定至 {config_path}...")
with open(config_path, 'w') as f:
    json.dump(config_to_save, f, indent=4)

# --- 6. 產生提交檔案 ---
print("\n使用最佳模型預測測試集並產生提交檔案...")
predictions_log = final_model.predict(X_submission)

# 將預測值轉換回原始價格尺度
predictions_orig = np.expm1(predictions_log)

submission_df = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': predictions_orig
})

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
submission_filename = SUBMISSIONS_DIR / 'submission_lgbm_optuna.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\n提交檔案 '{submission_filename}' 已成功產生。")
print(submission_df.head())