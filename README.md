# 房屋價格預測專案 (Home Data ML Practice)

## 專案簡介
這個專案旨在利用機器學習技術，根據房屋的各種特徵來預測其銷售價格。我們使用了來自 [Kaggle 房屋價格預測競賽](https://www.kaggle.com/competitions/home-data-for-ml-course/overview) 的數據集，涵蓋了多種數據處理、特徵工程、模型訓練與評估的方法。

## 專案目標
*   進行全面的探索性資料分析 (EDA)，理解資料特性。
*   對原始資料進行清洗和前處理，處理缺失值、異常值和類別特徵。
*   建立並比較多種機器學習模型 (如線性迴歸、LightGBM、XGBoost、深度學習模型)。
*   透過模型調優和集成學習，提高預測準確性。
*   最終目標是建立一個穩健且高預測能力的房屋價格預測模型。

## 專案成果簡述
*   **模型應用與比較**：嘗試並比較了多種機器學習模型，展示不同方法在房價預測上的表現。
*   **資料洞察**：透過探索性資料分析 (EDA)，發掘了影響房屋價格的關鍵特徵，並於 `reports/figures/` 提供多樣化的視覺化圖表。
*   **流程完整**：涵蓋資料前處理、特徵工程、模型訓練、調優與預測等完整流程，具備實務參考價值。

## 技術棧 (Tech Stack)
*   **程式語言**：Python 3.13+
*   **資料處理與分析**：`pandas`, `numpy`
*   **機器學習框架**：`scikit-learn`, `lightgbm`, `xgboost`, `tensorflow` / `pytorch`
*   **資料視覺化**：`matplotlib`, `seaborn`
*   **模型調優**：`optuna`
*   **其他**：`jupyter` (用於 EDA)

## 專案結構

- data/  <!-- 存放原始與處理過的資料 -->
  - processed/  <!-- 處理後的資料 -->
  - raw/  <!-- 原始資料 -->
- notebooks/  <!-- Jupyter Notebooks，用於探索性資料分析 (EDA) -->
  - data_analysis.ipynb
- reports/  <!-- 報告與視覺化成果 -->
  - figures/  <!-- 各種分析圖表 (例如：缺失值熱力圖、特徵分佈圖、類別特徵箱形圖) -->
- saved_models/  <!-- 存放訓練好的模型檔案與配置 -->
- src/  <!-- 核心程式碼 -->
  - 01_preprocess_data_v1.py  <!-- 資料前處理腳本 -->
  - 01_preprocess_data_v2.py
  - 02_train_linear_regression.py  <!-- 模型訓練腳本 -->
  - 02_tune_dnn.py
  - 02_tune_lightgbm.py
  - 02_tune_xgboost.py
  - 03_make_ensemble_predictions.py  <!-- 集成預測腳本 -->
  - 03_predict_with_dnn.py
- submissions/  <!-- 最終預測結果提交檔案 -->
- .gitignore  <!-- Git 忽略檔案配置 -->
- README.md  <!-- 專案說明文件 (您正在閱讀的檔案) -->
- requirements.txt  <!-- 專案依賴套件列表 -->

## 如何運行 (How to Run)

請按照以下步驟在您的本地環境中運行此專案：

1.  **克隆專案倉庫**：
    ```bash
    git clone https://github.com/jcleealg/home-data-ml-practice.git
    cd Home_Data_ML_Practice
    ```

2.  **建立並啟用虛擬環境** (推薦)：
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .venv\Scripts\activate  # Windows
    ```

3.  **安裝專案依賴**：
    ```bash
    pip install -r requirements.txt
    ```

4.  **下載原始資料**：
    由於原始資料集較大，未直接包含在 Git 倉庫中。請從 [Kaggle 競賽頁面](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) 下載 `train.csv` 和 `test.csv`，並將它們放置在 `Home_Data_ML_Practice/data/raw/` 資料夾中。

5.  **執行專案流程**：
    您可以按照以下順序執行腳本，從資料前處理到模型訓練和預測：

    ```bash
    # 執行資料前處理 (選擇一個版本，例如 v2)
    python Home_Data_ML_Practice/src/01_preprocess_data_v2.py

    # 訓練並調優模型 (您可以選擇運行一個或多個)
    python Home_Data_ML_Practice/src/02_tune_lightgbm.py
    python Home_Data_ML_Practice/src/02_tune_xgboost.py
    python Home_Data_ML_Practice/src/02_tune_dnn.py
    python Home_Data_ML_Practice/src/02_train_linear_regression.py # 如果您有單獨的線性迴歸訓練腳本

    # 產生最終預測結果 (例如集成預測)
    python Home_Data_ML_Practice/src/03_make_ensemble_predictions.py
    # 或者使用單一模型進行預測
    python Home_Data_ML_Practice/src/03_predict_with_dnn.py
    ```
    最終的預測結果將會生成在 `Home_Data_ML_Practice/submissions/` 資料夾中。

## 視覺化成果
您可以在 `Home_Data_ML_Practice/reports/figures/` 資料夾中找到專案中生成的所有視覺化圖表，包括：
*   缺失值熱力圖 (`missing_heatmap.png`)
*   各數值特徵的分佈圖 (`num_distributions/`)
*   各類別特徵與銷售價格的箱形圖 (`cat_boxplots/`)

---