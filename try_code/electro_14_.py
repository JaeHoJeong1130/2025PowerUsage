import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
import datetime
import random
import os
import joblib

# --- 경로 설정 및 시드 고정 ---
# !!! 사용 전 경로를 실제 환경에 맞게 수정해주세요 !!!
# 현재 파일들이 업로드된 가상 환경을 경로로 설정합니다.
path = "/home/jjh/Project/_data/dacon/electro/"
w_path = "/home/jjh/Project/_data/dacon/electro/wei/"
s_path = "/home/jjh/Project/_data/dacon/electro/sub/"

os.makedirs(w_path, exist_ok=True)
os.makedirs(s_path, exist_ok=True)
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
print("✅ 랜덤 시드를 42로 고정했습니다.")

# SMAPE 평가 산식 정의
def smape(true, pred):
    epsilon = 1e-10
    return np.mean((2 * np.abs(pred - true)) / (np.abs(true) + np.abs(pred) + epsilon)) * 100

# --- 1. 데이터 로딩 및 전처리 (제공된 코드 활용) ---
print("\n[단계 1/5] 데이터 로딩 및 전처리를 시작합니다...")
try:
    train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
    building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
    sample_submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')
except FileNotFoundError:
    print("오류: CSV 파일을 찾을 수 없습니다. 'path' 변수가 올바르게 설정되었는지 확인해주세요.")
    # 필요한 경우 여기서 스크립트 실행을 중단할 수 있습니다.
    exit()


building_info_df.replace('-', '0', inplace=True)
building_info_df['태양광용량(kW)'] = building_info_df['태양광용량(kW)'].astype(float)
building_info_df['ESS저장용량(kWh)'] = building_info_df['ESS저장용량(kWh)'].astype(float)
building_info_df['PCS용량(kW)'] = building_info_df['PCS용량(kW)'].astype(float)
building_info_df = pd.get_dummies(building_info_df, columns=['건물유형'], drop_first=True, prefix='건물유형')

# --- 2. 피처 엔지니어링 (제공된 코드 활용) ---
print("[단계 2/5] 피처 엔지니어링을 수행합니다...")
building_groups_str = """
87 80;50;30 17 63 18 31 53 49 51 52 43 48 64 29 76 78 40 60;54 84;56 38 36;39 59 92 73 16 15;9 10;71;46 95 94 93;98;75 58 91 19;77;90 89;72;82 55;41 88 68 83 12 11 13;66;22 23 21;97;34 33 37 1 27 3 2 5 6 7 4 96 67 86 35 47;85;57;8;81 61 74;28 14 69;24;44 100 26 45 70 20;62 25;32 42 79 65 99
"""
groups = building_groups_str.strip().replace('\n', ';').split(';')
building_group_map = {}
for i, group in enumerate(groups):
    buildings = map(int, group.split())
    for building_num in buildings:
        building_group_map[building_num] = i
building_info_df['building_group'] = building_info_df['건물번호'].map(building_group_map)

train_df = pd.merge(train_df, building_info_df, on='건물번호')
test_df = pd.merge(test_df, building_info_df, on='건물번호')

def feature_engineering(df):
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek # 0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일
    df['MMDDHH'] = df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2) + df['hour'].astype(str).str.zfill(2)
    df['MMDDHH'] = df['MMDDHH'].astype(int)
    holidays = [pd.to_datetime('2024-06-06'), pd.to_datetime('2024-08-15')]
    df['holiday'] = df['일시'].dt.date.isin([d.date() for d in holidays]).astype(int)
    df['discomfort_index'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['일시'].dt.dayofweek >= 5).astype(int)
    df['day_of_year'] = df['일시'].dt.dayofyear
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
print("✅ 피처 엔지니어링 완료!")

# --- 3. 모델 학습 및 예측 ---
print("[단계 3/5] 건물별 모델 학습 및 예측을 시작합니다...")

# Public Score 기간(8/25~27)은 일,월,화 요일
target_days = [6, 0, 1] # 6:일, 0:월, 1:화
train_filtered = train_df[train_df['dayofweek'].isin(target_days)]

features = [col for col in test_df.columns if col not in ['num_date_time', '건물번호', '일시']]
all_building_preds = []

for building_num in tqdm(range(1, 101), desc="건물별 예측 진행 중"):
    
    # --- 데이터 준비 ---
    train_building = train_filtered[train_filtered['건물번호'] == building_num].copy()
    test_building = test_df[test_df['건물번호'] == building_num].copy()

    X_train = train_building[features]
    y_train = train_building['전력소비량(kWh)']
    X_test = test_building[features]
    
    # 스케일러 적용
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

    # --- Optuna를 이용한 하이퍼파라미터 튜닝 ---
    def objective(trial, model_name, X, y):
        if model_name == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            model_class = xgb.XGBRegressor
        else: # cat
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_state': 42,
                'verbose': 0
            }
            model_class = cb.CatBoostRegressor

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = model_class(**params, early_stopping_rounds=50,)
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            preds = model.predict(X_val_fold)
            scores.append(smape(y_val_fold, preds))
        
        return np.mean(scores)

    # XGBoost 튜닝
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: objective(trial, 'xgb', X_train_scaled, y_train), n_trials=20, n_jobs=-1)
    best_params_xgb = study_xgb.best_params
    
    # CatBoost 튜닝
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(lambda trial: objective(trial, 'cat', X_train_scaled, y_train), n_trials=20, n_jobs=-1)
    best_params_cat = study_cat.best_params

    # --- K-Fold 교차검증 기반 모델 학습 및 앙상블 ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds_xgb = np.zeros(len(X_train_scaled))
    test_preds_xgb = np.zeros(len(X_test_scaled))
    oof_preds_cat = np.zeros(len(X_train_scaled))
    test_preds_cat = np.zeros(len(X_test_scaled))

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # XGBoost
        model_xgb = xgb.XGBRegressor(**best_params_xgb, random_state=42)
        model_xgb.fit(X_train_fold, y_train_fold)
        oof_preds_xgb[val_idx] = model_xgb.predict(X_val_fold)
        test_preds_xgb += model_xgb.predict(X_test_scaled) / kf.n_splits

        # CatBoost
        model_cat = cb.CatBoostRegressor(**best_params_cat, random_state=42, verbose=0)
        model_cat.fit(X_train_fold, y_train_fold)
        oof_preds_cat[val_idx] = model_cat.predict(X_val_fold)
        test_preds_cat += model_cat.predict(X_test_scaled) / kf.n_splits

    # SMAPE 기반 가중치 계산
    smape_xgb = smape(y_train, oof_preds_xgb)
    smape_cat = smape(y_train, oof_preds_cat)
    
    w_cat = smape_xgb / (smape_xgb + smape_cat)
    w_xgb = smape_cat / (smape_xgb + smape_cat)
    
    # 가중치 저장
    weights = {'w_xgb': w_xgb, 'w_cat': w_cat, 'smape_xgb': smape_xgb, 'smape_cat': smape_cat}
    joblib.dump(weights, f'{w_path}building_{building_num}_weights.pkl')

    # 최종 예측
    final_preds = w_xgb * test_preds_xgb + w_cat * test_preds_cat
    final_preds[final_preds < 0] = 0 # 전력량은 음수가 될 수 없음

    building_pred_df = pd.DataFrame({'num_date_time': test_building['num_date_time'], 'answer': final_preds})
    all_building_preds.append(building_pred_df)

print("✅ 모든 건물의 예측이 완료되었습니다.")

# --- 4. 최종 제출 파일 생성 ---
print("[단계 4/5] 최종 제출 파일을 생성합니다...")
final_submission = pd.concat(all_building_preds, ignore_index=True)
final_submission = final_submission.sort_values(by=['num_date_time']).reset_index(drop=True)

# --- 5. 파일 저장 ---
final_submission_path = f"{s_path}final_submission.csv"
final_submission.to_csv(final_submission_path, index=False)

print(f"✅ 최종 제출 파일이 '{final_submission_path}' 경로에 저장되었습니다.")
print("[단계 5/5] 모든 작업이 성공적으로 완료되었습니다. 🎉")