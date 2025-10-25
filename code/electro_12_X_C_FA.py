import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error
import warnings
from tqdm import tqdm
import datetime
import random
import os
import joblib

# --- 경로 설정 및 시드 고정 ---
# !!! 사용 전 경로를 실제 환경에 맞게 수정해주세요 !!!
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
print("랜덤 시드를 42로 고정했습니다.")

# SMAPE 평가 산식 정의
def smape(true, pred):
    return np.mean((2 * np.abs(pred - true)) / (np.abs(true) + np.abs(pred))) * 100

# --- 1. 데이터 로딩 및 전처리 ---
print("\n[단계 1/5] 데이터 로딩 및 전처리를 시작합니다.")
train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')

# building_info 전처리
building_info_df.replace('-', '0', inplace=True)
building_info_df['태양광용량(kW)'] = building_info_df['태양광용량(kW)'].astype(float)
building_info_df['ESS저장용량(kWh)'] = building_info_df['ESS저장용량(kWh)'].astype(float)
building_info_df['PCS용량(kW)'] = building_info_df['PCS용량(kW)'].astype(float)
building_info_df = pd.get_dummies(building_info_df, columns=['건물유형'], drop_first=True)

# --- 2. 피처 엔지니어링 ---
print("\n[단계 2/5] 피처 엔지니어링을 수행합니다.")

# 요청하신 건물 그룹 피처 추가
building_groups_str = """
87 80
50
30 17 63 18 31 53 49 51 52 43 48 64 29 76 78 40 60
54 84
56 38 36
39 59 92 73 16 15
9 10
71
46 95 94 93
98
75 58 91 19
77
90 89
72
82 55
41 88 68 83 12 11 13
66
22 23 21
97
34 33 37 1 27 3 2 5 6 7 4 96 67 86 35 47
85
57
8
81 61 74
28 14 69
24
44 100 26 45 70 20
62 25
32 42 79 65 99
"""
groups = building_groups_str.strip().split('\n')
building_group_map = {}
for i, group in enumerate(groups):
    buildings = map(int, group.split())
    for building_num in buildings:
        building_group_map[building_num] = i

building_info_df['building_group'] = building_info_df['건물번호'].map(building_group_map)
print("새로운 피처 'building_group'을 추가했습니다.")

# 데이터 병합
train_df = pd.merge(train_df, building_info_df, on='건물번호')
test_df = pd.merge(test_df, building_info_df, on='건물번호')

# 기본 피처 엔지니어링 함수
def feature_engineering(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['MMDDHH'] = df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2) + df['hour'].astype(str).str.zfill(2)
    df['MMDDHH'] = df['MMDDHH'].astype(int)
    holidays = [pd.to_datetime('2024-06-06'), pd.to_datetime('2024-08-15')]
    df['holiday'] = df['일시'].dt.date.isin([d.date() for d in holidays]).astype(int)
    df['discomfort_index'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_weekend'] = (df['일시'].dt.dayofweek >= 5).astype(int)
    df['day_of_year'] = df['일시'].dt.dayofyear
    df['rolling_3h_temp_mean'] = df.groupby('건물번호')['기온(°C)'].rolling(window=3, min_periods=1).mean().values
    df['rolling_24h_temp_mean'] = df.groupby('건물번호')['기온(°C)'].rolling(window=24, min_periods=1).mean().values
    df['rolling_24h_temp_max'] = df.groupby('건물번호')['기온(°C)'].rolling(window=24, min_periods=1).max().values
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
print("기본 피처 엔지니어링을 완료했습니다.")

# 전처리 완료된 데이터 저장
preprocessed_train_path = os.path.join(path, "preprocessed_train.csv")
preprocessed_test_path = os.path.join(path, "preprocessed_test.csv")
train_df.to_csv(preprocessed_train_path, index=False, encoding='utf-8-sig')
test_df.to_csv(preprocessed_test_path, index=False, encoding='utf-8-sig')
print(f"전처리된 훈련 데이터를 '{preprocessed_train_path}'에 저장했습니다.")
print(f"전처리된 테스트 데이터를 '{preprocessed_test_path}'에 저장했습니다.")


# --- 3. 모델 학습 준비 ---
print("\n[단계 3/5] 모델 학습을 준비합니다.")
# 피처 및 타겟 정의
features = [col for col in train_df.columns if col not in ['num_date_time', '일시', '전력소비량(kWh)', '일조(hr)', '일사(MJ/m2)']]
# 일조, 일사 피처는 test 데이터에 없으므로 제외
target = '전력소비량(kWh)'

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

categorical_features = ['건물번호', 'dayofweek', 'building_group']
for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# --- 4. Optuna를 이용한 모델 튜닝 및 학습 ---
print("\n[단계 4/5] Optuna를 사용하여 모델 튜닝 및 학습을 시작합니다. (시간이 다소 소요될 수 있습니다)")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost Objective
def xgb_objective(trial):
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'seed': 42, 'n_jobs': -1, 'tree_method': 'hist', 'device': 'cuda'
    }
    
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_fold_scaled, y_train_fold,
                  eval_set=[(X_val_fold_scaled, y_val_fold)],
                  early_stopping_rounds=50, verbose=False)
        
        preds = model.predict(X_val_fold_scaled)
        preds[preds < 0] = 0 # 전력량은 음수가 될 수 없음
        scores.append(smape(y_val_fold, preds))
        
    return np.mean(scores)

# CatBoost Objective
def cat_objective(trial):
    params = {
        'iterations': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'loss_function': 'RMSE', 'eval_metric': 'SMAPE',
        'random_seed': 42, 'verbose': 0, 'task_type': 'GPU'
    }

    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # CatBoost는 자체적으로 스케일링 및 범주형 피처 처리를 하므로 스케일러 불필요
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  early_stopping_rounds=50,
                  cat_features=categorical_features,
                  verbose=False)

        preds = model.predict(X_val_fold)
        preds[preds < 0] = 0
        scores.append(smape(y_val_fold, preds))

    return np.mean(scores)

# Optuna Study 실행
print("XGBoost 튜닝 중...")
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, n_trials=30) # n_trials: 시도 횟수. 늘릴수록 성능이 좋아질 수 있으나 시간이 오래 걸림
print(f"XGBoost 최적 SMAPE: {xgb_study.best_value:.4f}")
print("XGBoost 최적 파라미터:", xgb_study.best_params)

print("\nCatBoost 튜닝 중...")
cat_study = optuna.create_study(direction='minimize')
cat_study.optimize(cat_objective, n_trials=30)
print(f"CatBoost 최적 SMAPE: {cat_study.best_value:.4f}")
print("CatBoost 최적 파라미터:", cat_study.best_params)

# --- 5. 최종 모델 학습 및 앙상블 ---
print("\n[단계 5/5] 최종 모델 학습, 앙상블 및 예측을 수행합니다.")

# 최적 파라미터로 최종 모델 학습 및 OOF 예측 생성
xgb_best_params = {**xgb_study.best_params, 'n_estimators': 1000, 'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'seed': 42, 'n_jobs': -1, 'tree_method': 'hist', 'device': 'cuda'}
cat_best_params = {**cat_study.best_params, 'iterations': 1000, 'loss_function': 'RMSE', 'eval_metric': 'SMAPE', 'random_seed': 42, 'verbose': 0, 'task_type': 'GPU'}

oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
predictions_xgb = np.zeros(len(X_test))
predictions_cat = np.zeros(len(X_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(w_path, 'scaler.pkl')) # 스케일러 저장

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"--- Fold {fold+1} ---")
    X_train_fold_scaled, X_val_fold_scaled = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    X_train_fold_cat, X_val_fold_cat = X_train.iloc[train_idx], X_train.iloc[val_idx]

    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_best_params)
    xgb_model.fit(X_train_fold_scaled, y_train_fold, eval_set=[(X_val_fold_scaled, y_val_fold)], early_stopping_rounds=50, verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_fold_scaled)
    predictions_xgb += xgb_model.predict(X_test_scaled) / kf.n_splits
    xgb_model.save_model(os.path.join(w_path, f'xgb_model_fold{fold+1}.json'))

    # CatBoost
    cat_model = cb.CatBoostRegressor(**cat_best_params)
    cat_model.fit(X_train_fold_cat, y_train_fold, eval_set=[(X_val_fold_cat, y_val_fold)], early_stopping_rounds=50, cat_features=categorical_features, verbose=False)
    oof_cat[val_idx] = cat_model.predict(X_val_fold_cat)
    predictions_cat += cat_model.predict(X_test) / kf.n_splits
    cat_model.save_model(os.path.join(w_path, f'cat_model_fold{fold+1}.cbm'))

oof_xgb[oof_xgb < 0] = 0
oof_cat[oof_cat < 0] = 0

# OOF 점수로 앙상블 가중치 계산
smape_xgb = smape(y_train, oof_xgb)
smape_cat = smape(y_train, oof_cat)
print(f"\nOOF SMAPE (XGBoost): {smape_xgb:.4f}")
print(f"OOF SMAPE (CatBoost): {smape_cat:.4f}")

# SMAPE의 역수를 가중치로 사용 (SMAPE가 낮을수록 가중치가 높아짐)
weight_xgb = (1 / smape_xgb) / ((1 / smape_xgb) + (1 / smape_cat))
weight_cat = 1 - weight_xgb
print(f"앙상블 가중치 -> XGB: {weight_xgb:.4f}, Cat: {weight_cat:.4f}")

# 최종 예측
predictions_xgb[predictions_xgb < 0] = 0
predictions_cat[predictions_cat < 0] = 0
final_predictions = (predictions_xgb * weight_xgb) + (predictions_cat * weight_cat)

timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")

# 제출 파일 생성
sample_submission_df['answer'] = final_predictions

file_name = f"submission_{timestamp}.csv"

submission_file_path = os.path.join(s_path, file_name)
sample_submission_df.to_csv(submission_file_path, index=False)

print(f"\n✨ 작업 완료! 최종 제출 파일이 '{submission_file_path}'에 저장되었습니다.")

"""
OOF SMAPE (XGBoost): 5.7825
OOF SMAPE (CatBoost): 7.5858
앙상블 가중치 -> XGB: 0.5674, Cat: 0.4326

✨ 작업 완료! 최종 제출 파일이 '/home/jjh/Project/_data/dacon/electro/sub/submission_0722212854.csv'에 저장되었습니다.
"""