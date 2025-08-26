import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import optuna
import warnings
from tqdm import tqdm
import datetime
import random
import os
import joblib

# --- 경로 설정 및 시드 고정 ---
# 사용자 환경에 맞게 경로를 수정해주세요.
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

# --- SMAPE 평가 함수 ---
def smape(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    smape_val = np.mean((2 * np.abs(pred - true)) / (np.abs(true) + np.abs(pred) + 1e-8)) * 100
    return smape_val

smape_scorer = make_scorer(smape, greater_is_better=False)
print("SMAPE 평가 함수를 정의했습니다.")


# --- 데이터 로딩 및 피처 엔지니어링 ---
print("\n1 & 2. 데이터 로딩, 전처리, 심화 피처 엔지니어링 수행...")
train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')

building_info_df.replace('-', '0', inplace=True)
# ... (building_info_df 전처리 코드는 동일)
building_info_df['태양광용량(kW)'] = building_info_df['태양광용량(kW)'].astype(float)
building_info_df['ESS저장용량(kWh)'] = building_info_df['ESS저장용량(kWh)'].astype(float)
building_info_df['PCS용량(kW)'] = building_info_df['PCS용량(kW)'].astype(float)
building_info_df = pd.get_dummies(building_info_df, columns=['건물유형'], drop_first=True)


train_df = pd.merge(train_df, building_info_df, on='건물번호')
test_df = pd.merge(test_df, building_info_df, on='건물번호')

train_df.sort_values(['건물번호', '일시'], inplace=True)
test_df.sort_values(['건물번호', '일시'], inplace=True)


def feature_engineering(df):
    df['일시'] = pd.to_datetime(df['일시'])
    
    # 1. 시간 피처
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # 2. 주기성 피처
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 3. 휴일 피처
    holidays = [pd.to_datetime('2024-06-06'), pd.to_datetime('2024-08-15')]
    df['holiday'] = df['일시'].dt.date.isin([d.date() for d in holidays]).astype(int)
    df['before_holiday'] = df['holiday'].shift(-1).fillna(0)
    df['after_holiday'] = df['holiday'].shift(1).fillna(0)
    
    # 4. 기상 관련 피처
    df['discomfort_index'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    
    # 건물별 이동 통계량
    for window in [3, 6, 12, 24]:
        df[f'temp_roll_mean_{window}h'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'temp_roll_std_{window}h'] = df.groupby('건물번호')['기온(°C)'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        df[f'humid_roll_mean_{window}h'] = df.groupby('건물번호')['습도(%)'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# 검증 세트 분리
train_df['일시'] = pd.to_datetime(train_df['일시'])
last_date = train_df['일시'].max()
validation_start_date = last_date - pd.Timedelta(days=4)
train_final_df = train_df[train_df['일시'] < validation_start_date].copy()
valid_final_df = train_df[train_df['일시'] >= validation_start_date].copy()
print(f"최종 훈련 데이터: {train_final_df.shape}, 최종 검증 데이터: {valid_final_df.shape}")
print("전처리 완료.")


# --- 모델링 및 예측 ---
print("\n3. 모델 학습 및 예측 시작...")
all_preds = []
N_TRIALS = 15 

for building_num in tqdm(range(1, 101), desc="건물별 모델링 진행"):
    # --- 1. 데이터 준비 ---
    train_building = train_final_df[train_final_df['건물번호'] == building_num].copy()
    valid_building = valid_final_df[valid_final_df['건물번호'] == building_num].copy()
    test_building = test_df[test_df['건물번호'] == building_num].copy()

    features = [col for col in train_building.columns if col not in ['num_date_time', '일시', '전력소비량(kWh)', '건물번호', '일조(hr)', '일사(MJ/m2)']]
    
    X_train = train_building[features]
    y_train = train_building['전력소비량(kWh)']
    X_valid = valid_building[features]
    y_valid = valid_building['전력소비량(kWh)'] # SMAPE 계산을 위해 원본 y_valid 유지
    X_test = test_building[features]

    # <<<--- 🌟 y 로그 변환 --- START
    y_train_log = np.log1p(y_train)
    # <<<--- 🌟 y 로그 변환 --- END

    # --- 2. 피처 스케일링 ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(w_path, f'scaler_building_{building_num}.pkl'))

    # --- 3. Optuna 튜닝 (XGBoost) ---
    def xgb_objective(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 1000,
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.7, 0.8, 0.9, 1.0]),
            'subsample': trial.suggest_categorical('subsample', [0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 9),
            'random_state': 42, 'n_jobs': -1,
        }
        model = xgb.XGBRegressor(**params)
        # <<<--- 🌟 로그 변환된 y로 학습
        model.fit(X_train_scaled, y_train_log, eval_set=[(X_valid_scaled, np.log1p(y_valid))], early_stopping_rounds=50, verbose=False)
        preds_log = model.predict(X_valid_scaled)
        # <<<--- 🌟 예측값을 원래 스케일로 복원하여 SMAPE 계산
        preds_original = np.expm1(preds_log)
        # 음수 예측 방지
        preds_original[preds_original < 0] = 0
        return smape(y_valid, preds_original)

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_params_xgb = study_xgb.best_params
    
    # --- 4. Optuna 튜닝 (CatBoost) ---
    def cat_objective(trial):
        params = {
            'iterations': 1000, 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 3, 9),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
            'bootstrap_type': 'Bernoulli', 'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'random_seed': 42, 'verbose': 0,
        }
        model = cb.CatBoostRegressor(**params)
        # <<<--- 🌟 로그 변환된 y로 학습
        model.fit(X_train_scaled, y_train_log, eval_set=[(X_valid_scaled, np.log1p(y_valid))], early_stopping_rounds=50, verbose=False)
        preds_log = model.predict(X_valid_scaled)
        # <<<--- 🌟 예측값을 원래 스케일로 복원하여 SMAPE 계산
        preds_original = np.expm1(preds_log)
        preds_original[preds_original < 0] = 0
        return smape(y_valid, preds_original)

    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(cat_objective, n_trials=N_TRIALS, show_progress_bar=False)
    best_params_cat = study_cat.best_params

    # --- 5. 최종 모델 학습 ---
    xgb_model = xgb.XGBRegressor(**best_params_xgb, n_estimators=2000, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train_log, eval_set=[(X_valid_scaled, np.log1p(y_valid))], early_stopping_rounds=100, verbose=False)
    
    cat_model = cb.CatBoostRegressor(**best_params_cat, iterations=2000, random_seed=42, verbose=0)
    cat_model.fit(X_train_scaled, y_train_log, eval_set=[(X_valid_scaled, np.log1p(y_valid))], early_stopping_rounds=100, verbose=False)

    # --- 6. 앙상블 가중치 계산 및 예측 ---
    xgb_valid_pred_log = xgb_model.predict(X_valid_scaled)
    cat_valid_pred_log = cat_model.predict(X_valid_scaled)
    
    # <<<--- 🌟 원래 스케일로 복원 후 SMAPE 계산
    xgb_valid_pred = np.expm1(xgb_valid_pred_log)
    cat_valid_pred = np.expm1(cat_valid_pred_log)
    xgb_valid_pred[xgb_valid_pred < 0] = 0
    cat_valid_pred[cat_valid_pred < 0] = 0

    smape_xgb = smape(y_valid, xgb_valid_pred)
    smape_cat = smape(y_valid, cat_valid_pred)

    weight_xgb = 1 / (smape_xgb + 1e-8)
    weight_cat = 1 / (smape_cat + 1e-8)
    total_weight = weight_xgb + weight_cat
    
    w1 = weight_xgb / total_weight
    w2 = weight_cat / total_weight

    # <<<--- 🌟 최종 예측값도 원래 스케일로 복원
    xgb_pred_log = xgb_model.predict(X_test_scaled)
    cat_pred_log = cat_model.predict(X_test_scaled)
    
    xgb_pred = np.expm1(xgb_pred_log)
    cat_pred = np.expm1(cat_pred_log)

    ensemble_pred = w1 * xgb_pred + w2 * cat_pred
    ensemble_pred[ensemble_pred < 0] = 0

    # --- 7. 결과 저장 ---
    building_preds = pd.DataFrame({'num_date_time': test_building['num_date_time'], 'answer': ensemble_pred})
    all_preds.append(building_preds)
    
    xgb_model.save_model(os.path.join(w_path, f'xgb_model_building_{building_num}.json'))
    cat_model.save_model(os.path.join(w_path, f'catboost_model_building_{building_num}.cbm'))

print("모델 학습 및 예측 완료. 최종 제출 파일 생성 중...")

# --- 최종 제출 파일 생성 ---
timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
file_name = f"submission_{timestamp}.csv"
file_path = os.path.join(s_path, file_name)

final_submission = pd.concat(all_preds, ignore_index=True)
final_submission.to_csv(file_path, index=False)

print(f"\n제출 파일이 '{file_path}'에 성공적으로 저장되었습니다.")
print("프로세스 종료.")