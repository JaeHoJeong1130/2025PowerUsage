import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from tqdm import tqdm
import datetime
import random
import os

# --- 경로 설정 및 시드 고정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
w_path = "/home/jjh/Project/_data/dacon/electro/wei/"
s_path = "/home/jjh/Project/_data/dacon/electro/sub/"
os.makedirs(w_path, exist_ok=True)
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)
print("랜덤 시드를 42로 고정했습니다.")

# --- 데이터 로딩, 전처리, 피처 엔지니어링 (기존과 동일) ---
print("\n1 & 2. 데이터 로딩, 전처리, 피처 엔지니어링 수행...")
train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')

building_info_df.replace('-', '0', inplace=True)
building_info_df['태양광용량(kW)'] = building_info_df['태양광용량(kW)'].astype(float)
building_info_df['ESS저장용량(kWh)'] = building_info_df['ESS저장용량(kWh)'].astype(float)
building_info_df['PCS용량(kW)'] = building_info_df['PCS용량(kW)'].astype(float)
building_info_df = pd.get_dummies(building_info_df, columns=['건물유형'], drop_first=True)
train_df = pd.merge(train_df, building_info_df, on='건물번호')
test_df = pd.merge(test_df, building_info_df, on='건물번호')

def feature_engineering(df):
    df['일시'] = pd.to_datetime(df['일시'])
    df['month'] = df['일시'].dt.month; df['day'] = df['일시'].dt.day; df['hour'] = df['일시'].dt.hour
    df['dayofweek'] = df['일시'].dt.dayofweek
    df['MMDDHH'] = df['month'].astype(str).str.zfill(2) + df['day'].astype(str).str.zfill(2) + df['hour'].astype(str).str.zfill(2)
    df['MMDDHH'] = df['MMDDHH'].astype(int)
    holidays = [pd.to_datetime('2024-06-06'), pd.to_datetime('2024-08-15')]
    df['holiday'] = df['일시'].dt.date.isin([d.date() for d in holidays]).astype(int)
    df['discomfort_index'] = 9/5 * df['기온(°C)'] - 0.55 * (1 - df['습도(%)']/100) * (9/5 * df['기온(°C)'] - 26) + 32
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df
train_df = feature_engineering(train_df); test_df = feature_engineering(test_df)

train_df['일시'] = pd.to_datetime(train_df['일시'])
last_date = train_df['일시'].max()
validation_start_date = last_date - pd.Timedelta(days=7)
train_final_df = train_df[train_df['일시'] < validation_start_date].copy()
valid_final_df = train_df[train_df['일시'] >= validation_start_date].copy()
print(f"최종 훈련 데이터: {train_final_df.shape}, 최종 검증 데이터: {valid_final_df.shape}")

# ===================================================================
# [수정] 2.5. '일조'/'일사' 예측 모델 고도화
# ===================================================================
print("\n2.5. Test 데이터의 '일조', '일사' 피처 예측 시작 (고도화 버전)...")
sun_features = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', 'sin_hour', 'cos_hour', 'dayofweek', 'month']
sun_targets = ['일조(hr)', '일사(MJ/m2)']

# '일조'/'일사' 예측을 위한 하이퍼파라미터 그리드
sun_param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7],
    'n_estimators': [600, 700],
    'colsample_bytree': [0.8, 0.9]
}
# - '일조(hr)' 예측 모델 HPO 수행...
# - 최적 파라미터 발견: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 700}
# - '일사(MJ/m2)' 예측 모델 HPO 수행...
# - 최적 파라미터 발견: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 700}

train_sun = train_final_df.dropna(subset=sun_targets).copy()
for target in sun_targets:
    print(f"  - '{target}' 예측 모델 HPO 수행...")
    X_sun_train = train_sun[sun_features]
    y_sun_train = train_sun[target]
    
    # GridSearchCV를 사용하여 최적의 파라미터 탐색
    sun_model_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid_search_sun = GridSearchCV(estimator=sun_model_base, param_grid=sun_param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search_sun.fit(X_sun_train, y_sun_train)
    
    print(f"  - 최적 파라미터 발견: {grid_search_sun.best_params_}")
    
    # 최적의 모델로 예측 수행
    best_sun_model = grid_search_sun.best_estimator_
    
    for df in [test_df, valid_final_df]:
        X_sun_pred = df[sun_features]
        sun_predictions = best_sun_model.predict(X_sun_pred)
        sun_predictions[sun_predictions < 0] = 0
        df[target] = sun_predictions
print("전처리 완료.")
# ===================================================================
# train_final_df.to_csv(path+'train_final_df.csv', index=False)
# test_df.to_csv(path+'test_df.csv', index=False)
# valid_final_df.to_csv(path+'valid_final_df.csv', index=False)
# exit()

# --- 모델 학습, HPO, 앙상블 로직 ---
print("\n3. 건물별 3개 모델 HPO, 피처 선택 및 앙상블 예측 시작...")
# (이하 코드는 이전과 동일)
scaler = StandardScaler()
final_predictions = pd.DataFrame()
validation_predictions = pd.DataFrame()

param_grids = {
    'xgb': {'learning_rate': [0.02, 0.05], 'max_depth': [7, 9]},
    'catboost': {'learning_rate': [0.05, 0.1], 'depth': [6, 8]}
}
lgbm_params = {
    'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 1000,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 1, 'verbose': -1, 'n_jobs': -1, 'seed': 42
}
initial_features = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    'month', 'day', 'hour', 'dayofweek', 'MMDDHH', 'holiday',
    'discomfort_index', 'sin_hour', 'cos_hour'
]
initial_features += [col for col in building_info_df.columns if '건물유형_' in col]

for building_num in tqdm(range(1, 101), desc="전체 건물 학습 진행"):
    train_building = train_final_df[train_final_df['건물번호'] == building_num].copy()
    test_building = test_df[test_df['건물번호'] == building_num].copy()
    valid_building = valid_final_df[valid_final_df['건물번호'] == building_num].copy()

    train_building.dropna(axis=1, inplace=True)
    current_features = [f for f in initial_features if f in train_building.columns]
    
    X = train_building[current_features]; y = train_building['전력소비량(kWh)']
    current_features_in_X = list(X.columns)
    y_true_valid = valid_building['전력소비량(kWh)'].values
    test_building = test_building[current_features_in_X]
    valid_building = valid_building[current_features_in_X]
    
    if X.empty:
        # ... (생략) ...
        continue

    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(test_building)
    X_valid_scaled = scaler.transform(valid_building)
    y_log = np.log1p(y)
    
    test_preds_dict = {}; valid_preds_dict = {}

    for model_name in ['xgb', 'lgbm', 'catboost']:
        if model_name == 'lgbm':
            first_pass_model = lgb.LGBMRegressor(**lgbm_params); first_pass_model.fit(X_scaled, y_log)
        else:
            base_params = {'random_state': 42, 'n_estimators': 200}
            model = xgb.XGBRegressor(objective='reg:squarederror', **base_params) if model_name == 'xgb' else cb.CatBoostRegressor(verbose=0, **base_params)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='neg_mean_absolute_error', cv=3, verbose=0, n_jobs=-1)
            grid_search.fit(X_scaled, y_log)
            best_params = grid_search.best_params_
            first_pass_params = {**best_params, **base_params}
            first_pass_model = xgb.XGBRegressor(objective='reg:squarederror', **first_pass_params) if model_name == 'xgb' else cb.CatBoostRegressor(verbose=0, **first_pass_params)
            first_pass_model.fit(X_scaled, y_log)
        
        importances = first_pass_model.feature_importances_
        important_features_indices = np.where(importances > 0)[0]
        X_scaled_important = X_scaled[:, important_features_indices]
        X_test_scaled_important = X_test_scaled[:, important_features_indices]
        X_valid_scaled_important = X_valid_scaled[:, important_features_indices]
        X_train, X_val, y_train, y_val = train_test_split(X_scaled_important, y_log, test_size=0.15, random_state=42)
        
        if model_name == 'xgb':
            final_params = {**grid_search.best_params_, 'n_estimators': 1000}
            es_xgb = xgb.callback.EarlyStopping(rounds=50, metric_name='mae', data_name='validation_0', save_best=True)
            final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **final_params, eval_metric='mae', callbacks=[es_xgb])
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            test_preds_log = final_model.predict(X_test_scaled_important, iteration_range=(0, final_model.best_iteration))
            valid_preds_log = final_model.predict(X_valid_scaled_important, iteration_range=(0, final_model.best_iteration))
        elif model_name == 'lgbm':
            final_model = lgb.LGBMRegressor(**lgbm_params)
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae', callbacks=[lgb.early_stopping(50, verbose=False)])
            test_preds_log = final_model.predict(X_test_scaled_important, num_iteration=final_model.best_iteration_)
            valid_preds_log = final_model.predict(X_valid_scaled_important, num_iteration=final_model.best_iteration_)
        else:
            final_params = {**grid_search.best_params_, 'n_estimators': 1000}
            final_model = cb.CatBoostRegressor(random_state=42, **final_params, verbose=0)
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, use_best_model=True)
            test_preds_log = final_model.predict(X_test_scaled_important)
            valid_preds_log = final_model.predict(X_valid_scaled_important)
        
        test_preds_dict[model_name] = np.expm1(test_preds_log)
        valid_preds_dict[model_name] = np.expm1(valid_preds_log)

    # --- 가중치 앙상블 ---
    model_maes = {name: mean_absolute_error(y_true_valid, preds) for name, preds in valid_preds_dict.items()}
    epsilon = 1e-6
    inverse_maes = {name: 1 / (mae + epsilon) for name, mae in model_maes.items()}
    total_inverse_mae = sum(inverse_maes.values())
    model_weights = {name: inv_mae / total_inverse_mae for name, inv_mae in inverse_maes.items()}

    ensemble_test_preds = (test_preds_dict['xgb'] * model_weights['xgb'] +
                           test_preds_dict['lgbm'] * model_weights['lgbm'] +
                           test_preds_dict['catboost'] * model_weights['catboost'])
    ensemble_test_preds[ensemble_test_preds < 0] = 0
    
    ensemble_valid_preds = (valid_preds_dict['xgb'] * model_weights['xgb'] +
                            valid_preds_dict['lgbm'] * model_weights['lgbm'] +
                            valid_preds_dict['catboost'] * model_weights['catboost'])
    ensemble_valid_preds[ensemble_valid_preds < 0] = 0
    
    building_submission = pd.DataFrame({'answer': ensemble_test_preds})
    final_predictions = pd.concat([final_predictions, building_submission], ignore_index=True)
    building_validation = pd.DataFrame({'answer': ensemble_valid_preds})
    validation_predictions = pd.concat([validation_predictions, building_validation], ignore_index=True)


print("모든 건물 모델 HPO 및 앙상블 예측 완료.")

# --- 4. 최종 스코어 계산 및 제출 파일 생성 ---
print("\n4. 최종 스코어 계산 및 제출 파일 생성 시작...")
true_values = valid_final_df['전력소비량(kWh)'].values
predicted_values = validation_predictions['answer'].values
final_mae_score = mean_absolute_error(true_values, predicted_values)

print("="*60)
print(f"🏆 최종 로컬 검증 스코어 (MAE): {final_mae_score:.4f}")
print("="*60)

sample_submission_df['answer'] = final_predictions['answer']
timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
final_filename = s_path + f'submission_{timestamp}_{final_mae_score:.4f}.csv'
sample_submission_df.to_csv(final_filename, index=False)
print(f"제출 파일 '{final_filename}' 생성이 완료되었습니다.")