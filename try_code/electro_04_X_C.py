import pandas as pd
import numpy as np
import xgboost as xgb
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

sun_features = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', 'sin_hour', 'cos_hour', 'dayofweek', 'month']
sun_targets = ['일조(hr)', '일사(MJ/m2)']
train_sun = train_final_df.dropna(subset=sun_targets).copy()
for target in sun_targets:
    X_sun_train = train_sun[sun_features]; y_sun_train = train_sun[target]
    sun_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    sun_model.fit(X_sun_train, y_sun_train)
    for df in [test_df, valid_final_df]:
        X_sun_pred = df[sun_features]
        sun_predictions = sun_model.predict(X_sun_pred)
        sun_predictions[sun_predictions < 0] = 0
        df[target] = sun_predictions
print("전처리 완료.")

# --- 모델 학습, HPO, 앙상블 로직 ---
print("\n3. 건물별 2개 모델(XGB, CatBoost) GridSearchCV 및 앙상블 예측 시작...")
scaler = StandardScaler()
final_predictions = pd.DataFrame()
validation_predictions = pd.DataFrame()

# 파라미터
param_grids = {
    'xgb': {
        'learning_rate': np.round(np.arange(0.01, 0.1, 0.02), 2).tolist(),
        'max_depth': [5, 7, 9],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    },
    'catboost': {
        'learning_rate': [0.02, 0.05, 0.1],
        'depth': [6, 8, 10],
        'subsample': [0.7, 0.8],
        'l2_leaf_reg': [1, 3, 5] # L2 Regularization
    }
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
    
    X = train_building[current_features]
    y = train_building['전력소비량(kWh)']
    
    current_features_in_X = list(X.columns)
    test_building = test_building[current_features_in_X]
    valid_building = valid_building[current_features_in_X]
    
    if X.empty:
        preds_test = np.zeros(len(test_building))
        building_submission = pd.DataFrame({'answer': preds_test})
        final_predictions = pd.concat([final_predictions, building_submission], ignore_index=True)
        preds_valid = np.zeros(len(valid_building))
        building_validation = pd.DataFrame({'answer': preds_valid})
        validation_predictions = pd.concat([validation_predictions, building_validation], ignore_index=True)
        continue

    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(test_building)
    X_valid_scaled = scaler.transform(valid_building)
    y_log = np.log1p(y)
    
    test_preds_dict = {}
    valid_preds_dict = {}

    # [수정] lgbm 모델 루프에서 제거
    for model_name in ['xgb', 'catboost']:
        print(f"\n--- 건물 {building_num} / {model_name} 모델 HPO 시작 ---")
        
        # [수정] GridSearchCV 용 모델에는 n_estimators를 낮게 설정
        base_params = {'random_state': 42, 'n_estimators': 200}
        if model_name == 'xgb':
            model = xgb.XGBRegressor(objective='reg:squarederror', **base_params)
        else: # catboost
            model = cb.CatBoostRegressor(verbose=0, **base_params)
            
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], scoring='neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(X_scaled, y_log)
        
        best_params = grid_search.best_params_
        best_params['n_estimators'] = 1000 # 최종 모델은 나무 1000개로 학습
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_log, test_size=0.15, random_state=42)

        print(f"--- 건물 {building_num} / {model_name} 최종 모델 학습 시작 ---")
        if model_name == 'xgb':
            es_xgb = xgb.callback.EarlyStopping(rounds = 400,metric_name = 'mae',data_name = 'validation_0',save_best = True,)
            final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params, eval_metric='mae', callbacks = [es_xgb])
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            final_model.save_model(w_path + f'building_{building_num}_xgb_model.json')
            test_preds_log = final_model.predict(X_test_scaled, iteration_range=(0, final_model.best_iteration))
            valid_preds_log = final_model.predict(X_valid_scaled, iteration_range=(0, final_model.best_iteration))
        
        else: # catboost
            final_model = cb.CatBoostRegressor(random_state=42, **best_params, verbose=0)
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, use_best_model=True)
            final_model.save_model(w_path + f'building_{building_num}_cat_model.cbm')
            test_preds_log = final_model.predict(X_test_scaled)
            valid_preds_log = final_model.predict(X_valid_scaled)
        
        test_preds_dict[model_name] = np.expm1(test_preds_log)
        valid_preds_dict[model_name] = np.expm1(valid_preds_log)

    print(f"--- 건물 {building_num} / 앙상블 수행 ---")
    ensemble_test_preds = np.mean(list(test_preds_dict.values()), axis=0)
    ensemble_test_preds[ensemble_test_preds < 0] = 0
    
    ensemble_valid_preds = np.mean(list(valid_preds_dict.values()), axis=0)
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
final_filename = s_path + f'submission_score_{final_mae_score:.4f}_{timestamp}.csv'
sample_submission_df.to_csv(final_filename, index=False)
print(f"제출 파일 '{final_filename}' 생성이 완료되었습니다.")