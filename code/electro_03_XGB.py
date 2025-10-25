import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
import warnings
from tqdm import tqdm
import joblib
import datetime
import random
import os

# --- 경로 설정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
w_path = "/home/jjh/Project/_data/dacon/electro/wei/"
s_path = "/home/jjh/Project/_data/dacon/electro/sub/"

warnings.filterwarnings('ignore')

# ===================================================================
# 랜덤 시드 고정
# ===================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # 원하는 시드 값으로 고정
print("랜덤 시드를 42로 고정했습니다.")
# ===================================================================

# RMSE 함수 (내부 성능 비교용)
def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 1. 데이터 로딩 및 전처리 (이전과 동일) ---
print("\n1. 데이터 로딩 및 초기 전처리 시작...")
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
print("데이터 로딩 및 초기 전처리 완료.")

# --- 2. 피처 엔지니어링 (이전과 동일) ---
print("\n2. 피처 엔지니어링 시작...")
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
print("피처 엔지니어링 완료.")

# --- 2.5. '일조'/'일사' 예측 (이전과 동일) ---
print("\n2.5. Test 데이터의 '일조', '일사' 피처 예측 시작...")
sun_features = ['기온(°C)', '풍속(m/s)', '습도(%)', '강수량(mm)', 'sin_hour', 'cos_hour', 'dayofweek', 'month']
sun_targets = ['일조(hr)', '일사(MJ/m2)']
train_sun = train_df.dropna(subset=sun_targets).copy()
for target in sun_targets:
    X_sun_train = train_sun[sun_features]; y_sun_train = train_sun[target]
    X_sun_test = test_df[sun_features]
    sun_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    sun_model.fit(X_sun_train, y_sun_train)
    sun_predictions = sun_model.predict(X_sun_test)
    sun_predictions[sun_predictions < 0] = 0
    test_df[target] = sun_predictions
print("'일조', '일사' 피처 예측 및 추가 완료.")


print("\n3. 건물별 전력 사용량 모델 학습 및 최적화 시작...")
# --- 3. 모델 학습 및 최적화 (MAE 기준) ---
scaler = StandardScaler()
param_grid = {
    'max_depth': [5, 7],
    'learning_rate': np.round(np.arange(0.01, 0.1, 0.02), 2).tolist(),
    'n_estimators': [1000],
    'subsample': np.round(np.arange(0.7, 0.9, 0.1), 1).tolist(),
    'colsample_bytree': np.round(np.arange(0.7, 0.9, 0.1), 1).tolist(),
}
final_predictions = pd.DataFrame()
# [삭제] best_features_per_building 딕셔너리 제거
best_params_per_building = {}
initial_features = [
    '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)',
    '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
    'month', 'day', 'hour', 'dayofweek', 'MMDDHH', 'holiday',
    'discomfort_index', 'sin_hour', 'cos_hour'
]
initial_features += [col for col in building_info_df.columns if '건물유형_' in col]


for building_num in tqdm(range(1, 101), desc="전체 건물 학습 진행"):
    train_building = train_df[train_df['건물번호'] == building_num].copy()
    test_building = test_df[test_df['건물번호'] == building_num].copy()
    train_building.dropna(axis=1, inplace=True)
    current_features = [f for f in initial_features if f in train_building.columns and f in test_building.columns]
    X = train_building[current_features]; y = train_building['전력소비량(kWh)']
    X_test = test_building[current_features]

    if X.empty:
        preds = np.zeros(len(test_building))
        building_submission = pd.DataFrame({'answer': preds})
        final_predictions = pd.concat([final_predictions, building_submission], ignore_index=True)
        continue

    X_scaled = scaler.fit_transform(X); X_test_scaled = scaler.transform(X_test)
    y_log = np.log1p(y)

    # =================================================
    # [삭제] 피처 선택(Backward Elimination) 로직 제거
    # =================================================
    
    # GridSearchCV는 EarlyStopping을 직접 지원하지 않으므로, 이 부분은 그대로 둡니다.
    model_for_grid = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    # [수정] 모든 피처를 사용 (X_scaled, y_log)
    grid_search = GridSearchCV(estimator=model_for_grid, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=0, n_jobs=5)
    grid_search.fit(X_scaled, y_log) # 모든 피처로 학습
    best_params = grid_search.best_params_
    best_params_per_building[building_num] = best_params

    es_xgb = xgb.callback.EarlyStopping(
    rounds = 400,
    metric_name = 'mae',
    data_name = 'validation_0',
    save_best = True,
    )

    # 최종 모델도 콜백 대신 내장 EarlyStopping 사용
    final_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params, eval_metric='mae',
                                   callbacks = [es_xgb],)
    
    # [수정] 최종 학습을 위해 데이터를 다시 나누지 않고, 모든 학습 데이터로 학습
    # EarlyStopping을 위해 eval_set은 유지
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_log, test_size=0.15, random_state=42)

    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                    verbose=False)

    final_model.save_model(w_path + f'building_{building_num}_model.json')

    # [수정] 예측 시 모든 피처 사용
    predictions_log = final_model.predict(X_test_scaled, iteration_range=(0, final_model.best_iteration))
    predictions = np.expm1(predictions_log)
    predictions[predictions < 0] = 0

    building_submission = pd.DataFrame({'answer': predictions})
    final_predictions = pd.concat([final_predictions, building_submission], ignore_index=True)

print("모든 건물 모델 학습 및 최적화 완료.")

# --- 4. 최종 제출 파일 생성 ---
print("\n4. 최종 제출 파일 생성 시작...")
sample_submission_df['answer'] = final_predictions['answer']
timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")
final_filename = s_path + f'submission_{timestamp}.csv'
sample_submission_df.to_csv(final_filename, index=False)
print(f"제출 파일 '{final_filename}' 생성이 완료되었습니다.")
print("\n저장된 최적 파라미터 예시 (건물 1):")
print(f"  - Parameters: {best_params_per_building.get(1, 'N/A')}")