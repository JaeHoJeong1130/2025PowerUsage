import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import datetime
import os

# --- 경로 설정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
w_path = "/home/jjh/Project/_data/dacon/electro/wei/"
s_path = "/home/jjh/Project/_data/dacon/electro/sub/"

os.makedirs(w_path, exist_ok=True)
os.makedirs(s_path, exist_ok=True)

warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 전처리 (이전과 동일)
# ----------------------------------------------------------------------
try:
    train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
    building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
    submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')
    print("데이터 로드 성공")
except FileNotFoundError as e:
    print(f"오류: {e}. 지정된 경로에 파일이 있는지 확인해주세요.")
    exit()

cols_to_process = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in cols_to_process:
    building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)

train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

all_building_types = pd.concat([train_df['건물유형'], test_df['건물유형']]).unique()
train_df['건물유형'] = pd.Categorical(train_df['건물유형'], categories=all_building_types)
test_df['건물유형'] = pd.Categorical(test_df['건물유형'], categories=all_building_types)

train_df = pd.get_dummies(train_df, columns=['건물유형'], prefix='건물유형')
test_df = pd.get_dummies(test_df, columns=['건물유형'], prefix='건물유형')

def create_time_features(df):
    df_copy = df.copy()
    holidays = pd.to_datetime(['2024-06-06', '2024-08-15'])
    df_copy['공휴일'] = df_copy['일시'].dt.normalize().isin(holidays).astype(int)
    df_copy['요일'] = df_copy['일시'].dt.weekday
    df_copy['시간'] = df_copy['일시'].dt.hour
    df_copy['월'] = df_copy['일시'].dt.month
    df_copy['일'] = df_copy['일시'].dt.day
    df_copy['MMDDHH'] = df_copy['월'] * 10000 + df_copy['일'] * 100 + df_copy['시간']
    return df_copy

train_df = create_time_features(train_df)
test_df = create_time_features(test_df)
print("피처 엔지니어링 완료")
# ----------------------------------------------------------------------


# 2. 최종 모델 학습 및 저장
# ----------------------------------------------------------------------
optimal_features = [
    '습도(%)', '공휴일', '건물유형_아파트', '냉방면적(m2)', '시간', 'PCS용량(kW)',
    '강수량(mm)', '건물유형_IDC(전화국)', '건물유형_건물기타', '건물유형_상용', '일',
    '건물유형_호텔', '태양광용량(kW)', '기온(°C)', 'MMDDHH', '건물유형_연구소',
    '풍속(m/s)', '건물유형_백화점', '월', 'ESS저장용량(kWh)', '건물번호'
]

best_params = {
    'colsample_bytree': 0.6,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 1000, # 충분히 큰 값으로 설정하고 조기 종료에 맡김
    'subsample': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1
}

# 최종 학습 데이터 준비
X_train_full = train_df[optimal_features]

y_train_full = np.log1p(train_df['전력소비량(kWh)'])

# ✨✨✨ 얼리 스토핑을 위한 검증 데이터 분리 ✨✨✨
# 시계열 데이터이므로, 가장 마지막 데이터를 검증용으로 사용합니다.
# 예: 마지막 2주(100개 건물 * 24시간 * 14일)를 검증 데이터로 활용
val_size = 100 * 24 * 14
X_train_sub = X_train_full.iloc[:-val_size]
y_train_sub = y_train_full.iloc[:-val_size]
X_val = X_train_full.iloc[-val_size:]
y_val = y_train_full.iloc[-val_size:]

print(f"\n훈련 데이터: {X_train_sub.shape}, 검증 데이터: {X_val.shape}")
print("최종 모델 학습을 시작합니다 (Early Stopping 적용)...")

es_xgb = xgb.callback.EarlyStopping(
    rounds = 400,
    metric_name = 'mae',
    data_name = 'validation_0',
    save_best = True,
)

# 최적 하이퍼파라미터로 모델 초기화
final_model = xgb.XGBRegressor(**best_params,
                               callbacks = [es_xgb],
                               )

# ✨✨✨ 얼리 스토핑을 적용하여 모델 훈련 ✨✨✨
final_model.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],           # 검증용 데이터 지정
    verbose=100                          # 100라운드마다 학습 과정 출력
)

print("모델 학습 완료!")

# 모델 저장
model_filename = f'xgb_model_early_stopping_{datetime.datetime.now().strftime("%m%d%H%M")}.json'
final_model.save_model(w_path + model_filename)
print(f"학습된 모델을 '{w_path}{model_filename}'에 저장했습니다.")


# 3. 예측 및 제출 파일 생성
# ----------------------------------------------------------------------
print("\n테스트 데이터에 대한 예측을 시작합니다...")

X_test = test_df[optimal_features]
predictions_log = final_model.predict(X_test)
predictions = np.expm1(predictions_log)
predictions[predictions < 0] = 0

submission_df['answer'] = predictions

submission_filename = f'submission_final_{datetime.datetime.now().strftime("%m%d%H%M%S")}.csv'
submission_df.to_csv(s_path + submission_filename, index=False)

print("예측 완료!")
print(f"최종 제출 파일을 '{s_path}{submission_filename}'에 저장했습니다.")
print("\n--- 제출 파일 미리보기 ---")
print(submission_df.head())