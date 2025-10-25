import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings

# --- 경로 설정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 전처리
# ----------------------------------------------------------------------
try:
    train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
    building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
    print("데이터 로드 성공")
except FileNotFoundError as e:
    print(f"오류: {e}. 지정된 경로에 파일이 있는지 확인해주세요.")
    exit()

cols_to_process = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in cols_to_process:
    building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)

# ✨✨✨ 수정된 부분 시작 ✨✨✨
# '일시' 컬럼을 datetime 객체로 변환
train_df['일시'] = pd.to_datetime(train_df['일시'])
# test_df 전체를 덮어쓰지 않도록 컬럼을 명확히 지정
test_df['일시'] = pd.to_datetime(test_df['일시'])
# ✨✨✨ 수정된 부분 끝 ✨✨✨

# 건물 정보 병합
train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

train_df = pd.get_dummies(train_df, columns=['건물유형'], prefix='건물유형')
# test_df와 train_df의 컬럼을 맞춰주기 위해 reindex 사용
common_cols = list(set(train_df.columns) | set(test_df.columns))
test_df = pd.get_dummies(test_df, columns=['건물유형'], prefix='건물유형').reindex(columns=common_cols, fill_value=0)


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


# 2. 하이퍼파라미터 탐색 준비
# ----------------------------------------------------------------------
optimal_features = [
    '습도(%)', '공휴일', '건물유형_아파트', '냉방면적(m2)', '시간', 'PCS용량(kW)',
    '강수량(mm)', '건물유형_IDC(전화국)', '건물유형_건물기타', '건물유형_상용', '일',
    '건물유형_호텔', '태양광용량(kW)', '기온(°C)', 'MMDDHH', '건물유형_연구소',
    '풍속(m/s)', '건물유형_백화점', '월', 'ESS저장용량(kWh)', '건물번호'
]

if '건물번호' not in optimal_features:
    optimal_features.append('건물번호')

# train_df에만 있는 컬럼이 optimal_features에 포함될 수 있으므로, 실제 존재하는 컬럼만 사용
final_features = [col for col in optimal_features if col in train_df.columns]
X_train = train_df[final_features]
y_train = np.log1p(train_df['전력소비량(kWh)'])

print(f"\n하이퍼파라미터 탐색을 위해 {X_train.shape[1]}개의 피처를 사용합니다.")


# 3. GridSearchCV를 이용한 하이퍼파라미터 탐색
# ----------------------------------------------------------------------
print("GridSearchCV를 이용한 최적 하이퍼파라미터 탐색을 시작합니다...")

param_grid = {
    'max_depth': [5, 7, 9],
    'learning_rate': np.round(np.arange(0.01, 0.1, 0.01), 2).tolist(),
    'n_estimators': [1000],
    'subsample': np.round(np.arange(0.6, 0.9, 0.1), 1).tolist(),
    'colsample_bytree': np.round(np.arange(0.6, 0.9, 0.1), 1).tolist()
}

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# --- 최종 결과 출력 ---
print("\n\n===== 하이퍼파라미터 탐색 완료 =====")
print(f"최고 점수 (Negative MSE): {grid_search.best_score_:.4f}")
print("\n--- 최적 하이퍼파라미터 ---")
print(grid_search.best_params_)