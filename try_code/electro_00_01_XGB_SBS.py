import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler 임포트
import warnings

# --- 경로 설정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 전처리 (이전과 동일)
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

train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# get_dummies로 생성될 컬럼을 동일하게 유지하기 위한 처리
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

common_cols = list(set(train_df.columns) & set(test_df.columns))
initial_features = [col for col in common_cols if col not in ['num_date_time', '일시']]
print("\n--- 초기 피처 리스트 ---")
