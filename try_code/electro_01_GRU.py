import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import warnings

# --- 경로 설정 ---
path = "/home/jjh/Project/_data/dacon/electro/"
w_path = "/home/jjh/Project/_data/dacon/electro/wei/"
s_path = "/home/jjh/Project/_data/dacon/electro/sub/"

warnings.filterwarnings('ignore')

# --- GPU 설정 ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. 데이터 로드
try:
    train_df = pd.read_csv(path + 'train.csv', encoding='utf-8')
    test_df = pd.read_csv(path + 'test.csv', encoding='utf-8')
    building_info_df = pd.read_csv(path + 'building_info.csv', encoding='utf-8')
    submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')
    print("데이터 로드 성공")
except FileNotFoundError as e:
    print(f"오류: {e}. 지정된 경로에 파일이 있는지 확인해주세요.")
    exit()

# 2. 전처리 및 피처 엔지니어링

# ✨✨ 수정 1: building_info의 '-' 값을 0으로 변환 ✨✨
cols_to_process = ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
for col in cols_to_process:
    building_info_df[col] = building_info_df[col].replace('-', '0').astype(float)
print("'-' 값 처리 및 숫자 타입 변환 완료")

# '일시' 컬럼을 datetime 객체로 변환
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

# 건물 정보 병합
train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

# 결측치 처리
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

# 건물유형 원-핫 인코딩
train_df = pd.get_dummies(train_df, columns=['건물유형'], prefix='건물유형')
test_df = pd.get_dummies(test_df, columns=['건물유형'], prefix='건물유형')

# 시간 관련 피처 생성 함수
def create_time_features(df):
    df_copy = df.copy()
    holidays = pd.to_datetime(['2024-06-06', '2024-08-15'])
    
    # ✨✨ 수정 2: .dt.normalize()를 사용하여 날짜 비교 (ValueError 해결) ✨✨
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

# ✨✨ 수정 3: Train/Test 공통 피처만 사용 (KeyError 해결) ✨✨
common_cols = list(set(train_df.columns) & set(test_df.columns))
features_to_use = [col for col in common_cols if col not in ['num_date_time', '건물번호', '일시']]
print(f"Train/Test 공통 피처 {len(features_to_use)}개를 사용합니다.")


# 3. 모델 학습 및 예측
# 하이퍼파라미터 설정
INPUT_WINDOW = 24 * 14
OUTPUT_WINDOW = 24 * 7
N_FEATURES = len(features_to_use) # 루프 밖에서 한 번만 계산
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

all_predictions = []

# 100개 건물에 대해 모델 학습 및 예측 반복
for building_num in range(1, 101):
    print(f"\n===== 건물 {building_num} 처리 시작 =====")

    model_dir = w_path + 'model_weights/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = model_dir + f'building_{building_num}_gru_weights.h5'

    train_building_df = train_df[train_df['건물번호'] == building_num].copy()
    test_building_df = test_df[test_df['건물번호'] == building_num].copy()
    
    # X 데이터 스케일링
    scaler_X = MinMaxScaler()
    train_building_df[features_to_use] = scaler_X.fit_transform(train_building_df[features_to_use])
    test_building_df[features_to_use] = scaler_X.transform(test_building_df[features_to_use])

    # y 데이터 (타겟) 로그 변환
    train_building_df['전력소비량_log'] = np.log1p(train_building_df['전력소비량(kWh)'])

    # 시계열 데이터셋 생성 함수
    def create_sequences(train_data, test_data, feature_cols, target_col):
        encoder_input = []
        decoder_input = []
        decoder_target = []

        for i in range(len(train_data) - INPUT_WINDOW - OUTPUT_WINDOW + 1):
            enc_seq = train_data.iloc[i:i+INPUT_WINDOW][[target_col] + feature_cols]
            dec_target_seq = train_data.iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+OUTPUT_WINDOW][target_col]
            dec_input_seq = train_data.iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+OUTPUT_WINDOW][feature_cols]

            encoder_input.append(np.array(enc_seq))
            decoder_input.append(np.array(dec_input_seq))
            decoder_target.append(np.array(dec_target_seq).reshape(-1, 1))
        
        last_train_seq = train_data.iloc[-INPUT_WINDOW:][[target_col] + feature_cols]
        prediction_input_enc = np.array(last_train_seq).reshape(1, INPUT_WINDOW, N_FEATURES + 1)
        prediction_input_dec = np.array(test_data[feature_cols]).reshape(1, OUTPUT_WINDOW, N_FEATURES)

        return (np.array(encoder_input), np.array(decoder_input), np.array(decoder_target),
                prediction_input_enc, prediction_input_dec)

    enc_in, dec_in, dec_out, pred_enc_in, pred_dec_in = create_sequences(
        train_building_df, test_building_df, features_to_use, '전력소비량_log'
    )
    
    # 데이터가 부족하여 시퀀스를 만들 수 없는 경우 건너뛰기
    if len(enc_in) == 0:
        print(f"건물 {building_num} 데이터 부족으로 학습을 건너뜁니다. 0으로 예측합니다.")
        # 테스트 기간만큼 0으로 예측값 추가
        all_predictions.extend([0] * OUTPUT_WINDOW)
        continue

    print(f"학습 데이터 형태: {enc_in.shape}, {dec_in.shape}, {dec_out.shape}")

    # Seq2Seq GRU 모델 정의
    encoder_inputs = Input(shape=(INPUT_WINDOW, N_FEATURES + 1), name='encoder_input')
    encoder_gru = GRU(128, return_state=True, name='encoder_gru')
    _, state_h = encoder_gru(encoder_inputs)

    decoder_inputs = Input(shape=(OUTPUT_WINDOW, N_FEATURES), name='decoder_input')
    decoder_gru = GRU(128, return_sequences=True, name='decoder_gru')
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(1, name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mae')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    
    model.fit(
        [enc_in, dec_in], dec_out,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    model.save_weights(model_path)
    print(f"건물 {building_num} 모델 가중치 저장 완료: {model_path}")
    
    prediction_log = model.predict([pred_enc_in, pred_dec_in])
    prediction = np.expm1(prediction_log.flatten())
    prediction[prediction < 0] = 0
    all_predictions.extend(prediction)

# 4. 제출 파일 생성
print("\n===== 최종 제출 파일 생성 시작 =====")
submission_df['answer'] = all_predictions
current_time = datetime.datetime.now().strftime("%m%d%H%M%S")
submission_filename = f'submission_{current_time}.csv'
submission_df.to_csv(s_path + submission_filename, index=False)
print(f"제출 파일 생성 완료: {s_path}{submission_filename}")
print(submission_df.head())