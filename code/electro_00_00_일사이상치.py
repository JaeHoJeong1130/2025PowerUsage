# import pandas as pd

# # train.csv 파일을 불러옵니다.
# try:
#     df = pd.read_csv('/home/jjh/Project/_data/dacon/04_electro/train.csv')
# except FileNotFoundError:
#     print("train.csv 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
#     exit()

# # '일시' 컬럼에서 날짜와 시간 정보를 추출합니다.
# df['일시_str'] = df['일시'].astype(str)
# df['일자'] = df['일시_str'].str[:8]
# df['시간'] = df['일시_str'].str[-2:].astype(int)

# # 1. 12시, 13시, 14시 데이터만 필터링합니다.
# df_target_hours = df[df['시간'].isin([12, 13, 14])]

# # 2. 건물번호와 일자별로 그룹화하여 일사량의 합계를 계산합니다.
# daily_insolation_sum = df_target_hours.groupby(['건물번호', '일자'])['일사(MJ/m2)'].sum().reset_index()

# # 3. 해당 날짜에 3개 시간의 데이터가 모두 존재하는지 확인하기 위해 개수를 셉니다.
# hourly_counts = df_target_hours.groupby(['건물번호', '일자']).size().reset_index(name='record_count')

# # 4. 두 데이터를 합칩니다.
# merged_df = pd.merge(daily_insolation_sum, hourly_counts, on=['건물번호', '일자'])

# # 5. 데이터가 3개 모두 존재하고, 일사량의 합이 0인 경우를 필터링합니다.
# target_days = merged_df[(merged_df['일사(MJ/m2)'] == 0) & (merged_df['record_count'] == 3)]

# # 6. 결과를 '건물번호'와 '일자' 순으로 정렬합니다.
# result = target_days[['건물번호', '일자']].sort_values(by=['건물번호', '일자'])

# # 결과를 출력합니다.
# # to_string()을 사용하여 모든 행이 출력되도록 합니다.
# print("12시, 13시, 14시 모두 일사량이 0이었던 건물과 날짜:")
# print(result.to_string(index=False))


# =================================================================================================

import pandas as pd

# train.csv 파일을 불러옵니다.
# 사용자의 환경에 맞게 파일 경로를 수정해주세요.
try:
    df = pd.read_csv('/home/jjh/Project/_data/dacon/04_electro/train.csv')
except FileNotFoundError:
    print("train.csv 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

# '일시' 컬럼에서 시간 정보를 추출합니다.
df['일시_str'] = df['일시'].astype(str)
df['시간'] = df['일시_str'].str[-2:].astype(int)

# 1. 시간이 14시인 데이터만 필터링합니다.
df_14h = df[df['시간'] == 15]

# 2. 14시 데이터 중에서 '일사(MJ/m2)'가 0인 경우를 필터링합니다.
result_df = df_14h[df_14h['일사(MJ/m2)'] == 0]

# 결과를 출력합니다.
print("14시에 일사량이 0인 모든 데이터:")
# 모든 열을 보기 좋게 출력하기 위해 to_string()을 사용합니다.
print(result_df.to_string())