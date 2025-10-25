import pandas as pd
filepath = "/home/jjh/Project/_data/dacon/electro/train.csv"

import pandas as pd

# 입력 CSV 파일 경로
input_file_path = "/home/jjh/Project/_data/dacon/electro/train.csv"

# 최종 출력 CSV 파일 경로
output_file_path = "/home/jjh/Project/_data/dacon/electro/sorted_selected_rows.csv"

# 총 행의 개수 (헤더 포함) - 문제에서 주어진 값
total_rows = 204001

try:
    # 1단계: 특정 행 추출을 위한 인덱스 계산
    rows_to_extract_indices = []
    # 2번째 행 (인덱스 1)부터 시작하여 (2040*k + 2)번째 행 (인덱스 2040*k + 1)까지
    # 201962번째 행 (인덱스 201961)까지 포함
    for k in range(0, (201960 // 2040) + 1):
        index = (2040 * k) + 1
        if index < total_rows: # 실제 총 행의 개수를 넘지 않도록 확인
            rows_to_extract_indices.append(index)

    # train.csv 파일에서 데이터 불러오기
    df = pd.read_csv(input_file_path)

    # 2단계: 특정 행들만 선택
    selected_df = df.iloc[rows_to_extract_indices]

    # 3단계: '기온(°C)' 컬럼을 기준으로 오름차순 정렬
    sorted_df = selected_df.sort_values(by='기온(°C)', ascending=True)

    # 정렬된 데이터를 새 CSV 파일로 저장
    sorted_df.to_csv(output_file_path, index=False)

    print(f"총 {len(rows_to_extract_indices)}개의 행이 '{input_file_path}'에서 추출되었습니다.")
    print(f"추출된 데이터가 '기온(°C)' 컬럼을 기준으로 정렬되어 '{output_file_path}' 파일로 성공적으로 저장되었습니다.")

    print("\n최종 정렬된 데이터의 일부 미리보기:")
    print(sorted_df.head())

except FileNotFoundError:
    print(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
except KeyError:
    print(f"오류: 데이터에 '기온(°C)' 컬럼이 존재하지 않습니다. 컬럼 이름을 확인해주세요.")
except Exception as e:
    print(f"데이터 처리 중 오류가 발생했습니다: {e}")