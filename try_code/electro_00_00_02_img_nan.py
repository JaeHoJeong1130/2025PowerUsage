import pandas as pd
import numpy as np  # numpy 라이브러리 추가
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = "/home/jjh/Project/_data/dacon/04_electro/" # 사용자의 기존 경로 유지

# 한글 폰트 설정 (Windows, Mac, Linux 환경에 맞게 경로 수정)
try:
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    print("한글 폰트를 성공적으로 설정했습니다.")
except FileNotFoundError:
    print("지정된 경로에 한글 폰트 파일이 없습니다. 기본 폰트로 실행됩니다. (그래프의 한글이 깨질 수 있습니다)")
    pass

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


def visualize_building_power_consumption_with_nan():
    """
    train.csv 파일을 읽어 사용자가 입력한 건물의 전력소비량(kWh)을 시각화합니다.
    일별 최저 전력량 평균의 절반보다 낮은 값은 NaN으로 처리하여 시각화에서 제외합니다.
    """
    # 1. 데이터 불러오기
    try:
        # train.csv 파일이 코드와 다른 폴더에 있다면 파일 경로를 정확하게 지정해야 합니다.
        # df = pd.read_csv(path + 'train.csv', encoding='utf-8')
        df = pd.read_csv(path + 'train.csv', encoding='utf-8') # 실행 환경을 위해 경로 수정
        print("'train.csv' 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
        print("스크립트와 동일한 폴더에 파일이 있는지 확인해주세요.")
        return

    # 2. 데이터 전처리 (날짜/시간 데이터 변환)
    try:
        df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    except Exception as e:
        print(f"날짜/시간 컬럼('일시') 변환 중 오류가 발생했습니다: {e}")
        print("데이터의 날짜 형식을 확인해주세요. ('%Y%m%d %H' 형식이어야 합니다)")
        return

    # 3. 사용자 입력 및 시각화 반복
    while True:
        user_input = input("\n확인하고 싶은 건물 번호(1-100)를 입력하세요 (종료: q 또는 exit): ")

        if user_input.lower() in ['q', 'exit']:
            print("프로그램을 종료합니다.")
            break

        try:
            building_number = int(user_input)
            if not 1 <= building_number <= 100:
                print("경고: 건물 번호는 1과 100 사이의 숫자여야 합니다.")
                continue

            # 원본 데이터프레임에서 복사하여 사용
            building_df = df[df['건물번호'] == building_number].copy()

            if building_df.empty:
                print(f"결과 없음: {building_number}번 건물에 대한 데이터를 찾을 수 없습니다.")
                continue

            # --- 핵심 로직 시작 ---
            # 4. 일별 최저 전력량 계산 및 평균 산출
            # '일시' 컬럼에서 날짜 정보만 추출하여 새로운 '일' 컬럼 생성
            building_df['일'] = building_df['일시'].dt.date
            
            # 일별로 그룹화하여 최저 전력량 계산
            daily_min_power = building_df.groupby('일')['전력소비량(kWh)'].min()
            
            # 일별 최저 전력량들의 평균 계산
            avg_of_mins = daily_min_power.mean()
            
            # NaN으로 처리할 임계값 설정 (평균의 절반)
            threshold = avg_of_mins / 5 * 4
            
            print(f"\n[건물 {building_number} 분석 정보]")
            print(f"  - 일별 최저 전력량들의 평균: {avg_of_mins:.2f} kWh")
            print(f"  - NaN 처리 임계값 (평균의 50%): {threshold:.2f} kWh")

            # 5. 임계값보다 낮은 데이터를 NaN으로 변경
            original_count = len(building_df)
            building_df.loc[building_df['전력소비량(kWh)'] < threshold, '전력소비량(kWh)'] = np.nan
            nan_count = building_df['전력소비량(kWh)'].isna().sum()
            
            print(f"  - 총 {nan_count}개의 데이터 포인트를 NaN으로 변경했습니다.")
            # --- 핵심 로직 종료 ---


            # 6. 그래프 그리기
            plt.figure(figsize=(16, 7))
            plt.plot(building_df['일시'], building_df['전력소비량(kWh)'],
                     marker='.', markersize=3, linestyle='-',
                     label=f'건물 {building_number} 전력소비량')

            # 임계값 라인을 가로로 추가하여 시각적으로 기준 확인
            plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'NaN 처리 임계값 ({threshold:.2f} kWh)')

            plt.title(f'건물 {building_number}번 시간별 전력소비량 (kWh) - 이상치 제외', fontsize=16)
            plt.xlabel('날짜', fontsize=12)
            plt.ylabel('전력소비량 (kWh)', fontsize=12)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            # 파일로 저장 (실행 환경에서는 show() 대신 savefig() 사용)
            plt.savefig(f'building_{building_number}_power_consumption.png')
            print(f"\n'building_{building_number}_power_consumption.png' 파일로 그래프를 저장했습니다.")
            # plt.show() # 로컬 환경에서 직접 실행 시 이 코드의 주석을 해제하세요.


        except ValueError:
            print("오류: 숫자를 입력하거나 'q' 또는 'exit'를 입력하여 종료해주세요.")
        except Exception as e:
            print(f"알 수 없는 오류가 발생했습니다: {e}")


# 메인 함수 실행
if __name__ == '__main__':
    visualize_building_power_consumption_with_nan()