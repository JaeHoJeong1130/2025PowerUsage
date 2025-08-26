import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
path = "/home/jjh/Project/_data/dacon/04_electro/"

# 한글 폰트 설정 (Windows, Mac, Linux 환경에 맞게 경로 수정)
# Windows의 경우: 'C:/Windows/Fonts/malgun.ttf'
# Mac의 경우: '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
try:
    font_path = 'C:/Windows/Fonts/malgun.ttf' # 사용 중인 OS에 맞는 한글 폰트 경로를 지정해주세요.
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    print("지정된 경로에 한글 폰트 파일이 없습니다. 기본 폰트로 실행됩니다. (그래프의 한글이 깨질 수 있습니다)")
    pass

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


def visualize_building_power_consumption():
    """
    train.csv 파일을 읽어 사용자가 입력한 건물의 전력소비량(kWh)을 시각화합니다.
    데이터 기간: 2024-06-01 ~ 2024-08-24
    """
    # 1. 데이터 불러오기
    try:
        # train.csv 파일이 코드와 다른 폴더에 있다면 파일 경로를 정확하게 지정해야 합니다.
        df = pd.read_csv(path + 'train.csv', encoding='utf-8')
        print("'train.csv' 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
        print("스크립트와 동일한 폴더에 파일이 있는지 확인해주세요.")
        return

    # 2. 데이터 전처리 (날짜/시간 데이터 변환)
    # '일시' 컬럼이 '20240601 00'과 같은 형식이라고 가정합니다.
    # 만약 형식이 다르면 format 부분을 수정해야 합니다.
    try:
        # 일반적인 날짜 형식(예: 2024-06-01 00:00:00)을 먼저 시도
        df['일시'] = pd.to_datetime(df['일시'])
    except ValueError:
        try:
            # 제공된 형식(예: 20240601 00)으로 변환 시도
            df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
        except Exception as e:
            print(f"날짜/시간 컬럼('일시') 변환 중 오류가 발생했습니다: {e}")
            print("데이터의 날짜 형식을 확인해주세요.")
            return

    # 3. 사용자 입력 및 시각화 반복
    while True:
        # 사용자로부터 건물 번호 입력받기
        user_input = input("\n확인하고 싶은 건물 번호(1-100)를 입력하세요 (종료: q 또는 exit): ")

        if user_input.lower() in ['q', 'exit']:
            print("프로그램을 종료합니다.")
            break

        try:
            building_number = int(user_input)
            if not 1 <= building_number <= 100:
                print("경고: 건물 번호는 1과 100 사이의 숫자여야 합니다.")
                continue

            # 해당 건물 데이터 필터링
            building_df = df[df['건물번호'] == building_number]

            if building_df.empty:
                print(f"결과 없음: {building_number}번 건물에 대한 데이터를 찾을 수 없습니다.")
                continue

            # 4. 그래프 그리기
            plt.figure(figsize=(16, 7))
            plt.plot(building_df['일시'], building_df['전력소비량(kWh)'],
                     marker='.', markersize=3, linestyle='-',
                     label=f'건물 {building_number} 전력소비량')

            plt.title(f'건물 {building_number}번 시간별 전력소비량 (kWh)', fontsize=16)
            plt.xlabel('날짜', fontsize=12)
            plt.ylabel('전력소비량 (kWh)', fontsize=12)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout() # 그래프 레이아웃 최적화
            plt.show()

        except ValueError:
            print("오류: 숫자를 입력하거나 'q' 또는 'exit'를 입력하여 종료해주세요.")
        except Exception as e:
            print(f"알 수 없는 오류가 발생했습니다: {e}")


# 메인 함수 실행
if __name__ == '__main__':
    visualize_building_power_consumption()