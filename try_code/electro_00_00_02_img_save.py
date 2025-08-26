import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

path = "/home/jjh/Project/_data/dacon/04_electro/"

# 한글 폰트 설정 (Windows, Mac, Linux 환경에 맞게 경로 수정)
try:
    font_path = 'C:/Windows/Fonts/malgun.ttf' # 사용 중인 OS에 맞는 한글 폰트 경로를 지정해주세요.
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    print("지정된 경로에 한글 폰트 파일이 없습니다. 기본 폰트로 실행됩니다. (그래프의 한글이 깨질 수 있습니다)")
    pass

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


def save_all_building_plots():
    """
    train.csv 파일을 읽어 1번부터 100번까지 모든 건물의 전력소비량(kWh) 그래프를
    개별 이미지 파일로 저장합니다.
    """
    # 1. 데이터 불러오기
    try:
        df = pd.read_csv(path + 'train.csv', encoding='utf-8')
        print("'train.csv' 파일을 성공적으로 불러왔습니다.")
    except FileNotFoundError:
        print("오류: 'train.csv' 파일을 찾을 수 없습니다.")
        print("스크립트와 동일한 폴더에 파일이 있는지 확인해주세요.")
        return

    # 2. 데이터 전처리 (날짜/시간 데이터 변환)
    try:
        df['일시'] = pd.to_datetime(df['일시'], errors='coerce')
        if df['일시'].isnull().any():
            print("경고: '일시' 컬럼의 일부 데이터를 날짜 형식으로 변환할 수 없습니다. 해당 행은 제외될 수 있습니다.")
            df.dropna(subset=['일시'], inplace=True)
    except Exception as e:
        print(f"날짜/시간 컬럼('일시') 변환 중 오류가 발생했습니다: {e}")
        return

    # 3. 이미지 저장 폴더 생성
    output_dir = 'building_plots'
    os.makedirs(output_dir, exist_ok=True)
    print(f"그래프 이미지 파일은 현재 폴더 안의 '{output_dir}' 폴더에 저장됩니다.")

    # 4. 모든 건물에 대해 반복하며 그래프 저장
    for building_number in range(1, 101):
        # 해당 건물 데이터 필터링
        building_df = df[df['건물번호'] == building_number]

        if building_df.empty:
            print(f"-> 건물 {building_number}번: 데이터가 없어 건너뜁니다.")
            continue

        # 그래프 생성 (매번 새로운 Figure 객체를 만듦)
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(building_df['일시'], building_df['전력소비량(kWh)'],
                marker='.', markersize=3, linestyle='-')

        ax.set_title(f'건물 {building_number}번 시간별 전력소비량 (kWh)', fontsize=16)
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('전력소비량 (kWh)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()

        # 5. 파일로 저장
        # 파일명을 001, 002, ... 와 같이 만들어 정렬하기 편하게 합니다.
        file_name = f'building_{str(building_number).zfill(3)}.png'
        save_path = os.path.join(output_dir, file_name)
        
        try:
            plt.savefig(save_path)
            print(f"-> 건물 {building_number}번 그래프를 '{save_path}'에 저장했습니다.")
        except Exception as e:
            print(f"-> 건물 {building_number}번 그래프 저장 중 오류 발생: {e}")

        # 메모리 누수를 방지하기 위해 그래프 객체를 닫아줍니다.
        plt.close(fig)

    print(f"\n✅ 모든 작업이 완료되었습니다. '{output_dir}' 폴더를 확인해주세요.")


# 메인 함수 실행
if __name__ == '__main__':
    save_all_building_plots()