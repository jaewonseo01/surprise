import subprocess
import sys
import datetime
import warnings

warnings.filterwarnings('ignore')
# 250331 현재 실행한거 mortality 학습 2개, mimic cf 학습
# 250430 현재 Compare_reps에서 train, valid, test length 맞는지 확인중
# 250508 현재 compare_reps에서 target data에서 Test 사용으로 수정함
# 250514 현재 integ에서 CF / LOS 수정하였고 다시 돌려야함함
# 250528 현재 compare_reps에서 전체 데이터에 대해, brier score로 비교 예정, clip /cal 
# 250603 현재 compare_reps에서 결측치 제거하고 모델 및 calibration 시각화 추가하여여 돌려 볼 예정
# 250606 현재 compare_reps에서 static 제거한 경우 및 valid에 calibration으로 수정해서 돌려 볼 예정정


def run_notebook(nb):
    try:
        subprocess.run([
            sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', nb, '--inplace'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {nb}: {e}")
        # 여기서 continue를 위해 pass 또는 로깅 후 진행
        pass


notebooks = [
             'compare_reps_aki_nostatic.ipynb',
             'compare_reps_mor_nostatic.ipynb',
             'compare_reps_los_nostatic.ipynb',
             'compare_reps_cf_nostatic.ipynb',
             'compare_reps_aki_eicu_nostatic.ipynb',
             'compare_reps_mor_eicu_nostatic.ipynb',
             'compare_reps_los_eicu_nostatic.ipynb',
             'compare_reps_cf_eicu_nostatic.ipynb',
            #  'compare_reps_aki.ipynb',
            #  'compare_reps_mor.ipynb',
            #  'compare_reps_los.ipynb',
            #  'compare_reps_cf.ipynb',
            #  'compare_reps_aki_eicu.ipynb',
            #  'compare_reps_mor_eicu.ipynb',
            #  'compare_reps_los_eicu.ipynb',
            #  'compare_reps_cf_eicu.ipynb',
             ]
             
for nb in notebooks:
    print(f'[{nb}] Starting notebook at {datetime.datetime.now()}')
    run_notebook(nb)
    print(f'[{nb}] Notebook finished at {datetime.datetime.now()}')
