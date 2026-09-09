import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'merged_df.csv')

df=pd.read_csv('/Users/hyobeen/Documents/project/CN/merged_df.csv')
df.dropna()


# ==========================================
# 2. 모듈 1: 서로게이트 모델 학습 (변수 surrogates 생성)
# ==========================================
# 예측에 사용할 전체 입력 변수 (대기변수 + 통제변수)
feature_cols_all = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]
targets = ["NOX", "CO", "TEY"]

print("서로게이트 모델 학습 시작...")
surrogates = {} # <--- 이 변수가 에러의 원인이었습니다. 여기서 생성합니다.

for t in targets:
    # 파이프라인: 스케일링 + Gradient Boosting 회귀
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))
    ])
    
    # 시계열 교차검증으로 성능 확인 (선택사항이지만 권장)
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(pipe, df[feature_cols_all], df[t], cv=tscv, scoring="r2")
    
    # 전체 데이터로 최종 모델 학습
    pipe.fit(df[feature_cols_all], df[t])
    surrogates[t] = pipe
    print(f" - {t} 모델 학습 완료 (R²: {scores.mean():.3f})")

# ==========================================
# 3. 모듈 2: 최적화 문제 정식화 (대기 변수 고정)
# ==========================================
# 운전자가 조작할 수 있는 통제 변수만 최적화 대상으로 설정
control_vars = ["AFDP", "GTEP", "TIT", "TAT", "CDP"]

# 현재 주어진 대기 조건 (날씨) - 실제 현장 상황에 맞게 수정 가능
fixed_weather = {
    "AT": 15.0,  
    "AP": 1013.0, 
    "AH": 60.0   
}

# 제약 조건 (배출가스 상한선)
nox_limit = np.percentile(df["NOX"], 90)
co_limit  = np.percentile(df["CO"], 90)

# 통제 변수 탐색 공간 (데이터의 5%~95% 분위수 내에서 탐색)
bounds = {}
for c in control_vars:
    lo, hi = np.percentile(df[c], [5, 95])
    bounds[c] = (lo, hi)

dimensions = [Real(lo, hi, name=c) for c, (lo, hi) in bounds.items()]

# 목적 함수 정의
def objective(x, surrogates, control_vars, fixed_weather, feature_cols_all, nox_limit, co_limit):
    # 고정 대기조건 + 알고리즘이 제안한 통제변수(x) 결합
    input_data = fixed_weather.copy()
    for i, var in enumerate(control_vars):
        input_data[var] = x[i]
    
    # 모델 입력 순서에 맞게 DataFrame 생성
    X = pd.DataFrame([input_data])[feature_cols_all]
    
    # 예측
    tey_pred = surrogates["TEY"].predict(X)[0]
    nox_pred = surrogates["NOX"].predict(X)[0]
    co_pred  = surrogates["CO"].predict(X)[0]

    # 제약 위반 시 패널티
    penalty = 0.0
    if nox_pred > nox_limit:
        penalty += 1000 * (nox_pred - nox_limit)
    if co_pred > co_limit:
        penalty += 1000 * (co_pred - co_limit)
    
    return -tey_pred + penalty  # 최소화 문제로 변환 (TEY 최대화)

# ==========================================
# 4. 모듈 3: 베이지안 최적화 실행
# ==========================================
print("\n베이지안 최적화 시작...")
result = gp_minimize(
    func=lambda x: objective(x, surrogates, control_vars, fixed_weather, feature_cols_all, nox_limit, co_limit),
    dimensions=dimensions,
    n_calls=50,             # 테스트용으로 50회로 줄임 (빠른 실행을 위해)
    n_random_starts=10,     
    acq_func="EI",
    random_state=42
)

# ==========================================
# 5. 결과 출력
# ==========================================
print("\n" + "="*40)
print("최적화 완료!")
print(f"설정된 대기 조건: {fixed_weather}")
print(f"최대 발전량(TEY): {-result.fun:.2f} MWH")
print("최적 운전점(통제변수):")
optimal_point = dict(zip(control_vars, np.round(result.x, 3)))
for k, v in optimal_point.items():
    print(f"  - {k}: {v}")

