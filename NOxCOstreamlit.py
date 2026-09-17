import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# ==========================================
# 1. 데이터 로드 및 모델 학습 (캐싱 적용)
# ==========================================
# @st.cache_resource를 사용하면 앱을 새로고침해도 데이터와 모델을 매번 다시 학습하지 않습니다.
@st.cache_resource
def load_data_and_train_models():
    # .env 파일 로드
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(base_dir, '.env')
    load_dotenv(dotenv_path=env_path)
    file_path = os.getenv("DATA_PATH")
    
    if file_path is None:
        raise ValueError(".env 파일에 DATA_PATH가 없습니다.")
    if not os.path.isabs(file_path):
        file_path = os.path.join(base_dir, file_path)

    df = pd.read_csv(file_path)
    df = df.dropna()

    # 서로게이트 모델 학습
    feature_cols_all = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]
    targets = ["NOX", "CO", "TEY"]
    surrogates = {} 

    for t in targets:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)) # 속도를 위해 n_estimators 조정
        ])
        pipe.fit(df[feature_cols_all], df[t])
        surrogates[t] = pipe

    # 날씨 관계식 추출 (AT에 따른 AP, AH 선형 회귀)
    ap_slope, ap_intercept = np.polyfit(df['AT'], df['AP'], 1)
    ah_slope, ah_intercept = np.polyfit(df['AT'], df['AH'], 1)
    weather_eq = {
        "ap": (ap_slope, ap_intercept),
        "ah": (ah_slope, ah_intercept)
    }

    # 배출가스 제약 한계치 (90분위수)
    nox_limit = np.percentile(df["NOX"], 90)
    co_limit = np.percentile(df["CO"], 90)

    return df, surrogates, weather_eq, nox_limit, co_limit

# 데이터 및 모델 로드 (최초 1회 실행)
df, surrogates, weather_eq, nox_limit, co_limit = load_data_and_train_models()

# ==========================================
# 2. 대시보드 UI 구성
# ==========================================
st.title("⚡ 가스 터빈 발전량 최적화 대시보드")
st.markdown("대기 온도를 설정하면, 데이터 기반으로 대기 압력과 습도가 자동 계산되어 최적의 터빈 운전점을 도출합니다.")

st.sidebar.header("🌡️ 대기 조건 설정")

# 온도 슬라이더 (데이터의 Min ~ Max 범위)
at_input = st.sidebar.slider(
    "대기 온도 (AT) [°C]", 
    min_value=float(df['AT'].min()), 
    max_value=float(df['AT'].max()), 
    value=float(df['AT'].mean()), 
    step=0.1
)

# 온도에 따른 압력, 습도 자동 계산
ap_slope, ap_intercept = weather_eq["ap"]
ah_slope, ah_intercept = weather_eq["ah"]

ap_calc = ap_slope * at_input + ap_intercept
ah_calc = ah_slope * at_input + ah_intercept

# 물리적 한계 범위 내로 클리핑
ap_calc = np.clip(ap_calc, df['AP'].min(), df['AP'].max())
ah_calc = np.clip(ah_calc, df['AH'].min(), df['AH'].max())

st.sidebar.markdown("---")
st.sidebar.write(f"**연동된 압력 (AP):** `{ap_calc:.2f}` mbar")
st.sidebar.write(f"**연동된 습도 (AH):** `{ah_calc:.2f}` %")
st.sidebar.markdown("*(데이터 기반 선형회귀식으로 자동 계산됨)*")

# ==========================================
# 3. 최적화 실행 버튼
# ==========================================
if st.sidebar.button("🚀 최적 발전량 계산하기", use_container_width=True):
    
    # 스피너(로딩 표시)를 띄우면서 최적화 수행
    with st.spinner('베이지안 최적화 알고리즘이 최적의 운전점을 탐색 중...'):
        
        # 1. 고정된 대기 조건 설정
        fixed_weather = {
            "AT": at_input,
            "AP": float(ap_calc),
            "AH": float(ah_calc)
        }

        # 2. 통제 변수 설정
        control_vars = ["AFDP", "GTEP", "TIT", "TAT", "CDP"]
        feature_cols_all = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "CDP"]

        bounds = {}
        for c in control_vars:
            lo, hi = np.percentile(df[c], [5, 95])
            bounds[c] = (lo, hi)
        dimensions = [Real(lo, hi, name=c) for c, (lo, hi) in bounds.items()]

        # 3. 목적 함수 정의
        def objective(x):
            input_data = fixed_weather.copy()
            for i, var in enumerate(control_vars):
                input_data[var] = x[i]
            
            X = pd.DataFrame([input_data])[feature_cols_all]
            
            tey_pred = surrogates["TEY"].predict(X)[0]
            nox_pred = surrogates["NOX"].predict(X)[0]
            co_pred  = surrogates["CO"].predict(X)[0]

            penalty = 0.0
            if nox_pred > nox_limit:
                penalty += 1000 * (nox_pred - nox_limit)
            if co_pred > co_limit:
                penalty += 1000 * (co_pred - co_limit)
            
            return -tey_pred + penalty

        # 4. 베이지안 최적화 실행
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=30,          # 대시보드 반응 속도를 위해 30회로 설정
            n_random_starts=5,
            acq_func="EI",
            random_state=42
        )

        # 5. 최적화 결과에서 예측값 다시 추출
        optimal_x = result.x
        input_data = fixed_weather.copy()
        for i, var in enumerate(control_vars):
            input_data[var] = optimal_x[i]
        X_opt = pd.DataFrame([input_data])[feature_cols_all]

        opt_tey = surrogates["TEY"].predict(X_opt)[0]
        opt_nox = surrogates["NOX"].predict(X_opt)[0]
        opt_co = surrogates["CO"].predict(X_opt)[0]

        # ==========================================
        # 4. 결과 출력
        # ==========================================
        st.success("최적화 완료! 아래 결과를 확인하세요.")
        
        # 상단 지표 (Metric) 3개 출력
        col1, col2, col3 = st.columns(3)
        col1.metric("최대 발전량 (TEY)", f"{opt_tey:.2f} MWh")
        col2.metric("예측 NOx 배출량", f"{opt_nox:.2f} mg/m³", delta=f"한계치: {nox_limit:.2f}", delta_color="inverse")
        col3.metric("예측 CO 배출량", f"{opt_co:.2f} mg/m³", delta=f"한계치: {co_limit:.2f}", delta_color="inverse")

        st.markdown("---")
        st.subheader("🔧 추천하는 최적 운전점 (통제 변수)")
        
        # 결과를 데이터프레임으로 만들어 표 출력
        result_df = pd.DataFrame({
            "통제 변수": control_vars,
            "최적값": [round(x, 2) for x in optimal_x],
            "현재 평균": [round(df[c].mean(), 2) for c in control_vars],
            "변화량": [round(x - df[c].mean(), 2) for x, c in zip(optimal_x, control_vars)]
        })
        st.dataframe(result_df, use_container_width=True)

        # 참고용: 설정된 날씨 다시 확인
        with st.expander("설정된 대기 조건 확인"):
            st.json(fixed_weather)

