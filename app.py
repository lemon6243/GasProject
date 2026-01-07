import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# 1. 모델 경로 설정 (GitHub 서버 환경 대응)
# 현재 app.py 파일이 있는 위치를 기준으로 best.pt를 찾습니다.
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'best.pt')

# 2. 모델 로드
# 모델 로딩 중 에러가 발생할 경우를 대비해 캐싱 처리를 하면 더 빠릅니다.
@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# 3. UI 구성
st.set_page_config(page_title="가스 시설 AI 탐지기", page_icon="🔥")
st.title("🔥 가스 시설물 AI 탐지기")
st.write("현장에서 찍은 사진을 업로드하면 AI가 시설물(밸브, 보일러 등)을 자동으로 분석합니다.")

# 4. 파일 업로드
uploaded_file = st.file_uploader("가스 시설 사진을 선택하세요 (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 열기
    image = Image.open(uploaded_file)
    
    # 화면을 두 칸으로 나누어 출력
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("원본 사진")
        st.image(image, use_container_width=True)

    with col2:
        st.info("AI 분석 결과")
        with st.spinner('분석 중...'):
            # 모델 예측 (BGR 변환 없이 PIL 이미지 바로 사용 가능)
            results = model.predict(image, conf=0.25) # 신뢰도 0.25 이상만 표시
            
            # 결과 이미지 그리기 (numpy 배열로 변환)
            res_plotted = results[0].plot()
            
            # RGB로 변환하여 출력 (OpenCV 형식을 Streamlit 형식으로)
            st.image(res_plotted, channels="BGR", use_container_width=True)

    # 분석 상세 정보 출력
    st.success("분석 완료!")
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.write(f"총 {len(boxes)}개의 시설물을 발견했습니다.")
    else:
        st.write("감지된 시설물이 없습니다. 다른 사진으로 시도해 보세요.")
