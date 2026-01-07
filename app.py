import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 모델 로드
model = YOLO('C:/GasProject/Gas_AI/v1_nano/weights/best.pt')

st.title("🔥 가스 시설물 AI 탐지기")
st.write("사진을 업로드하면 AI가 시설물을 분석합니다.")

uploaded_file = st.file_uploader("가스 시설 사진을 선택하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='업로드된 사진', use_column_width=True)
    st.write("분석 중...")

    # 예측 실행
    results = model.predict(image)
    
    # 결과 이미지 그리기
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='AI 분석 결과', use_column_width=True)