from ultralytics import YOLO

if __name__ == '__main__':
    # 1. YOLOv8 Nano 모델 불러오기 (v가 빠졌던 부분 수정 및 쉼표 제거)
    model = YOLO('yolov8n.pt') 

    # 2. 모델 학습 시작
    model.train(
        data='C:/GasProject/datasets/gas_data/data.yaml', # yaml 파일의 전체 경로
        epochs=50,       # 50번 반복학습
        imgsz=640,      # 이미지 크기
        batch=16,       # 컴퓨터 사양이 낮으면 8로 줄이세요
        device='cpu',    # CPU 사용 설정 (DLL 에러 방지)
        project='Gas_AI', 
        name='v1_nano' 
    )