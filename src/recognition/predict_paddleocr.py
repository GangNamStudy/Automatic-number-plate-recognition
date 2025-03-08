import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO
import time
import os

def predict(ckpt_path, image_path, inference_path, bbox_path):
    model = YOLO(ckpt_path)

    image = cv2.imread(image_path)
    result = model.predict(image_path)

    # YOLO 이미지 bbox는 중심점 (x, y)와 사각형의 너비, 높이 (w, h)
    x, y, w, h = result[0].boxes.xywh[0].cpu().numpy()
    x, y, w, h = int(x), int(y), int(w), int(h)
    print(x, y, w, h)

    # object detection 결과 이미지에 사각형 그리기
    image = cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0))
    cv2.imwrite(inference_path, image)

    # bbox 이미지 저장 (좌표에 여유 10픽셀 추가)
    bbox_img = image[y - h // 2 - 10 : y + h // 2 + 10, x - w // 2 - 10 : x + w // 2 + 10]
    cv2.imwrite(bbox_path, bbox_img)

def extract_text(bbox_path):
    ocr = PaddleOCR(lang="korean")
    result = ocr.ocr(bbox_path, cls=False)
    print("인식된 문자 출력" , result)


if __name__ == '__main__':
    # 상대경로 사용 (현재 작업 디렉토리 기준)
    ckpt_path = "./resources/LicensePlateDetect/train/weights/best.pt"
    image_path = "./resources/inference_result/inference_data/bus_plate.png"
    inference_path = "./resources/inference_result/result/bus_plate_detected.png"
    bbox_path = "./resources/inference_result/plate_image/bux_plate_bbox.png"

    # 폴더 생성 (중복 검사 없이, 이미 존재해도 에러 발생하지 않음)
    os.makedirs(os.path.dirname(inference_path), exist_ok=True)
    os.makedirs(os.path.dirname(bbox_path), exist_ok=True)

    t1 = time.time()
    predict(ckpt_path, image_path, inference_path, bbox_path)
    t2 = time.time()
    extract_text(bbox_path)
    t3 = time.time()

    print("전체 걸린 시간 :", t3 - t1)
    print("predict 걸린 시간 :", t2 - t1)
    print("PaddleOCR 걸린 시간 :", t3 - t2)
