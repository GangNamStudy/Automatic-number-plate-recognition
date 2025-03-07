import cv2
import easyocr
import time
from ultralytics import YOLO

def predict(ckpt_path, image_path, inference_path, bbox_path):
    model = YOLO(ckpt_path)

    image = cv2.imread(image_path)
    result = model.predict(image_path)

    #yolo image bbox는 중심점 x,y 사각형의 너비높이 w,h
    x,y,w,h = result[0].boxes.xywh[0].cpu().numpy()
    x,y,w,h = int(x),int(y),int(w),int(h)
    print(x,y,w,h)

    #object detection 결과 이미지 저장
    image = cv2.rectangle(image, (x-w//2,y-h//2), (x+w//2, y+h//2), (0,255,0))
    cv2.imwrite(inference_path, image)

    #bbox 이미지 저장
    bbox_img = image[y-h//2-10:y+h//2+10,x-w//2-10:x+w//2+10]
    cv2.imwrite(bbox_path, bbox_img)

def EasyOCR(bbox_path):
    reader = easyocr.Reader(['ko'])
    result = reader.readtext(bbox_path)
    result = ' '.join(t[1] for t in result)
    print(result)


if __name__ == '__main__':
    ckpt_path = "C:/Users/chang/Desktop/cv/gangnam/Automatic-number-plate-recognition/resources/LicensePlateDetect/train/weights/best.pt"
    image_path = "C:/Users/chang/Desktop/cv/gangnam/Automatic-number-plate-recognition/resources/inference_result/inference_data/bus_plate.png"
    inference_path = "C:/Users/chang/Desktop/cv/gangnam/Automatic-number-plate-recognition/resources/inference_result/result/bus_plate_detected.png"
    bbox_path = "C:/Users/chang/Desktop/cv/gangnam/Automatic-number-plate-recognition/resources/inference_result/plate_image/bux_plate_bbox.png"
    t1= time.time()
    predict(ckpt_path, image_path, inference_path, bbox_path)
    t2= time.time()
    EasyOCR(bbox_path)
    t3= time.time()
    print("전체 걸린 시간 :", t3-t1)
    print("predict 걸린 시간 :", t2-t1)
    print("paddleOCR 걸린 시간 :", t3-t2)