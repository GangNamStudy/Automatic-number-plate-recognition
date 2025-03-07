from ultralytics import YOLO
import torch

ckpt_path = "C:/Users/chang/Desktop/cv/gangnam/Automatic-number-plate-recognition/src/ckpt/yolo11n.pt"
model = YOLO(ckpt_path)

def train():
    torch.multiprocessing.freeze_support()
    model.train(data='LicensePlate.yaml', project="LicensePlateDetect", epochs=100, imgsz=640, device=0)

if __name__ =="__main__":
    train()
