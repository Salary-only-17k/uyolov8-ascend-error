from ultralytics import YOLO
import torch_npu
from torch_npu.npu import amp # 导入AMP模块
from torch_npu.contrib import transfer_to_npu    # 使能自动迁移
def train():
    model = YOLO("yolov8n.pt")
    model.replace()
    model.save("tmp.pt",use_dill=True)
    # results = model.train(data="coco128.yaml", epochs=2, imgsz=640, device="npu:0",amp=True,workers=0)
    
train()