from ultralytics import YOLO
from multiprocessing import freeze_support
PATH = 'D:/junyoung/4-2/캡디2/runs/detect/train2/weights'
# Load the YOLOv8 model
model = YOLO(f'{PATH}/best.pt')
if __name__=="__main__": 
    freeze_support()
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category