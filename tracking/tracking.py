import torch
from ultralytics import YOLO

# print(torch.cuda.is_available())
if __name__ == '__main__':
    # import torch
    # Load a model
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # print(torch.__version__)
    # model = YOLO('yolov8n.pt', device='gpu')  # load a pretrained model (recommended for training)
    model = model.cuda()
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    dataset_path = "D:/junyoung/4-2/캡디2/Club-head-3"
    # Train the model
    results = model.train(data=f'{dataset_path}/data.yaml', epochs=50, imgsz=640)