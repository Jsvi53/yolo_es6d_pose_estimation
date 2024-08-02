from ultralytics import YOLO

model = YOLO("best.pt")
# out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=1, workspace=2, int8=True, data="coco8.yaml")
# out = model.export(format="engine", imgsz=640, dynamic=True, verbose=False, batch=1, workspace=2)
out = model.export(
    format="onnx", imgsz=640, dynamic=True, verbose=False, batch=1, workspace=2
)
