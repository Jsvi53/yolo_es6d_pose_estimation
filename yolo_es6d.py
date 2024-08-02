import time
import open3d as o3d
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np
import os
print(os.path.abspath(__file__))
from utils import DataProcess, load_engine, \
    infer, project_3d_to_2d, draw_bbox_on_image, \
    prepare_data, pose_PostProcessing


def main():
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # global varibles
    K_list = [598.7568359375, 0.0, 326.3443298339844, 0.0, 598.7568969726562, 250.24488830566406, 0.0, 0.0, 1.0, ]
    K = np.array(K_list).reshape([3, 3])

    # 数据预处理
    dataProcessor = DataProcess()

    # 加载模型
    model = YOLO("models/yolo/best.engine", task="segment")
    engine = load_engine("models/es6d/es6d_simplified.engine")

    # 3Dbbox 数据准备
    pcd = o3d.io.read_point_cloud("datasets/green/models/1.pcd")
    bbox = pcd.get_oriented_bounding_box()
    vertices = np.array(bbox.get_box_points())
    num = 0
    while num < 1007:
        depth = np.array(Image.open(f"datasets/green/depth/{num}.png")).astype(np.float32)
        rgb = np.array(Image.open(f"datasets/green/JPEGImages/{num}.jpg"))
        print(num)
        # processed_image
        _, scale, pad, original_shape = dataProcessor.pre_process(rgb)
        # inference
        # yolo_results = model(rgb, device=0)
        yolo_results = model(rgb)
        if yolo_results is None or len(yolo_results) == 0:
            num += 1
            continue
        if yolo_results[0].masks is None or yolo_results[0].boxes is None:
            num += 1
            continue
        mask = yolo_results[0].masks[0].data[0].cpu().numpy()
        bbox = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(np.uint16)
        if not (mask.any() and bbox.any()):
            num += 1
            continue
        mask = dataProcessor.restore_to_original_shape(mask, scale, pad, original_shape)
        all_data = prepare_data(rgb, depth, mask, K_list, bbox)
        resized_xyz = all_data[1]

        # begin to inference
        start = time.time()
        data_pred = infer(engine, all_data)
        end = time.time()
        print("{0} POSE inferring time is {1}ms".format(num, int((end - start) * 1000)))

        if any(np.isnan(arr).any() for arr in data_pred):
            num += 1
            continue

        # Post-processing data
        results = [[] for _ in range(4)]
        for i, data in enumerate(data_pred):
            results[i] = data
        T = pose_PostProcessing(results, resized_xyz)
        if T is None:
            num += 1
            continue
        projected_vertices = project_3d_to_2d(vertices, K, T)
        img = draw_bbox_on_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), projected_vertices)
        cv2.imshow("bbox", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        num += 1


if __name__ == "__main__":
    main()
