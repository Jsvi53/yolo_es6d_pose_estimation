import cv2
import numpy as np


def draw_bbox_on_image(img, vertices_2d):
    # 定义线条连接的顶点对
    lines = [[1, 3], [1, 4], [4, 6], [3, 6], [2, 8], [2, 7], [5, 8], [5, 7], [3, 8], [1, 2], [4, 7], [5, 6],]
    # 绘制边界框
    for line in lines:
        point1 = vertices_2d[line[0] - 1]  # -1 because lines are 1-indexed
        point2 = vertices_2d[line[1] - 1]
        cv2.line(img, tuple(point1), tuple(point2), (255, 0, 0), 2)  # 绘制蓝色边界框
    return img


def project_3d_to_2d(points_3d, K, Rt):
    # 将3D点转换为齐次坐标
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    # 投影到2D图像平面
    points_2d_homogeneous = K.dot(Rt).dot(points_3d_homogeneous.T)

    # 除以齐次坐标
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

    return points_2d.T.astype(np.int16)
