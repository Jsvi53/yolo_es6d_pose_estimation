from typing import Tuple, List, Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


norm = transforms.Normalize(
    mean=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
    std=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
)


class DataProcess:
    def __init__(
        self,
        new_shape: Union[Tuple[int, int], List[int]] = (640, 640),
        color: Union[Tuple[int, int, int], List[int]] = (114, 114, 114),
        mean: Union[Tuple[float, float, float], List[float]] = (0.0, 0.0, 0.0),
        std: Union[Tuple[float, float, float], List[float]] = (1.0, 1.0, 1.0),
    ):
        """
        初始化DataProcess类的实例，用于图像预处理和后处理。

        参数:
            new_shape: 整数或一对整数，表示预处理后图像的新尺寸，默认为(640, 640)。
                       如果提供单个整数，则图像将被调整为该尺寸的正方形。

            color: 用于填充边缘的RGB颜色值，默认为(114, 114, 114)。

            mean: 用于图像归一化的RGB通道的均值，默认为(0.0, 0.0, 0.0)。

            std: 用于图像归一化的RGB通道的标准差，默认为(1.0, 1.0, 1.0)。
        """
        self.new_shape = new_shape
        self.color = color
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def pre_process(
        self, im: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
        """
        对图像进行预处理，包括调整大小、填充以及归一化。

        参数:
            im: np.ndarray, 输入的图像数组。

        返回:
            Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
            包含以下元素的元组：
            1. 预处理后的图像数组。
            2. 图像的缩放比例。
            3. 填充尺寸，格式为(width_padding, height_padding)。
            4. 图像原始的形状，格式为(original_height, original_width)。
        """
        original_shape = im.shape[:2]  # 保存原始形状 [高度, 宽度]
        shape = im.shape[:2]  # 当前形状 [高度, 宽度]

        # 确保new_shape是元组形式
        if isinstance(self.new_shape, int):
            new_shape = (self.new_shape, self.new_shape)
        else:
            new_shape = self.new_shape

        # 计算缩放比例
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        # 计算未填充前的新尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算填充量并平分到宽度和高度的两边
        dw, dh = (new_shape[0] - new_unpad[0]) / 2, (new_shape[1] - new_unpad[1]) / 2

        if shape != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 调整图像大小

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # 添加边框

        # 归一化处理
        # im = im.astype(np.float32) / 255.0  # 归一化到[0, 1]
        # im = (im - self.mean) / self.std  # 标准化
        return im, r, (dw, dh), original_shape

    def restore_to_original_shape(
        self,
        mask: np.ndarray,
        scale: float,
        pad: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        将处理过的掩膜恢复到预处理之前的形状。

        参数:
            mask: np.ndarray, 处理过的掩膜数组。
            scale: float, 掩膜的缩放比例。
            pad: Tuple[float, float], 掩膜的填充尺寸，格式为(width_padding, height_padding)。
            original_shape: Tuple[int, int], 掩膜原始的形状，格式为(original_height, original_width)。

        返回:
            np.ndarray: 恢复到原始形状的掩膜数组。
        """
        # 根据缩放比例调整掩膜大小
        resized_mask = cv2.resize(mask, None, fx=1 / scale, fy=1 / scale)
        # 裁剪掩膜以恢复到原始形状
        top_crop = int(pad[1])
        left_crop = int(pad[0])

        restored_mask = resized_mask[top_crop: (top_crop + original_shape[0]),
                                     left_crop: left_crop + original_shape[1],]
        return restored_mask

    def restore_bboxes_to_original_shape(
        self,
        bboxes: np.ndarray,
        scale: float,
        pad: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        将边界框恢复到预处理之前的形状。

        参数:
            bboxes: np.ndarray, 处理过的边界框数组，格式为[x_min, y_min, x_max, y_max]。
            scale: float, 边界框的缩放比例。
            pad: Tuple[float, float], 边界框的填充尺寸，格式为(width_padding, height_padding)。
            original_shape: Tuple[int, int], 边界框原始的形状，格式为(original_height, original_width)。

        返回:
            np.ndarray: 恢复到原始形状的边界框数组。
        """
        # 根据缩放比例调整边界框大小
        scaled_bboxes = bboxes / np.array([scale, scale, scale, scale], dtype=np.float32)
        # 裁剪边界框以恢复到原始形状
        top_crop = int(pad[1] / 2)
        left_crop = int(pad[0] / 2)
        restored_bboxes = scaled_bboxes - np.array([left_crop, top_crop, left_crop, top_crop], dtype=np.float32)
        # 确保边界框坐标不超出原始图像尺寸
        restored_bboxes = np.clip(restored_bboxes, 0, 640)
        return restored_bboxes


def resize(rgb, xyz, mask, width=128, height=128):
    rgb = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    xyz = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(dim=0).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
    rgb = F.interpolate(rgb, size=(height, width), mode="bilinear").squeeze(dim=0).permute(1, 2, 0).contiguous()
    xyz = F.interpolate(xyz, size=(height, width), mode="nearest").squeeze(dim=0).permute(1, 2, 0).contiguous()
    mask = F.interpolate(mask, size=(height, width), mode="nearest").squeeze(dim=0).squeeze(dim=0)
    return rgb.numpy(), xyz.numpy(), mask.numpy()


def prepare_data(rgb, depth, mask, K, bbox):
    # const
    resize_img_width = 128
    resize_img_width = 128
    obj_radius = 0.09677225
    depth *= 0.001
    rows, cols = depth.shape
    ymap = np.array([[j for i in range(cols)] for j in range(rows)]).astype(np.float32)
    xmap = np.array([[i for i in range(cols)] for j in range(rows)]).astype(np.float32)
    cam_cx = K[2]
    cam_cy = K[5]
    cam_fx = K[0]
    cam_fy = K[4]
    cmin, rmin, cmax, rmax = bbox
    img_crop = rgb[rmin:rmax, cmin:cmax, :]
    mask_crop = mask[rmin:rmax, cmin:cmax]
    depth_crop = depth[rmin:rmax, cmin:cmax, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax, np.newaxis]
    ymap_masked = ymap[rmin:rmax, cmin:cmax, np.newaxis]

    pt2 = depth_crop
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy

    depth_xyz = np.concatenate((pt0, pt1, pt2), axis=2)
    depth_mask_xyz = depth_xyz * mask_crop[:, :, np.newaxis]
    choose = depth_mask_xyz[:, :, 2].flatten().nonzero()[0]

    mask_x = depth_xyz[:, :, 0].flatten()[choose][:, np.newaxis]  # 选择的mask的x坐标
    mask_y = depth_xyz[:, :, 1].flatten()[choose][:, np.newaxis]
    mask_z = depth_xyz[:, :, 2].flatten()[choose][:, np.newaxis]
    mask_xyz = np.concatenate((mask_x, mask_y, mask_z), axis=1)

    mean_xyz = mask_xyz.mean(axis=0)
    mean_xyz = mean_xyz.reshape((1, 1, 3))
    depth_xyz = (depth_xyz - mean_xyz) * mask_crop[:, :, np.newaxis]

    xyz = depth_xyz.astype(np.float32)
    mask_crop = mask_crop.astype("uint8")

    resized_rgb, resized_xyz, resized_mask = resize(
        img_crop, xyz, mask_crop, resize_img_width, resize_img_width
    )

    dis_xyz = np.sqrt(
        resized_xyz[:, :, 0] * resized_xyz[:, :, 0]
        + resized_xyz[:, :, 1] * resized_xyz[:, :, 1]
        + resized_xyz[:, :, 2] * resized_xyz[:, :, 2]
    )
    mask_xyz = np.where(dis_xyz > obj_radius, 0.0, 1.0).astype(np.float32)
    resized_xyz = resized_xyz * mask_xyz[:, :, np.newaxis]
    resized_mask = resized_mask * mask_xyz
    resized_xyz = resized_xyz / obj_radius
    cv2.imshow("mask1", (255 * resized_mask).astype(np.uint8))
    cv2.imshow("rgb1", resized_rgb.astype(np.uint8))
    cv2.imshow("xyz1", (255 * resized_xyz).astype(np.uint8))
    cv2.moveWindow("rgb1", 0, 0)
    cv2.moveWindow("mask1", 128, 0)
    cv2.moveWindow("xyz1", 256, 0)
    resized_rgb = torch.from_numpy(resized_rgb.astype(np.float32)).permute(2, 0, 1).contiguous()
    resized_xyz = torch.from_numpy(resized_xyz.astype(np.float32)).permute(2, 0, 1).contiguous()

    if resized_mask.sum() == 0.0:
        resized_mask = np.ones(resized_mask.shape, dtype=np.float32)

    resized_mask = torch.from_numpy(resized_mask.astype(np.float32)).unsqueeze(dim=0)
    resized_rgb = norm(resized_rgb)
    mean_xyz = mean_xyz.astype("float32")
    icls_id = torch.LongTensor([int(1) - 1])
    # resized_rgb torch.Size([3, 128, 128]), resized_xyz torch.Size([3, 128, 128]), icls_id torch.Size([1])
    all_data = np.array((resized_rgb.numpy(), resized_xyz.numpy(), icls_id.numpy()), dtype=object)
    return all_data
