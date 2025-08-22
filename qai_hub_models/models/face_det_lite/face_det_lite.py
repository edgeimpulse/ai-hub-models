# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
# import torch
# import torch.nn.functional as F

# pulled from qai_hub_models.utils.bounding_box_processing import get_iou

def get_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Given two tensors of shape (4,) in xyxy format,
    compute the iou between the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return inter_area / float(boxA_area + boxB_area - inter_area)

class BBox:
    # Bounding Box
    def __init__(
        self,
        label: str,
        xyrb: list[int],
        score: float = 0,
        landmark: list | None = None,
        rotate: bool = False,
    ):
        """
        A bounding box plus landmarks structure to hold the hierarchical result.
        parameters:
            label:str the class label
            xyrb: 4 list for bbox left, top,  right bottom coordinates
            score:the score of the deteciton
            landmark: 10x2 the landmark of the joints [[x1,y1], [x2,y2]...]
        """
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate

        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = (
            ",".join([str(item[:2]) for item in self.landmark])
            if self.landmark is not None
            else "empty"
        )
        return (
            f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, "
            + f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"
        )

    @property
    def width(self) -> int:
        return self.r - self.x + 1

    @property
    def height(self) -> int:
        return self.b - self.y + 1

    @property
    def box(self) -> list[int]:
        return [self.x, self.y, self.r, self.b]

    @box.setter
    def box(self, newvalue: list[int]) -> None:
        self.x, self.y, self.r, self.b = newvalue

    @property
    def haslandmark(self) -> bool:
        return self.landmark is not None

    @property
    def xywh(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]


def nms(objs: list[BBox], iou: float = 0.5) -> list[BBox]:
    """
    nms function customized to work on the BBox objects list.
    parameter:
        objs: the list of the BBox objects.
    return:
        the rest of the BBox after nms operation.
    """
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            # if flags[j] == 0 and obj.iou(objs[j]) > iou:
            if (
                flags[j] == 0
                and get_iou(np.array(obj.box), np.array(objs[j].box)) > iou
            ):
                flags[j] = 1
    return keep


def max_pool2d(input: np.ndarray, kernel_size: int, stride: int, padding: int = 0, dilation: int = 1, ceil_mode: bool = False, return_indices: bool = False) -> np.ndarray:
    # Pad the input
    padded = np.pad(
        input,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant',
        constant_values=float('-inf')
    )

    N, C, H, W = padded.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    output = np.empty((N, C, out_h, out_w), dtype=input.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    h_end = h_start + kernel_size
                    w_start = j * stride
                    w_end = w_start + kernel_size
                    window = padded[n, c, h_start:h_end, w_start:w_end]
                    output[n, c, i, j] = np.max(window)
    return output

def numpy_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def detect(
    hm: np.ndarray,
    box: np.ndarray,
    landmark: np.ndarray,
    threshold: float = 0.2,
    nms_iou: float = 0.2,
    stride: int = 8,
) -> list[BBox]:
    hm = numpy_sigmoid(hm)
    hm_pool = max_pool2d(hm, 3, 1, 1)
    lens = ((hm == hm_pool).astype(float) * hm).reshape(1, -1).shape[1]
    print(f"lens: {lens}")
    # Get top-k scores and indices using numpy
    flat_scores = ((hm == hm_pool).astype(float) * hm).reshape(-1)
    print(f"flat_scores: {flat_scores}")
    k = min(lens, 2000)
    print(f"k: {k}")
    topk_indices = np.argpartition(flat_scores, -k)[-k:]
    topk_scores = flat_scores[topk_indices]
    # Sort top-k scores and indices in descending order
    sorted_order = np.argsort(-topk_scores)
    scores = topk_scores[sorted_order]
    indices = topk_indices[sorted_order]

    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list(np.divide(indices, hm_width).astype(int))
    xs = list((indices % hm_width).astype(int))
    scores = list(scores)

    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[0, :, cy, cx]
        xyrb: list[int] = (
            (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        ).tolist()
        x5y5 = landmark[0, :, cy, cx]
        x5y5 = (x5y5 + ([cx] * 5 + [cy] * 5)) * stride

        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(BBox("0", xyrb=xyrb, score=score, landmark=box_landmark))

    if nms_iou != -1:
        return nms(objs, iou=nms_iou)
    return objs

