import os
import copy
import tensorflow as tf
import copy
import cv2
import numpy as np

class BoundBox:
    def __init__(self, x, y, w, h, c = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        self.c     = c
    def get_score(self):
        return self.c

def normalize(image):
    return image.astype(np.float32) / 255.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2

    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union

def draw_boxes(image_, boxes):
    image = copy.deepcopy(image_)
    for box in boxes:
        xmin  = int((box.x - box.w/2) * image.shape[1])
        xmax  = int((box.x + box.w/2) * image.shape[1])
        ymin  = int((box.y - box.h/2) * image.shape[0])
        ymax  = int((box.y + box.h/2) * image.shape[0])

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image,
                    str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0,255,0), 2)
    return image

### modified version for binary classification
### ref. from https://github.com/experiencor/basic-yolo-keras/blob/master/utils.py
def decode_netout(netout, obj_threshold, nms_threshold, anchors):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4]  = sigmoid(netout[..., 4]) # confidence

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                confidence = netout[row,col,b,4]
                if confidence > obj_threshold:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height

                    box = BoundBox(x, y, w, h, confidence)

                    boxes.append(box)

    # suppress non-maximal boxes, NMS on boxes
    sorted_indices = list(reversed(np.argsort([box.c for box in boxes])))

    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]

        if boxes[index_i].c == 0:
            continue
        else:
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                    boxes[index_j].c = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes

