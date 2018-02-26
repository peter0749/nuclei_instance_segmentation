import os
import math
import copy
import tensorflow as tf
import copy
import cv2
import numpy as np
import config as conf
from keras.callbacks import Callback

class BoundBox:
    def __init__(self, x, y, c = None):
        self.x     = x
        self.y     = y
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

def draw_dots(image_, boxes, radius):
    image = copy.deepcopy(image_)
    for box in boxes:
        x, y = int(box.x*image.shape[1]), int(box.y*image.shape[0])
        cv2.circle(image, (x,y), radius, (0,255,0), -1) # Green
    return image

### modified version for binary classification
### ref. from https://github.com/experiencor/basic-yolo-keras/blob/master/utils.py
def decode_netout(netout, obj_threshold):
    grid_h, grid_w = netout.shape[:2]

    boxes = []

    # decode the output by the network
    netout[..., 2]  = sigmoid(netout[..., 2]) # confidence

    for row in range(grid_h):
        for col in range(grid_w):
            confidence = netout[row,col,2]
            if confidence > obj_threshold:
                # first 4 elements are x, y, w, and h
                x, y = netout[row,col,:2]

                x = (col + sigmoid(x)) / grid_w # center position, unit: image width
                y = (row + sigmoid(y)) / grid_h # center position, unit: image height

                box = BoundBox(x, y, confidence)

                boxes.append(box)

    return boxes

## ref: https://github.com/keras-team/keras/issues/8463
class multi_gpu_ckpt(Callback):
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(multi_gpu_ckpt, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

from scipy import ndimage as ndi
from skimage.morphology import watershed
# Run-length encoding from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
# reference from basic-yolo-keras (https://github.com/experiencor/basic-yolo-keras/blob/master/utils.py)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def lb(image, marker):
    if np.sum(image) < np.sum(marker):
        image = marker
    else:
        marker = np.array((marker==1) & (image==1))
    distance = ndi.distance_transform_edt(image)
    markers = ndi.label(marker)[0]
    labels = watershed(-distance, markers, mask=image)
    if np.sum(labels) == 0:
        labels[0,0] = 1
    return labels

def prob_to_rles(x, marker, cutoff=0.5, cutoff_marker=0.5):
    lab_img = lb(x > cutoff, marker > cutoff_marker)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def reset(self):
        self.offset = 4
