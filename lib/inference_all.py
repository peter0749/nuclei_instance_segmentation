import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import models
import reader
import config as conf
from sklearn.model_selection import train_test_split
from generators import YOLO_BatchGenerator
from utils import decode_netout, draw_boxes, rle_encoding, get_rles
import cv2

if conf.USE_U_YOLO_PRED:
    YOLO_CKPT = conf.U_YOLO_CKPT
    YOLO_OUT_DIR = conf.U_YOLO_OUT_DIR
    YOLO_BATCH_SIZE = conf.U_YOLO_BATCH_SIZE
    ANCHORS = conf.U_YOLO_ANCHORS
else:
    YOLO_CKPT = conf.YOLO_CKPT
    YOLO_OUT_DIR = conf.YOLO_OUT_DIR
    YOLO_BATCH_SIZE = conf.YOLO_BATCH_SIZE
    ANCHORS = conf.ANCHORS

print('Loading trained weights...')
yolo_model, _ = models.get_yolo_model(gpus=1, load_weights=conf.YOLO_CKPT)
yolo_model.summary()

unet_model, _ = models.get_U_Net_model(gpus=1, load_weights=conf.U_NET_CKPT)
unet_model.summary()

print('Generating metadata...')
imgs_meta  = reader.dataset_filepath(conf.TEST_PATH, get_masks=False)
imgs_batch, imgs_original, imgs_path = reader.dir_reader(imgs_meta)

netouts = yolo_model.predict([imgs_batch, np.zeros((len(imgs_batch), 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))], batch_size=conf.YOLO_BATCH_SIZE, verbose=1)
del imgs_batch
gc.collect() # release memory

if not os.path.exists(conf.U_NET_OUT_DIR):
    os.makedirs(conf.U_NET_OUT_DIR)
if not os.path.exists(conf.YOLO_OUT_DIR):
    os.makedirs(conf.YOLO_OUT_DIR)

rles = []
new_test_ids = []

for i, netout in tqdm(enumerate(netouts), total=len(netouts)):
    boxes = decode_netout(netout,
                      obj_threshold=conf.OBJECT_THRESHOLD,
                      nms_threshold=conf.NMS_THRESHOLD,
                      anchors=conf.ANCHORS)
    image = copy.deepcopy(imgs_original[i]) # get an image
    imgcrops = []
    regions  = []
    for box in boxes:
        xmax  = np.clip(int((box.x + box.w/2) * image.shape[1]), 3, image.shape[1])
        xmin  = np.clip(int((box.x - box.w/2) * image.shape[1]), 0, xmax-3)
        ymax  = np.clip(int((box.y + box.h/2) * image.shape[0]), 3, image.shape[0])
        ymin  = np.clip(int((box.y - box.h/2) * image.shape[0]), 0, ymax-3)

        regions.append((xmin,xmax,ymin,ymax))
        imgcrops.append(cv2.resize(image[ymin:ymax, xmin:xmax], (conf.U_NET_DIM, conf.U_NET_DIM)))
    imgcrops = np.array(imgcrops, dtype=np.float32) / 255. # a batch of images
    preds = unet_model.predict(imgcrops, batch_size=conf.U_NET_BATCH_SIZE)

    _, filename = os.path.split(imgs_path[i])
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    for j in reversed(range(len(regions))): # j in increasing confidence order
        (xmin,xmax,ymin,ymax) = regions[j]
        mask = np.zeros(image.shape[:2], dtype=np.bool)
        resized_pred = cv2.resize(np.squeeze(preds[j]), (xmax-xmin, ymax-ymin))
        mask[ymin:ymax, xmin:xmax] = (resized_pred>conf.U_NET_THRESHOLD)
        labels[mask] = j+1
        image[mask, :3] = 255, 0, 0 # R, G, B
        #cv2.imwrite(os.path.join(conf.U_NET_OUT_DIR, filename+'_%d'%j), (mask*255.).astype(np.uint8))
    rle = list(get_rles(labels))
    if len(rle)==0:
        rle = [1,1]
    rles.extend(rle)
    fileid, _ = os.path.splitext(filename)
    new_test_ids.extend([fileid] * len(rle))

    image = draw_boxes(image, boxes)
    cv2.imwrite(os.path.join(conf.YOLO_OUT_DIR, filename), image.astype(np.uint8)[...,::-1]) # RGB -> BGR

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(conf.SUBMISSION, index=False)
