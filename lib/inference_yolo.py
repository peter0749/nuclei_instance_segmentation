import os
import numpy as np
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
from utils import decode_netout, draw_boxes
import cv2

print('Loading trained weights...')
yolo_model, _ = models.get_yolo_model(gpus=1, load_weights=conf.YOLO_CKPT)
yolo_model.summary()

print('Generating metadata...')
imgs_meta  = reader.dataset_filepath(conf.TEST_PATH, get_masks=False)
imgs_batch, imgs_original, imgs_path = reader.dir_reader(imgs_meta)

netouts = yolo_model.predict([imgs_batch, np.zeros((len(imgs_batch), 1, 1, 1, conf.TRUE_BOX_BUFFER, 4))], batch_size=conf.YOLO_BATCH_SIZE, verbose=1)
del imgs_batch
gc.collect() # release memory

if not os.path.exists(conf.YOLO_OUT_DIR):
    os.makedirs(conf.YOLO_OUT_DIR)

for i, netout in tqdm(enumerate(netouts), total=len(netouts)):
    boxes = decode_netout(netout,
                      obj_threshold=conf.OBJECT_THRESHOLD,
                      nms_threshold=conf.NMS_THRESHOLD,
                      anchors=conf.ANCHORS)
    image = draw_boxes(imgs_original[i], boxes)[...,::-1] # RGB -> BGR
    _ , filename = os.path.split(imgs_path[i])
    newpath = os.path.join(conf.YOLO_OUT_DIR , filename)
    cv2.imwrite(newpath, image)

