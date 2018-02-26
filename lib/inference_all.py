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
from utils import decode_netout, draw_dots, rle_encoding, prob_to_rles
import cv2

print('Loading trained weights...')
HEIGHT, WIDTH = conf.YOLO_DIM, conf.YOLO_DIM
yolo_model, _ = models.get_yolo_model(gpus=1, load_weights=conf.YOLO_CKPT)

yolo_model.summary()

unet_model, _ = models.get_U_Net_model(gpus=1, load_weights=conf.U_NET_CKPT)
unet_model.summary()

print('Generating metadata...')
imgs_meta  = reader.dataset_filepath(conf.TEST_PATH, get_masks=False)
imgs_batch, imgs_u_net, imgs_path = reader.dir_reader(imgs_meta, height=HEIGHT, width=WIDTH)

for n, img in enumerate(imgs_u_net):
    imgs_u_net[n] = cv2.resize(img, (conf.U_NET_DIM, conf.U_NET_DIM))
imgs_u_net = np.array(imgs_u_net)

netouts = yolo_model.predict(imgs_batch, batch_size=conf.YOLO_BATCH_SIZE, verbose=1)
preds   = np.squeeze(unet_model.predict(imgs_u_net, batch_size=conf.U_NET_BATCH_SIZE, verbose=1))

del imgs_batch, imgs_u_net
gc.collect() # release memory

if not os.path.exists(conf.U_NET_OUT_DIR):
    os.makedirs(conf.U_NET_OUT_DIR)
if not os.path.exists(conf.YOLO_OUT_DIR):
    os.makedirs(conf.YOLO_OUT_DIR)

rles = []
new_test_ids = []

for i, netout in tqdm(enumerate(netouts), total=len(netouts)):
    boxes = decode_netout(netout,
                      obj_threshold=conf.OBJECT_THRESHOLD)
    pred = preds[i]
    marker = np.zeros_like(pred, dtype=np.uint32)
    h, w = marker.shape
    for j, box in enumerate(boxes):
        px, py = int(box.x//w), int(box.y//h)
        marker[py,px] = j+1 # set watershed marker

    _, filename = os.path.split(imgs_path[i])

    rle = list(prob_to_rles(pred, marker))
    rles.extend(rle)
    fileid, _ = os.path.splitext(filename)
    new_test_ids.extend([fileid] * len(rle))

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(conf.SUBMISSION, index=False)
