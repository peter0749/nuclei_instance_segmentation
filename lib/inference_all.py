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
from utils import decode_netout, draw_dots, rle_encoding, prob_to_rles
import cv2

print('Loading trained weights...')

unet_model, _ = models.get_U_Net_model(gpus=1, load_weights=conf.U_NET_CKPT)
unet_model.summary()

print('Generating metadata...')
imgs_meta  = reader.dataset_filepath(conf.TEST_PATH, get_masks=False)
imgs_batch, imgs_path, imgs_shape = reader.dir_reader(imgs_meta, height=conf.U_NET_DIM, width=conf.U_NET_DIM)

preds = np.squeeze(unet_model.predict(imgs_batch, batch_size=conf.U_NET_BATCH_SIZE, verbose=1))

print('Resizing...')
preds_test_upsampled = []
for n in tqdm(range(len(imgs_batch)), total=len(imgs_batch)):
    nuclei = cv2.resize(preds[0][n], imgs_shape[n])
    marker = cv2.resize(preds[1][n], imgs_shape[n])
    preds_test_upsampled.append((nuclei, marker))

del imgs_batch, preds
gc.collect() # release memory

if not os.path.exists(conf.U_NET_OUT_DIR):
    os.makedirs(conf.U_NET_OUT_DIR)

print('Post-processing...')
rles = []
new_test_ids = []

for n, path in tqdm(enumerate(imgs_path), total=len(imgs_path)):
    _ , filename = os.path.split(path)
    id_ , _ = os.path.splitext(filename)
    rle = list(prob_to_rles(preds_test_upsampled[n][0], preds_test_upsampled[n][1]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(conf.SUBMISSION, index=False)
print('Done!')
