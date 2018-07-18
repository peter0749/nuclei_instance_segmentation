import os
import copy
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
from utils import rle_encoding, get_label, lb, label_to_rles
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mean_average import average_scores

print('Loading trained weights...')

unet_model = models.get_U_Net_model(gpus=1, load_weights=conf.U_NET_CKPT)[0]
unet_model.summary()

print('Generating metadata...')
imgs_meta  = reader.dataset_filepath(conf.VALID_PATH, get_masks=True)
imgs_batch, imgs_path, imgs_shape = reader.dir_reader(imgs_meta, height=conf.U_NET_DIM, width=conf.U_NET_DIM)

preds = np.squeeze(unet_model.predict(imgs_batch, batch_size=conf.U_NET_BATCH_SIZE, verbose=1))

print('Resizing...')
preds_test_upsampled = []
for n in tqdm(range(len(imgs_batch)), total=len(imgs_batch), ascii=True):
    nuclei = cv2.resize(preds[n,...,0], imgs_shape[n])
    marker = cv2.resize(preds[n,...,1], imgs_shape[n])
    dt     = cv2.resize(preds[n,...,2], imgs_shape[n])
    preds_test_upsampled.append((nuclei, marker, dt))

del imgs_batch, preds, dt
gc.collect() # release memory

print('Post-processing...')

labels = []
for n, path in tqdm(enumerate(imgs_path), total=len(imgs_path), ascii=True):
    label = get_label(*preds_test_upsampled[n])
    labels.append(label)

masks = []
for meta in tqdm(imgs_meta, total=len(imgs_meta), ascii=True):
    mask_paths = meta['masks']
    mask_4_1_image = []
    for mask_path in mask_paths:
        mask_bool = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)>0).astype(np.bool)
        mask_4_1_image.append(mask_bool)
    masks.append(mask_4_1_image)

print('Evaluating...')
thresholds = np.arange(0.5, 1.0, 0.05)
aps, ars = average_scores(masks, labels, thresholds)
for ap, ar, t in zip(aps, ars, thresholds):
    print('AP@%.2f = %.4f'%(t, ap))
    print('AR@%.2f = %.4f'%(t, ar))
    print('-'*5)
print('mAP@[.5:.95] = %.4f'%np.mean(aps))
print('mAR@[.5:.95] = %.4f'%np.mean(ars))
