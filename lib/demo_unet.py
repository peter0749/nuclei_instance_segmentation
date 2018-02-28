import os
import numpy as np
from tqdm import tqdm
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
from utils import normalize
from generators import U_NET_BatchGenerator
import cv2

print('Loading trained weights...')
unet_model, _ = models.get_U_Net_model(img_size=conf.U_NET_DIM, gpus=1, load_weights=conf.U_NET_CKPT)
unet_model.summary()

print('Generating metadata...')
train_imgs = reader.dataset_filepath(conf.DATA_PATH)

if not os.path.exists(conf.U_NET_OUT_DIR):
    os.makedirs(conf.U_NET_OUT_DIR)

imgs  = [] # xs
masks = [] # ys
names = []

print('Getting masks...')
for img in tqdm(train_imgs, total=len(train_imgs)):
    base_img = cv2.imread(img['image'], cv2.IMREAD_COLOR)[...,:3] # BGR channels
    base_img = base_img[...,::-1] # BGR -> RGB
    for mask in img['masks']:
        xmin, xmax, ymin, ymax = mask['xmin'], mask['xmax'], mask['ymin'], mask['ymax']
        crop_img = cv2.resize(base_img[ymin:ymax, xmin:xmax], (conf.U_NET_DIM, conf.U_NET_DIM))
        imgs.append(crop_img)
        real_region = np.squeeze(cv2.imread(mask['mask'], cv2.IMREAD_GRAYSCALE) > 0).astype(np.bool)
        real_region = real_region[ymin:ymax, xmin:xmax]
        masks.append(real_region)
        names.append(os.path.split(mask['mask'])[-1])

imgs = np.array(imgs).astype(np.float32) / 255. # normalize
print('data shape: %s'%(str(imgs.shape)))

preds = unet_model.predict(imgs, batch_size=conf.U_NET_BATCH_SIZE, verbose=1)

for i, pred in tqdm(enumerate(preds), total=len(preds)):
    mask = masks[i]
    name = names[i]
    pred = (cv2.resize(np.squeeze(pred), mask.shape[::-1]) > conf.U_NET_THRESHOLD).astype(np.bool)
    union= mask & pred
    fp   = pred & ~union
    fn   = mask & ~union
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[union, :3] = 255, 255, 255 # B, G, R
    result[fp   , :3] =   0,   0, 255 # Red
    result[fn   , :3] =   0, 255,   0 # Green
    cv2.imwrite(os.path.join(conf.U_NET_OUT_DIR, name), result)

