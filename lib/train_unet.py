import os
import gc
import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=tfconfig))
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
import keras.backend as K
import models
import reader
import config as conf
from sklearn.model_selection import train_test_split
from utils import normalize, multi_gpu_ckpt
from generators import U_NET_BatchGenerator

SCALES = [64, 96, 128, 160, 192, 224, 256] # different scales
LAST_CKPT_PATH = os.path.join(conf.U_NET_CKPT, 'last.hdf5')
CKPT_PATH = os.path.join(conf.U_NET_CKPT, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

print('Generating metadata...')
train_imgs = reader.dataset_filepath(conf.DATA_PATH)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=conf.VALID_SPLIT, shuffle=True)

if not os.path.exists(conf.U_NET_CKPT):
    os.makedirs(conf.U_NET_CKPT)

print('Begin to train U-Net model')

for EPOCH in range(0, conf.U_NET_EPOCHS, conf.U_NET_CH_DIM_EPOCHS):
    U_NET_GENERATOR_CONF = conf.unet_generator_config
    img_size = np.random.choice(SCALES)
    U_NET_GENERATOR_CONF['IMAGE_H'] = U_NET_GENERATOR_CONF['IMAGE_W'] = img_size

    unet_model, base_model = models.get_U_Net_model(img_size=img_size, gpus=conf.U_NET_USE_MULTI_GPU, load_weights=LAST_CKPT_PATH, verbose=True)

    checkpoint = multi_gpu_ckpt( CKPT_PATH,
                             base_model,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='min',
                             period=1)

    last_checkpoint = multi_gpu_ckpt( LAST_CKPT_PATH,
                             base_model,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='min',
                             period=1)

    csv_logger = CSVLogger('u_net_training.log', append=True)

    train_batch = U_NET_BatchGenerator(train_imgs, U_NET_GENERATOR_CONF, shuffle=True, jitter=True, norm=normalize) # shuffle and aug
    valid_batch = U_NET_BatchGenerator(val_imgs, U_NET_GENERATOR_CONF, shuffle=False, jitter=False, norm=normalize) # not shuffle and not aug

    end_epoch = min(conf.U_NET_EPOCHS, EPOCH+conf.U_NET_CH_DIM_EPOCHS)

    unet_model.fit_generator(generator        = train_batch,
                        steps_per_epoch  = len(train_batch),
                        epochs           = end_epoch,
                        verbose          = 1,
                        validation_data  = valid_batch,
                        validation_steps = len(valid_batch),
                        callbacks        = [checkpoint, last_checkpoint, csv_logger],
                        max_queue_size   = 3, 
                        workers = conf.GENERATOR_WORKERS,
                        initial_epoch = EPOCH
                        )
    K.clear_session()
    gc.collect()

