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
from utils import normalize, multi_gpu_ckpt
from generators import U_NET_BatchGenerator

unet_model, base_model = models.get_U_Net_model(gpus=conf.U_NET_USE_MULTI_GPU, load_weights=conf.U_NET_CKPT, verbose=True)
unet_model.summary()

print('Generating metadata...')
train_imgs = reader.dataset_filepath(conf.DATA_PATH)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=conf.VALID_SPLIT, shuffle=True)

train_batch = U_NET_BatchGenerator(train_imgs, conf.unet_generator_config, shuffle=True, jitter=True, norm=normalize) # shuffle and aug
valid_batch = U_NET_BatchGenerator(val_imgs, conf.unet_generator_config, shuffle=False, jitter=False, norm=normalize) # not shuffle and not aug

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.0001,
                           patience=conf.U_NET_EARLY_STOP,
                           mode='min',
                           verbose=1)

checkpoint = multi_gpu_ckpt( conf.U_NET_CKPT,
                             base_model,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min',
                             period=1)

tensorboard = TensorBoard(log_dir=conf.U_NET_TFBOARD_DIR,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

unet_model.fit_generator(generator        = train_batch,
                    steps_per_epoch  = len(train_batch),
                    epochs           = conf.U_NET_EPOCHS,
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard],
                    max_queue_size   = 3,
                    shuffle          = True,
                    workers = conf.GENERATOR_WORKERS)

