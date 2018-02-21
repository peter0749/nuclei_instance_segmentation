import tensorflow as tf
import keras
import models
import reader
import config as conf
from sklearn.model_selection import train_test_split
from utils import normalize

yolo_model = models.get_yolo_model()
yolo_model.summary()

train_imgs = reader.dataset_filepath(conf.DATA_PATH)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=conf.VALID_SPLIT, shuffle=True)

train_batch = BatchGenerator(train_imgs, conf.generator_config, norm=normalize)
valid_batch = BatchGenerator(val_imgs, conf.generator_config, norm=normalize)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=3,
                           mode='min',
                           verbose=1)

checkpoint = ModelCheckpoint('weights_nuclei.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=1)

tensorboard = TensorBoard(log_dir=conf.YOLO_TFBOARD_DIR,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

model.fit_generator(generator        = train_batch,
                    steps_per_epoch  = len(train_batch),
                    epochs           = 100,
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard],
                    max_queue_size   = 3)

