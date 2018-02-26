import config as conf
import numpy as np
import tensorflow as tf
import keras.backend as K

def unet_loss(y_true, y_pred):
    from keras.losses import binary_crossentropy
    from metrics import dice_coef
    return .5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


