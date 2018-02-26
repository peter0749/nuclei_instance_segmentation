import config as conf
import numpy as np
import tensorflow as tf
import keras.backend as K

def unet_loss(y_true, y_pred):
    from keras.losses import binary_crossentropy
    from metrics import dice_coef
    return .5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def yolo_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:3]

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(conf.YOLO_GRID), [conf.YOLO_GRID]), (1, conf.YOLO_GRID, conf.YOLO_GRID, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [conf.YOLO_BATCH_SIZE, 1, 1, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    """
    Adjust prediction
    """
    ### adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 2])

    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell

    true_box_conf = y_true[..., 2]

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 2], axis=-1) * conf.COORD_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 2] * conf.OBJECT_SCALE

    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))

    loss_xy    = tf.clip_by_value( .5 * tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-8) , conf.YOLO_MIN_LOSS, conf.YOLO_MAX_LOSS)
    loss_conf  = tf.clip_by_value( .5 * tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-8) , conf.YOLO_MIN_LOSS, conf.YOLO_MAX_LOSS)

    loss = loss_xy + loss_conf

    """
    Debugging code
    """

    loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)

    return loss

