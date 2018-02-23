# ROOT TO DATASET (DATA & TEST)
DATA_PATH = '/hdd/dataset/nuclei_dataset/stage1_train' # this will split into training/validation
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage1_test'
VALID_SPLIT = 65

# U-Net for semantic segmentation
U_NET_DIM = 64
U_NET_OUT_DIR = '/hdd/dataset/nuclei_dataset/unet_out'

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 416
YOLO_GRID= 13
YOLO_BOX = 5
OBJECT_THRESHOLD = 0.2 # <- notice here
NMS_THRESHOLD = 0.2
U_NET_THRESHOLD = 0.5
# ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50
BOX = 5

YOLO_DRAW_LINE_W = 1
YOLO_SHOW_CONF = False

YOLO_USE_MULTI_GPU=2
U_NET_USE_MULTI_GPU=2

YOLO_BATCH_SIZE=48 ## each gpus's batch size = YOLO_BATCH_SIZE / YOLO_USE_MULTI_GPU
U_NET_BATCH_SIZE=8

GENERATOR_WORKERS=4

YOLO_EPOCHS=500
U_NET_EPOCHS=300

YOLO_CKPT = '/hdd/dataset/nuclei_dataset/yolo.h5'
U_NET_CKPT = '/hdd/dataset/nuclei_dataset/unet.h5'

YOLO_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/yolo_tfboard'
U_NET_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/unet_tfboard'
YOLO_OPT_ARGS = {
    'lr'              : 2e-4,
    'clipvalue'       : 0.1 ,
    'clipnorm'        : 1.0 ,
}
U_NET_OPT_ARGS = {
    'lr'              : 1e-3,
}

YOLO_MIN_LOSS = 0
YOLO_MAX_LOSS = 10 # This prevent nans. If your loss is not chaning, then set a higher value.

YOLO_EARLY_STOP = 50
U_NET_EARLY_STOP = 50

YOLO_OUT_DIR = '/hdd/dataset/nuclei_dataset/detection_output'
U_NET_OUT_DIR = '/hdd/dataset/nuclei_dataset/detection_output'

### !!! DO NOT EDITING THE CONFIGURATION BELOW !!! ###

yolo_generator_config = {
    'IMAGE_H'         : YOLO_DIM,
    'IMAGE_W'         : YOLO_DIM,
    'GRID_H'          : YOLO_GRID,
    'GRID_W'          : YOLO_GRID,
    'BOX'             : BOX,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : YOLO_BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

unet_generator_config = {
    'IMAGE_H'         : U_NET_DIM,
    'IMAGE_W'         : U_NET_DIM,
    'BATCH_SIZE'      : U_NET_BATCH_SIZE,
}
