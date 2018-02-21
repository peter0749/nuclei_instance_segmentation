# ROOT TO DATASET (DATA & TEST)
DATA_PATH = '/home/peter/stage1_train' # this will split into training/validation
TEST_PATH = '/home/peter/stage1_test'
VALID_SPLIT = 0.1

# U-Net for semantic segmentation
U_NET_DIM = 64

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 416
YOLO_GRID= 13
YOLO_BOX = 5
OBJECT_THRESHOLD = 0.3 # <- notice here
NMS_THRESHOLD = 0.3
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50
BOX = 5

YOLO_BATCH_SIZE=8
U_NET_BATCH_SIZE=8

YOLO_EPOCHS=250
U_NET_EPOCHS=300

YOLO_CKPT = '/home/peter/yolo.h5'

YOLO_TFBOARD_DIR = '/home/peter/yolo_tfboard'
YOLO_OPT_ARGS = {
    'lr'              : 0.5e-4,
    'beta_1'          : 0.9,
    'beta_2'          : 0.999,
    'epsilon'         : 1e-08,
    'decay'           : 0.0
}
YOLO_EARLY_STOP = 20

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
