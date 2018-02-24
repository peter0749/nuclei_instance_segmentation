# ROOT TO DATASET (DATA & TEST)
DATA_PATH = '/hdd/dataset/nuclei_dataset/stage1_train' # this will split into training/validation
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage1_test'
SUBMISSION= '/hdd/home/peter0749/nuclei_instance_segmentation/submission.csv'
VALID_SPLIT = 0.1

# U-Net for semantic segmentation
U_NET_DIM = 64

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 608 ## must be integer (odd number) * 32. 
OBJECT_THRESHOLD = 0.3 # <- notice here
NMS_THRESHOLD = 0.1 # less overlapping
U_NET_THRESHOLD = 0.6
ANCHORS = [0.29,0.42, 0.72,1.23, 0.73,0.72, 1.20,1.07, 1.61,1.86, 2.97,3.00] # from gen_anchorbox.py, 6 anchorboxes
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 8.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50

YOLO_DRAW_LINE_W = 1
YOLO_SHOW_CONF = False

YOLO_USE_MULTI_GPU=2
U_NET_USE_MULTI_GPU=2

YOLO_BATCH_SIZE=20 ## each gpus's batch size = YOLO_BATCH_SIZE / YOLO_USE_MULTI_GPU
U_NET_BATCH_SIZE=32

GENERATOR_WORKERS=5

YOLO_EPOCHS=400
U_NET_EPOCHS=200

YOLO_CKPT = '/hdd/dataset/nuclei_dataset/yolo.h5'
YOLO_PRETRAINED = '/hdd/dataset/nuclei_dataset/yolo.weights'

U_NET_CKPT = '/hdd/dataset/nuclei_dataset/unet.h5'

YOLO_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/yolo_tfboard'
U_NET_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/unet_tfboard'
YOLO_OPT_ARGS = {
    'lr'              : 1e-5,
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
U_NET_OUT_DIR = '/hdd/dataset/nuclei_dataset/unet_out'

### !!! DO NOT EDIT THE CONFIGURATION BELOW !!! ###

BOX = int(len(ANCHORS) // 2) # number of anchorboxes, default:5 
YOLO_GRID= int(YOLO_DIM // 32)  # 19
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
