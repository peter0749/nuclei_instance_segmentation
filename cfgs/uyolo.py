USE_U_YOLO_PRED = True

# ROOT TO DATASET (DATA & TEST)
DATA_PATH = '/hdd/dataset/nuclei_dataset/stage1_train' # this will split into training/validation
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage1_test'
SUBMISSION= '/hdd/home/peter0749/nuclei_instance_segmentation/submission.csv'
VALID_SPLIT = 0.1

# U-Net for semantic segmentation
U_NET_DIM = 64

U_YOLO_DIM = 416 ## must be integer (odd number) * 32
U_YOLO_ANCHORS = [0.46,0.77, 0.79,2.63, 1.12,1.18, 1.73,1.93, 1.86,2.93, 2.07,4.29, 2.52,1.04, 2.53,2.27, 2.88,3.22, 3.52,6.28, 3.98,4.08, 4.25,2.36, 5.38,5.12, 6.45,8.69, 8.98,6.45, 10.18,11.27]

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 608 ## must be integer (odd number) * 32. 
OBJECT_THRESHOLD = 0.3 # <- notice here
NMS_THRESHOLD = 0.3 # less overlapping
U_NET_THRESHOLD = 0.5
ANCHORS = [0.16,0.27, 0.25,0.91, 0.38,0.39, 0.51,0.64, 0.62,1.08, 0.71,1.64, 0.71,0.75, 0.85,0.36, 0.87,1.17, 0.93,0.84, 1.20,1.46, 1.24,1.02, 1.28,2.36, 1.46,0.64, 1.61,1.58, 1.99,2.06, 2.37,3.23, 2.51,1.27, 3.30,2.48, 3.76,4.22] 
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 0

U_YOLO_NO_OBJECT_SCALE  = 1.0
U_YOLO_OBJECT_SCALE     = 5.0
U_YOLO_COORD_SCALE      = 1.0
U_YOLO_WARM_UP_BATCHES  = 0

TRUE_BOX_BUFFER  = 50

YOLO_DRAW_LINE_W = 1
YOLO_SHOW_CONF = False

YOLO_USE_MULTI_GPU=1
U_YOLO_USE_MULTI_GPU=1
U_NET_USE_MULTI_GPU=1

YOLO_BATCH_SIZE=4 ## each gpus's batch size = YOLO_BATCH_SIZE / YOLO_USE_MULTI_GPU
U_YOLO_BATCH_SIZE=4 ## each gpus's batch size = YOLO_BATCH_SIZE / YOLO_USE_MULTI_GPU
U_NET_BATCH_SIZE=4

GENERATOR_WORKERS=5

YOLO_EPOCHS=400
U_NET_EPOCHS=200
U_YOLO_EPOCHS=300

YOLO_CKPT = '/hdd/dataset/nuclei_dataset/yolo.h5'
YOLO_PRETRAINED = None

U_YOLO_CKPT = '/hdd/dataset/nuclei_dataset/uyolo.h5'

U_NET_CKPT = '/hdd/dataset/nuclei_dataset/unet.h5'

YOLO_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/yolo_tfboard'
U_YOLO_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/uyolo_tfboard'
U_NET_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/unet_tfboard'
YOLO_OPT_ARGS = {
    'lr'              : 1e-5,
    'clipvalue'       : 0.1 ,
    'clipnorm'        : 1.0 ,
}
U_YOLO_OPT_ARGS = {
    'lr'              : 1e-3,
    'clipvalue'       : 0.1 ,
    'clipnorm'        : 1.0 ,
}
U_NET_OPT_ARGS = {
    'lr'              : 1e-3,
}

YOLO_MIN_LOSS = 0
YOLO_MAX_LOSS = 10 # This prevent nans. If your loss is not chaning, then set a higher value.

YOLO_EARLY_STOP = 50
U_YOLO_EARLY_STOP = 50
U_NET_EARLY_STOP = 50

YOLO_OUT_DIR =   '/hdd/dataset/nuclei_dataset/detection_output'
U_YOLO_OUT_DIR = '/hdd/dataset/nuclei_dataset/uyolo_detection_output'
U_NET_OUT_DIR =  '/hdd/dataset/nuclei_dataset/unet_out'

### !!! DO NOT EDIT THE CONFIGURATION BELOW !!! ###

BOX = int(len(ANCHORS) // 2) # number of anchorboxes, default:5 
U_YOLO_BOX = int(len(U_YOLO_ANCHORS) // 2) # number of anchorboxes, default:5 
YOLO_GRID= int(YOLO_DIM // 32)  # 19
U_YOLO_GRID= int(U_YOLO_DIM // 8)  

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

uyolo_generator_config = {
    'IMAGE_H'         : U_YOLO_DIM,
    'IMAGE_W'         : U_YOLO_DIM,
    'GRID_H'          : U_YOLO_GRID,
    'GRID_W'          : U_YOLO_GRID,
    'BOX'             : U_YOLO_BOX,
    'ANCHORS'         : U_YOLO_ANCHORS,
    'BATCH_SIZE'      : U_YOLO_BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

unet_generator_config = {
    'IMAGE_H'         : U_NET_DIM,
    'IMAGE_W'         : U_NET_DIM,
    'BATCH_SIZE'      : U_NET_BATCH_SIZE,
}
