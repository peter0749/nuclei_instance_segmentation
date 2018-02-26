# ROOT TO DATASET (DATA & TEST)
DATA_PATH = '/hdd/dataset/nuclei_dataset/stage1_train' # this will split into training/validation
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage1_test'
SUBMISSION= '/hdd/home/peter0749/nuclei_instance_segmentation/submission.csv'
VALID_SPLIT = 0.1

# U-Net for semantic segmentation
U_NET_DIM = 352

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
U_NET_THRESHOLD = 0.5

U_NET_USE_MULTI_GPU=2

U_NET_BATCH_SIZE=16

GENERATOR_WORKERS=5

U_NET_EPOCHS=200

U_NET_CKPT = '/hdd/dataset/nuclei_dataset/unet.h5'

U_NET_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/unet_tfboard'
U_NET_OPT_ARGS = {
    'lr'              : 1e-3,
}

U_NET_EARLY_STOP = 50

U_NET_OUT_DIR = '/hdd/dataset/nuclei_dataset/unet_out'

### !!! DO NOT EDIT THE CONFIGURATION BELOW !!! ###

unet_generator_config = {
    'IMAGE_H'         : U_NET_DIM,
    'IMAGE_W'         : U_NET_DIM,
    'BATCH_SIZE'      : U_NET_BATCH_SIZE,
}
