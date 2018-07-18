# ROOT TO DATASET (DATA & TEST)
TRAIN_PATH = '/hdd/dataset/nuclei_dataset/stage1_train'
VALID_PATH = '/hdd/dataset/nuclei_dataset/stage1_valid'
TEST_PATH = '/hdd/dataset/nuclei_dataset/stage2_test'
SUBMISSION= '/hdd/home/peter0749/nuclei_instance_segmentation/submission.csv'

# U-Net for semantic segmentation
U_NET_DIM = 256

MARKER_W = 8

U_NET_THRESHOLD = 0.5
U_NET_THRESHOLD_MARKER = 0.5

U_NET_USE_MULTI_GPU=2

U_NET_BATCH_SIZE=16

GENERATOR_WORKERS=7

U_NET_EPOCHS=400

U_NET_CKPT = '/hdd/dataset/nuclei_dataset/unet.h5'

U_NET_TFBOARD_DIR = '/hdd/dataset/nuclei_dataset/unet_tfboard'
U_NET_OPT_ARGS = {
    'lr'              : 1e-3,
}

U_NET_EARLY_STOP = 30

U_NET_OUT_DIR = '/hdd/dataset/nuclei_dataset/unet_out'

### !!! DO NOT EDIT THE CONFIGURATION BELOW !!! ###

unet_generator_config = {
    'IMAGE_H'         : U_NET_DIM,
    'IMAGE_W'         : U_NET_DIM,
    'BATCH_SIZE'      : U_NET_BATCH_SIZE,
}
