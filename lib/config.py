# U-Net for semantic segmentation
U_NET_DIM = 64

# YOLO step-by-step ref:
# https://github.com/experiencor/basic-yolo-keras/blob/master/Yolo%20Step-by-Step.ipynb
YOLO_DIM = 416
YOLO_GRID= 13
YOLO_BOX = 5
OBJECT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50
BOX = 5

YOLO_BATCH_SIZE=16
U_NET_BATCH_SIZE=8
