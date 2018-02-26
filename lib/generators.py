import os
import math
import glob
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
from utils import BoundBox, normalize
from reader import dataset_filepath
from scipy import ndimage
import config as conf

### U-Net generator ###
class U_NET_BatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       shuffle=True,
                       jitter=True,
                       norm=None):
        self.generator = None

        self.images = [] # pairs of (img, mask)
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode='reflect',
                    pad_cval=0
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=0, # if mode is constant, use a cval between 0 and 255
                    mode='reflect' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), 
            ],
            random_order=True
        )

        self.images = copy.deepcopy(images)

        if self.shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        y0_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))                # desired network output
        y1_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))                # desired network output

        for instance_count, train_instance in enumerate(self.images[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img, lab, marker = self.aug_image(train_instance, jitter=self.jitter)

            x_batch[instance_count,...]   = self.norm(img)
            y0_batch[instance_count,...,0] = lab
            y1_batch[instance_count,...,0] = marker

        return x_batch, [y0_batch, y1_batch]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['image']
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)[...,:3]
        assert image is not None
        image = image[...,::-1] ## BGR -> RGB

        marker = np.zeros(image.shape[:2], dtype=np.uint8)
        mask   = np.zeros(image.shape[:2], dtype=np.uint8)
        for maskpath in train_instance['masks']:
            mask_  = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
            assert mask_ is not None
            col, row = list(map(int, list(ndimage.measurements.center_of_mass(mask_)))) # col, row
            cv2.circle(marker, (row, col), int(math.ceil(np.max(mask_.shape)*0.01)), 255, -1)
            mask = np.maximum(mask, mask_)

        h, w, c = image.shape

        if jitter:
            seq_det = self.aug_pipe.to_deterministic()
            image   = seq_det.augment_image(image)
            mask    = np.squeeze(seq_det.augment_image(np.expand_dims(mask,   -1)))
            marker  = np.squeeze(seq_det.augment_image(np.expand_dims(marker, -1)))

            # random amplify each channel
            a = .1 # amptitude
            t  = [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t = np.array(t)

            image = image.astype(np.float32) / 255.
            image = np.clip(image * (1. + t), 0, 1) # channel wise amplify
            up = np.random.uniform(0.95, 1.05) # change gamma
            image = np.clip(image**up * 255., 0, 255) # apply gamma and convert back to range [0,255]
            image = image.astype(np.uint8) # convert back to uint8

        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W'])) # shape: (IMAGE_H, IMAGE_W, 3)
        mask  = (cv2.resize(np.squeeze(mask) , (self.config['IMAGE_H'], self.config['IMAGE_W']))>128).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)
        marker  = (cv2.resize(np.squeeze(marker) , (self.config['IMAGE_H'], self.config['IMAGE_W']))>128).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)

        return image, mask, marker
### end U-Net generator ###
