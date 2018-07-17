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
from scipy import ndimage as ndi
from  skimage.drawskimage  import circle
import config as conf

### U-Net generator ###
class U_NET_BatchGenerator(Sequence):
    def __init__(self, images,
                       config,
                       jitter=True,
                       norm=None):
        self.generator = None

        self.images = [] # pairs of (img, mask)
        self.config = config

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
                    percent=(-0.05, 0.05),
                    pad_mode='reflect',
                    pad_cval=0
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-24, 24), # rotate by -45 to +45 degrees
                    shear=(-3, 3), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=0, # if mode is constant, use a cval between 0 and 255
                    mode='reflect' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.04))), 
            ],
            random_order=True
        )

        self.images = copy.deepcopy(images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))

    def __getitem__(self, idx):
        l_bound =  idx   * self.config['BATCH_SIZE']
        r_bound = (idx+1)* self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        x_batch  = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        y_batch  = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                # desired network output

        for instance_count, train_instance in enumerate(self.images[l_bound:r_bound]):
            # augment input image and fix object's position and size
            img, lab, marker, dt = self.aug_image(train_instance, jitter=self.jitter)

            x_batch[instance_count,...]    = self.norm(img)
            y0_batch[instance_count,...,0] = lab
            y1_batch[instance_count,...,1] = marker
            y2_batch[instance_count,...,2] = dt

        return x_batch, y_batch

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['image']
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)[...,:3]
        assert image is not None
        image = image[...,::-1] ## BGR -> RGB

        marker = np.zeros(image.shape[:2], dtype=np.bool)
        mask   = np.zeros(image.shape[:2], dtype=np.bool)
        dt     = np.zeros(image.shape[:2], dtype=np.float32)
        r = max( image.shape[0], image.shape[1] ) * .009
        for maskpath in train_instance['masks']:
            mask_  = (cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)>0).astype(np.bool)
            assert mask_ is not None
            dt_   = ndi.distance_transform_edt(mask_).astype(np.float32)
            cY, cX = np.unravel_index(np.argmax(dt_, axis=None), dt_.shape) # find local maximum of edt image
            cY = np.clip(cY, r, image.shape[0]-r)
            cX = np.clip(cX, r, image.shape[1]-r)
            dt_   = dt_ / np.max(dt_) # get a distance transform of an instance
            dt    = np.maximum(dt, dt_)
            rr, cc = circle(cY, cX, r, shape=image.shape[:2])
            marker[rr,cc] = True
            mask = np.maximum(mask, mask_)

        h, w, c = image.shape
        
        marker = marker.astype(np.float32)
        mask   = mask.astype(np.float32)

        if jitter:
            seq_det = self.aug_pipe.to_deterministic()
            image   = seq_det.augment_image(image)
            mask    = np.squeeze(seq_det.augment_image(np.expand_dims(mask,   -1)))
            marker  = np.squeeze(seq_det.augment_image(np.expand_dims(marker, -1)))
            dt      = np.squeeze(seq_det.augment_image(np.expand_dims(dt, -1)))

            # random amplify each channel
            a = .1 # amptitude
            t  = [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t += [np.random.uniform(-a,a)]
            t = np.array(t)

            image = image.astype(np.float32) / 255.
            image = np.clip(image * (1. + t), 0, 1) # channel wise amplify
            up = np.random.uniform(0.95, 1.05) # change gamma
            image = np.clip(image**up, 0, 1)
            # additive random noise
            sigma = np.random.rand()*0.03
            image = np.clip(image + np.random.randn(*image.shape)*sigma, 0, 1)
            image = np.clip(image * 255, 0, 255) # apply gamma and convert back to range [0,255]
            image = image.astype(np.uint8) # convert back to uint8
            if np.random.binomial(1, .05):
                ksize = np.random.choice([3,5,7])
                image = cv2.GaussianBlur(image, (ksize,ksize), 0)

        # resize the image to standard size
        inter   = cv2.INTER_LINEAR if (image.shape[0]<self.config['IMAGE_H'] or image.shape[1]<self.config['IMAGE_W']) else cv2.INTER_AREA
        image   = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']), interpolation=inter) # shape: (IMAGE_H, IMAGE_W, 3)
        mask    = (cv2.resize(np.squeeze(mask) , (self.config['IMAGE_H'], self.config['IMAGE_W']), interpolation=inter)>.5).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)
        marker  = (cv2.resize(np.squeeze(marker) , (self.config['IMAGE_H'], self.config['IMAGE_W']), interpolation=inter)>.5).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)
        dt      = cv2.resize(np.squeeze(dt) , (self.config['IMAGE_H'], self.config['IMAGE_W']), interpolation=inter).astype(np.float32) # shape: (IMAGE_H, IMAGE_W)

        return image, mask, marker, dt
### end U-Net generator ###
