import os
import math
import glob
import cv2
import copy
import numpy as np
from keras.utils import Sequence
from utils import normalize
from reader import dataset_filepath
from scipy import ndimage as ndi
from  skimage.draw import circle
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
            y_batch[instance_count,...,0] = lab
            y_batch[instance_count,...,1] = marker
            y_batch[instance_count,...,2] = dt

        return x_batch, y_batch

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['image']
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)[...,:3]
        assert image is not None
        if np.max(image.shape[:2])>416 and np.random.rand()<.7:
            image = cv2.resize(image, (416,416), interpolation=cv2.INTER_AREA)
        image = image[...,::-1] ## BGR -> RGB

        marker = np.zeros(image.shape[:2], dtype=np.bool)
        mask   = np.zeros(image.shape[:2], dtype=np.bool)
        dt     = np.zeros(image.shape[:2], dtype=np.float32)
        r = max( image.shape[0], image.shape[1] ) * .009
        for maskpath in train_instance['masks']:
            mask_  = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
            if mask_.shape != mask.shape:
                mask_ = cv2.resize(mask_, mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
            mask_  = (mask_>0).astype(np.bool)
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
            if np.random.rand() < .5: # flip vertical
                image = image[::-1,...]
                mask  = mask[::-1,...]
                marker= marker[::-1,...]
                dt    = dt[::-1,...]
            if np.random.rand() < .5: # flip horizonal
                image = image[:,::-1,...]
                mask  = mask[:,::-1]
                marker= marker[:,::-1]
                dt    = dt[:,::-1]
            # rotation, shearing
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-30,30)
                cx = int(image.shape[1]//2)
                cy = int(image.shape[0]//2)
                M = cv2.getRotationMatrix2D((cx,cy),angle,1)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                (h, w) = image.shape[:2]
                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cx
                M[1, 2] += (nH / 2) - cy
                image = np.clip(cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC if np.random.rand()<.1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), 0, 255)
                mask = np.clip(cv2.warpAffine(mask, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), 0, 1)
                marker = np.clip(cv2.warpAffine(marker, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), 0, 1)
                dt = np.clip(cv2.warpAffine(dt, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101), 0, 1)
            if np.random.rand() < 0.3:
                crop_ratio = np.random.uniform(0.01, 0.05, size=4)
                u, r, d, l = np.round(crop_ratio * np.array([image.shape[0], image.shape[1]]*2)).astype(np.uint8)
                image   = image[u:-d,l:-r] # crop image
                mask    = mask[u:-d,l:-r] # crop image
                marker  = marker[u:-d,l:-r] # crop image
                dt      = dt[u:-d,l:-r] # crop image

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
