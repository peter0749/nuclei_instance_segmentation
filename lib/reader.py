import config as conf
import numpy as np
import tqdm
import glob
from get_image_size import get_image_size
from skimage.measure import regionprops
from skimage.io import imread
import copy
import cv2
'''
An utility to read all filepaths of images and their masks
'''

def dataset_filepath(root, get_masks=True):
    '''
    input: a full path to root of dataset
    output: a list of filepath of images and their masks
    '''
    root += '/*'
    dir_list  = []
    path_list = glob.glob(root)
    for subdir in tqdm(path_list, total=len(path_list)):
        image = {'image': str(glob.glob(subdir+'/images/*.png')[0])}
        width, height = get_image_size(image['image'])
        image['width'] = width
        image['height']= height

        if get_masks:
            image['masks'] = []
            for mask_path in glob.glob(subdir+'/masks/*.png'):
                mask = imread(mask_path, as_grey=True)
                assert mask.ndim == 2
                (ymin, xmin, ymax, xmax) = regionprops((mask>0).astype(np.int32), cache=True)[0].bbox
                image['masks'].append({'mask': mask_path, 'ymin': ymin, 'xmin': xmin, 'ymax': ymax, 'xmax': xmax}) # get bounding box

        dir_list.append(image)
    return dir_list

def dir_reader(img_meta_wo_markers):
    l = len(img_meta_wo_markers)
    input_images = np.zeros((l,conf.YOLO_DIM, conf.YOLO_DIM, 3))
    original_images  = []
    image_filenames  = []
    for i, img_meta in tqdm(enumerate(img_meta_wo_markers), total=l):
        img = imread(img_meta['image'])[...,:3] # only BGR channels
        img = img[...,::-1] # BGR -> RGB
        img_input = cv2.resize(img, (conf.YOLO_DIM, conf.YOLO_DIM))
        img_input = img_input / 255.
        input_images[i,...] = img_input
        original_images.append(img)
        image_filenames.append(img_meta['image'])
    return input_images, original_images, image_filenames # return resized images and their original shapes, and orignal images

if __name__ == '__main__':
    l = dataset_filepath('./stage1_train')
    print(l)

