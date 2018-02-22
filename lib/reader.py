import config as conf
import numpy as np
import glob
from get_image_size import get_image_size
from skimage.measure import regionprops
from skimage.io import imread
'''
An utility to read all filepaths of images and their masks
'''

def dataset_filepath(root, get_masks=True):
    '''
    input: a full path to root of dataset
    output: a list of filepath of images and their masks
    '''
    root += '/*'
    dir_list = []
    for subdir in glob.glob(root):
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

if __name__ == '__main__':
    l = dataset_filepath('./stage1_train')
    print(l)

