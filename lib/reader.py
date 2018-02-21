import config as conf
import glob
'''
An utility to read all filepaths of images and their masks
'''

def dataset_filepath(root):
    '''
    input: a full path to root of dataset
    output: a list of filepath of images and their masks
    '''
    root += '/*'
    dir_list = []
    for subdir in glob.glob(root):
        image = {'image': str(glob.glob(subdir+'/images/*.png')[0])}
        masks = list(glob.glob(subdir+'/masks/*.png'))
        image['masks'] = masks
        dir_list.append(image)
    return dir_list

if __name__ == '__main__':
    l = dataset_filepath('./stage1_train')
    print(l)

