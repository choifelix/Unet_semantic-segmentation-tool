# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:24:15 2021

@author: felix choi
"""

from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.io import ImageReadMode

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

class PortraitData(Dataset):
    """
    pytorch dataset class for the matting human dataset
    dataset will be saved as pathfile data and 
    actual images will be loaded during the training and the testing
    """
    def __init__(self, data, clip_im_dir, matting_dir, transform=None, target_transform=None):

        self.clip_im_dir = clip_im_dir
        self.matting_dir = matting_dir

        self.data_df = data

        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data_df)


    def __getitem__(self, idx):
        
        clip_path = self.data_df.iloc[idx, 0]
        matting_path = self.data_df.iloc[idx, 1]
        
        try :
            
            clip_img  = read_image(clip_path, mode=ImageReadMode.RGB)
    
            matting_img  = read_image(matting_path, mode = ImageReadMode.RGB_ALPHA)
    
    
            if self.transform:
                clip_img    = self.transform(clip_img)
                matting_img = self.target_transform(matting_img)
                
            matting_img = matting_img[-1,:,:] == 0
            matting_img = matting_img.int()
            
            return clip_img/255, matting_img
        
        except RuntimeError as err:
            print(err)
            print('Runtime error on image :{}\n {}'.format(clip_path, matting_path))
            exit()
            
            
        






def MergeClipData(clip_im_dir):
    """
    merge all clip data as one dictionary (1 column)

    output:
        image_list_dict - python dict with all images paths
    """

    image_list_dict = {'image_id':[]} 

    clip_img = os.listdir(clip_im_dir)


    for folder_clip in clip_img :

        clip_path = os.path.join(clip_im_dir, folder_clip)
        # clip_path = clip_im_dir + '/' + folder_clip
        clip_list = os.listdir(clip_path)

        
        for folder in clip_list:

            images_path = os.path.join(clip_path, folder)
            # images_path = clip_path + '/' + folder
            image_list  = os.listdir(images_path)

            image_list_dict['image_id'] += [os.path.join(images_path, name) for name in image_list]

    return image_list_dict


def MergeMateData(matting_dir):
    """
    merge all matting data as one dictionary (1 column)

    output:
        image_list_dict - python dict with all images paths
    """

    image_list_dict = {'image_id':[]} 

    clip_img = os.listdir(matting_dir)

    for folder_clip in clip_img :

        clip_path = os.path.join(matting_dir, folder_clip)
        # clip_path = matting_dir + '/' + folder_clip
        clip_list = os.listdir(clip_path)

        
        for folder in clip_list:
            
            if folder != '._matting_00000000' :
                images_path = os.path.join(clip_path, folder)
                # images_path = clip_path + '/' + folder
                image_list  = os.listdir(images_path)
    
                image_list_dict['image_id'] += [os.path.join(images_path, name) for name in image_list]

    return image_list_dict

def MergeDataframe(clip_im_dir, matting_dir):
    """
    create a dataframe with clip and matting data in a single dataframe
    sava data as csv.

    input :
        clip_im_dir - clip image os path
        matting_dir - matting image os path
        save_data   - path to save merge dataframe as csv # removed 

    output :
    """

    clip_dict = MergeClipData(clip_im_dir)
    mate_dict = MergeMateData(matting_dir)

    clip_df = pd.DataFrame.from_dict(clip_dict)
    mate_df = pd.DataFrame.from_dict(mate_dict)

    data_df = pd.merge(clip_df, mate_df, left_index=True, right_index=True)

    # save
    # data_df.to_csv('save_data' + '.csv', index=False)

    return data_df