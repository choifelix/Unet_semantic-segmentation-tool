# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:24:15 2021

@author: felix choi
"""

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split 

import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path

from src.dataset import *


            
        






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
    data_df.to_csv('H:/Projet/ia/blurry_background/data/protrait.csv', index=False)

    return data_df


def create_train_test(option, transform, params, split=0.2):
    """
    create a train and test dataLoader from the images directories and csv file. 
    train and test are separated regarding the split proportion ([0,1])
    
    input :
        - option - argparser, retrieve image directories and csv file for image Ids
        - transform - pytorch transform object, transform the images (used in the pytorch Dataset class )
        - params - python dict, parameters for the dataloader surch as batch sizes, etc. (check pytorch documentation for more)
        - split - float, float between 0 and 1. spit proportion between train and test.

    """
    clip_im_dir = option.clip_im_dir
    matting_dir = option.matting_dir
    csv_path    = option.csv_path
    
    print("create datasets")
    
    
    data_df = pd.read_csv(csv_path)
    # data_df = MergeDataframe(clip_im_dir, matting_dir)
    
    #separate data in training and test data (20/80)
    train_df, test_df = train_test_split(data_df, test_size=split)
    
    #search right Dataset class
    package_dir = Path(src.dataset.__file__).resolve().parent

    for (_, module_name, _) in iter_modules([package_dir]):
        # print(module_name, self.ComType)
        if option.dataset.lower() == module_name.lower() :
            modelModule = importlib.import_module("."+module_name)
            break
    
    # train data
    training_set = modelModule(train_df, clip_im_dir, matting_dir, transform, transform)
    train_loader = DataLoader(training_set, **params)
    
    
    #test data
    testing_set = modelModule(test_df, clip_im_dir, matting_dir, transform, transform)
    test_loader = DataLoader(testing_set, **params)
    
    return train_loader, test_loader