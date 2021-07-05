# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:33:13 2021

@author: felix
"""

from torch.utils.data import Dataset


from torchvision.io import read_image
from torchvision.io import ImageReadMode

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
            