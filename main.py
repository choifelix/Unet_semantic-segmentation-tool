# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:27:14 2021

@author: felix
"""

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io import ImageReadMode

from sklearn.model_selection import train_test_split 

from dataHandle import PortraitData, MergeDataframe

from Unet_model import Unet_model

from utils import train, test, segmented_img

from argparse import *
import configparser 
import json
import os

import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(os.getcwd())
    
    # ================================================================================================================ #
    # ARGV CHECK                                                                                                       #
    # ================================================================================================================ #

    # Parser for the options
    parser = ArgumentParser()
    parser.add_argument("-config", "--config-file", type=str,default=None, help=" configurations for the interface, that includes running option but also specific parameters for custom models.")
    parser.add_argument("-train", "--train", action="store_true", default=None, help="Specify the training mode of the interface, make sure that the model is a learning one.")
    parser.add_argument("-ep", "--epoch", type=int, default=None, help="Number of epoch to train. default : 10")
    parser.add_argument("-ws", "--width-size", type=int, default=None, help="width of the input image. Perform a resizing. default : 96")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="size of batch. default : 64")
    parser.add_argument("-lw", "--load-weight", type=str, default=None, help="path to the weights file for loading weights.")
    parser.add_argument("-sw", "--save-weight", type=str, default=None, help="path to the weights file for saving weights.")
    parser.add_argument("-v", "--visualize", action="store_true", default=None, help="option to vizualize some samples.")
    parser.add_argument("-t", "--treshold", type=float, default=None, help="treshold for pixel classification from probabilities.")
    parser.add_argument("-p", "--predict", type=str, default=None, help="path to image.")
   
    argvs  = parser.parse_args()
    option = parser.parse_args()

    # ================================================================================================================ #
    # DEFAULT VALUE                                                                                                    #
    # ================================================================================================================ #
    

    option.input_size  = 3
    option.stage_sizes = [64,128,256,512,1024]
    option.head_filter = 32
    option.nb_classes  = 2
    option.train       = False
    option.epoch       = 10
    option.width_size  = 96
    option.clip_im_dir = 'data\\clip_img'
    option.matting_dir = 'data\\matting'
    option.batch_size  = 64
    option.save_weight = None
    option.load_weight = None
    option.visualize   = False
    option.treshold    = 0.5
    
    # ================================================================================================================ #
    # CONFIG CHECK                                                                                                     #
    # ================================================================================================================ #

    if argvs.config_file != None:
        config_parser = configparser.ConfigParser()
        config_parser.read(argvs.config_file)

        if config_parser.has_option("MODEL","input-channel"):
            option.input_size  = int(config_parser.get("MODEL","input-channel"))
            
        if config_parser.has_option("MODEL","stage-sizes"):
            option.stage_sizes = json.loads(config_parser.get("MODEL","stage-sizes")) 
        
        if config_parser.has_option("MODEL","head-filter"):
            option.head_filter = int(config_parser.get("MODEL","head-filter"))
        
        if config_parser.has_option("MODEL","nb-classes"):
            option.nb_classes  = int(config_parser.get("MODEL","nb-classes"))
            
        if config_parser.has_option("MODEL","save-weight"):
            if config_parser.get("MODEL","save-weight") == "None" or config_parser.get("MODEL","save-weight") == "":
                option.save_weight = None
            else:
                option.save_weight  = config_parser.get("MODEL","save-weight")
            
        if config_parser.has_option("MODEL","load-weight"):
            if config_parser.get("MODEL","load-weight") == "None" or config_parser.get("MODEL","load-weight") == "":
                option.load_weight = None
            else:
                option.load_weight  = config_parser.get("MODEL","load-weight")
                
        if config_parser.has_option("MODEL","treshold"):
            option.treshold  = float(config_parser.get("MODEL","treshold"))


            
        if config_parser.has_option("TRAIN","batch-size"):
            option.batch_size  = int(config_parser.get("TRAIN","batch-size"))
        

        if config_parser.has_option("TRAIN","train"):
            if config_parser.get("TRAIN","train") == "True":
                option.train = True
            else :
                option.train = False
                
        
        if config_parser.has_option("TRAIN","epoch"):
            option.epoch     = int(config_parser.get("TRAIN","epoch"))
            
        if config_parser.has_option("TRAIN","visualize"):
            if config_parser.get("TRAIN","visualize") == "True" :
                option.visualize = True
            else :
                option.visualize = False
            
        
        if config_parser.has_option("DATA","width"):
            option.width_size  = int(config_parser.get("DATA","width"))
            
        if config_parser.has_option("DATA","clip-dir"):
            option.clip_im_dir = config_parser.get("DATA","clip-dir")
        
        if config_parser.has_option("DATA","label-dir"):
            option.matting_dir = config_parser.get("DATA","label-dir")
        





    # ================================================================================================================ #
    # OPTION PARSER                                                                                                       #
    # ================================================================================================================ #



    if argvs.train != None :
        option.train = argvs.train
        
        
    if argvs.epoch != None :
        option.epoch = argvs.epoch


    if argvs.width_size != None :
        option.width_size = argvs.width_size
        
    if argvs.batch_size != None :
        option.batch_size = argvs.batch_size
        
    if argvs.save_weight != None :
        option.save_weight = argvs.save_weight
        
    if argvs.load_weight != None :
        option.load_weight = argvs.load_weight
        
    if argvs.visualize != None :
        option.visualize = argvs.visualize
        
    if argvs.treshold != None :
        option.treshold = argvs.treshold
        
    if argvs.predict != None :
        option.predict = argvs.predict


    print(option)
    params = {'batch_size': option.batch_size,
              'shuffle': True,
              'drop_last': True}
    
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        params.update(cuda_kwargs)
    else :
        device = 'cpu'
        
    transform=transforms.Compose([
        transforms.Resize([option.width_size])
        ])
    
    
    # clip_im_dir = 'H:/Projet/ia/blurry_background/data/clip_img'
    # matting_dir = 'H:/Projet/ia/blurry_background/data/matting'
    clip_im_dir = option.clip_im_dir
    matting_dir = option.matting_dir
    
    print("create datasets")
    data_df = MergeDataframe(clip_im_dir, matting_dir)
    
    #separate data in training and test data (20/80)
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    
    # train data
    training_set = PortraitData(train_df, clip_im_dir, matting_dir, transform, transform)
    train_loader = DataLoader(training_set, **params)
    
    
    #test data
    testing_set = PortraitData(test_df, clip_im_dir, matting_dir, transform, transform)
    test_loader = DataLoader(testing_set, **params)
    
    

    input_size  = option.input_size
    stage_sizes = option.stage_sizes
    head_filter = option.head_filter
    nb_classes  = option.nb_classes
    
    print('create model')
    model = Unet_model(input_size, stage_sizes, head_filter, nb_classes)
    
    if option.load_weight != None :
        model.load_state_dict(torch.load(option.load_weight))
        
        
    if torch.cuda.is_available():
        model.to(device)
        # print(model)
    
    
    if option.train :
        print('begin training')
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn   = nn.CrossEntropyLoss()
        
        epoch = option.epoch
        
        # train the model
        for ep in range(epoch):
            train(model, device, train_loader, loss_fn, optimizer, ep, option.visualize, option.treshold)
        
        # test model
        print('begin testing')
        test(model, device, test_loader, loss_fn)
        
        print('end testing')
        if option.save_weight != None :
            torch.save(model.state_dict(), option.save_weight)
    else :
        
        #read image
        in_img = read_image(option.predict, mode=ImageReadMode.RGB)
        in_img = in_img.unsqueeze(0)
        in_img = transform(in_img)
        in_img = in_img/255
        
        in_img_cuda = in_img.to(device)
        
        #make prediction
        out = model(in_img_cuda)
        
        out_img = segmented_img(out, option.treshold)
        
        #plot
        in_img = in_img.squeeze(0)
        print(in_img.size(), out_img.size())
        
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('Input image')
        ax[0].imshow(in_img.permute(1,2,0))
        
        ax[1].set_title('Output image')
        ax[1].imshow(out_img.permute(1,2,0))
        
        
        plt.xticks([]), plt.yticks([])
        plt.show()
        
        
print('end')