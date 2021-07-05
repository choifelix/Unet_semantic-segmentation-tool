# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 14:41:10 2021

@author: felix
"""

import cv2





import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io import ImageReadMode

from sklearn.model_selection import train_test_split 

from dataHandle import PortraitData, create_train_test

from Unet_model import Unet_model

from utils import train, test, segmented_img

from argparse import *
import configparser 
import json
import os

import matplotlib.pyplot as plt


def crop_portrait(frame, h,w)

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
    parser.add_argument("-d", "--dropout", type=float, default=None, help="dropout hyperparameter")
    parser.add_argument("-gpu", "--gpu", action="store_true", default=None, help="run on gpu")
   
    argvs  = parser.parse_args()
    option = parser.parse_args()

    # ================================================================================================================ #
    # DEFAULT VALUE                                                                                                    #
    # ================================================================================================================ #
    

    option.input_size  = 3
    option.stage_sizes = [64,128,256,512,1024]
    option.head_filter = 32
    option.nb_classes  = 2
    option.width_size  = 96
    option.treshold    = 0.5
    option.gpu         = False

    
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
            
            
        if config_parser.has_option("MODEL","load-weight"):
            if config_parser.get("MODEL","load-weight") == "None" or config_parser.get("MODEL","load-weight") == "":
                option.load_weight = None
            else:
                option.load_weight  = config_parser.get("MODEL","load-weight")
                
        if config_parser.has_option("MODEL","treshold"):
            option.treshold  = float(config_parser.get("MODEL","treshold"))
            



            
            
            
        
        if config_parser.has_option("DATA","width"):
            option.width_size  = int(config_parser.get("DATA","width"))
            

        





    # ================================================================================================================ #
    # OPTION PARSER                                                                                                       #
    # ================================================================================================================ #






    if argvs.width_size != None :
        option.width_size = argvs.width_size
        
        
    if argvs.save_weight != None :
        option.save_weight = argvs.save_weight
        
    if argvs.load_weight != None :
        option.load_weight = argvs.load_weight
        
        
    if argvs.treshold != None :
        option.treshold = argvs.treshold
        
    if argvs.gpu != None :
        option.gpu = argvs.gpu



    print(option)
    
    # ================================================================================================================ #
    # PARAMETERS AND ADJUSTEMENT                                                                                                       #
    # ================================================================================================================ #

    
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available() and option.gpu:
        device = 'cuda'
        
    else :
        device = 'cpu'
        
    transform=transforms.Compose([
        transforms.Resize([option.width_size])
        ])
    
    
    # clip_im_dir = 'H:/Projet/ia/blurry_background/data/clip_img'
    # matting_dir = 'H:/Projet/ia/blurry_background/data/matting'
    
    
    

    input_size  = option.input_size
    stage_sizes = option.stage_sizes
    head_filter = option.head_filter
    nb_classes  = option.nb_classes
    
    print('create model')
    model = Unet_model(input_size, stage_sizes, head_filter, nb_classes)
    model.eval()
    

    model.load_state_dict(torch.load(option.load_weight))
        
        
    if torch.cuda.is_available():
        model.to(device)
        # print(model)
    
    
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame.shape)
        py_frame = torch.tensor(frame, dtype=torch.float32).permute([2,1,0])
        py_frame = py_frame.unsqueeze(0)
        # print(py_frame.size())
        py_frame = transform(py_frame)
        tran_frame = py_frame.squeeze(0).permute([2,1,0])
        py_frame /= 255
        
        py_frame = py_frame.to(device)
        # print(py_frame.size())
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        out = model(py_frame)
        # out = F.softmax(output, dim=1)
        # out = out > treshold
        visual_img, prob_output = segmented_img(out, option.treshold)
        # print(visual_img.size())
        
        visual_img = visual_img.squeeze(0)
        visual_img = visual_img.permute([2,1,0])
        visual_img = visual_img.to('cpu')
        # print(visual_img.size())
        
        prob_output = prob_output.squeeze(0)
        prob_output = prob_output.permute([1,0])
        prob_output = prob_output.to('cpu')
        
        
        
        cv2.imshow('Input', frame)
        cv2.imshow('Input_TRANSFORM', tran_frame.numpy())
        cv2.imshow('Output', visual_img.numpy())
        cv2.imshow('Prob', prob_output.numpy())
    
        c = cv2.waitKey(1)
        if c == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
   
        
        
print('end')