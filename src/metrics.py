# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:36:40 2021

@author: felix
"""
import torch

def pixel_accuracy(output, target):
    """
        compute mean pixel accuracy of the batch
    input :
        output - bool tensor[b,c,h,w], predicted images 
        target - bool tensor[b,c,h,w], label images 
        
    
    """
    # if len(target.size()) == 3 :
    #     target = target.unsqueeze(1)
       
    b,h,w = target.size()
    total   = b*h*w
    correct = (output == target).int().sum().item()
    # print(output.size(),target.size())
    # print((output == target).size())
    # print(correct, total)
    
    acc = correct/total
    
    return acc



def IoU(output, target):
    """
        compute mean Intersection over Union accuracy of the batch
    input :
        output - bool tensor[b,1,h,w], predicted images on one class
        terget - bool tensor[b,1,h,w], label images on one class
        
    
    """
    
    # pred = output[:,class_index,:,:]
    pred  = output.squeeze(1).detach()
    label = target.squeeze(1).detach()
    
    
    intersection = (torch.logical_and(pred, label)).int().sum(dim=(1,2))
    union        = (torch.logical_or( pred, label)).int().sum(dim=(1,2))
    
    smooth   = 1e-6
    iou      = (intersection + smooth)/ (union + smooth)
    mean_iou = iou.mean().item()
    
    return mean_iou


def Dice_coef(output, target):
    """
        compute mean Dice coef of the batch
    input :
        output - bool tensor[b,1,h,w], predicted images on one class
        terget - bool tensor[b,1,h,w], label images on one class
        
    
    """
    
    # pred = output[:,class_index,:,:]
    pred  = output.squeeze(1).detach()
    label = target.squeeze(1).detach()
    
    
    intersection = (torch.logical_and(pred, label)).int().sum(dim=(1,2))
    union        = (torch.logical_or( pred, label)).int().sum(dim=(1,2))
    
    smooth   = 1e-6
    dice      = (2*intersection + smooth)/ (intersection + union + smooth)
    mean_dice = dice.mean().item()
    
    return mean_dice
    
    