# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 23:30:56 2021

@author: felix
"""
import torch

import torch.nn.functional as F

import matplotlib.pyplot as plt

from src.metrics import *

def choose_color(n, r, g, b):
    """
    compute rgb combination from integer n and rgb starting point (usually 0,0,0)
    aim to create a color per classe that is very differentiable (not random).
    recursive function.
    
    input :
        n - int, number of class
        r - int, value for red
        g - int, value for green
        b - int, value for blue
    output :
        r,g,b - 3-tuple int, rgb combination
    """
    a = int(n/3)
    s = n%3
    
    if s == 0 :
        r += (int(256/(2**a)) -1)
        r %= 256
    elif s == 1 :
        g += (int(256/(2**a)) -1)
        g %= 256
    elif s == 2 :
        b += (int(256/(2**a)) -1)
        b %= 256

        
    if a > 0 :
        r , g, b = choose_color(a, r, g, b)
    
    return r,g,b


def segmented_img(tensor_output, treshold):
    """
    from a tensor oupt of format [b,c,w,h] output an [3,h,w] semantically segmented image of the first image of the batch (b=0)
    
    input :
        tensor_output - tensor[b,c,w,h]
        treshold - float[0,1], treshold for defining masks for each classes
        
    output:
        visual_img - tensor[3,h,w], rgb image of the semantically segmented image (input image of a model)

    """
    visual_output = tensor_output[0,:,:,:] 
    visual_output = visual_output.squeeze(0)
    visual_output = F.softmax(visual_output, dim=0)
    visual_output = visual_output.to('cpu')
    prob_output   = visual_output[1,:,:].detach()
    visual_output = visual_output > treshold
 
    n, h, w = visual_output.size()
    
    # visual = torch.zeros(h, w) 
    glob_r = torch.zeros(h, w) 
    glob_g = torch.zeros(h, w) 
    glob_b = torch.zeros(h, w) 
    
    for classe in range(n) :
        seg = visual_output[classe,:,:]
        
        r = torch.zeros(h, w) 
        g = torch.zeros(h, w) 
        b = torch.zeros(h, w)
        
        #choose class color
        r_int, g_int, b_int = choose_color(classe, 0, 0, 0)
        
        r += r_int 
        g += g_int
        b += b_int

        r *= seg.int()
        g *= seg.int()
        b *= seg.int()
        
        
        glob_r += r
        glob_g += g
        glob_b += b
        
    visual_img = torch.stack((glob_r, glob_g, glob_b), dim=0)
    
    return visual_img, prob_output

def format_input_img(img):
    """
    format input image
    """
    visual_output = img[0,:,:,:] 
    visual_output = visual_output.squeeze(0)
    visual_output = visual_output.to('cpu')
    
    return visual_output


def format_label_img(target):
    """
    format input image
    """
    visual_output = target[0,:,:] 
    visual_output = visual_output.squeeze(0)
    visual_output = visual_output.to('cpu')
    
    
    return visual_output

def plot_images(in_img, prob_img, out_img, label_img):
    """
    plot a sample of the model [input, output, label].


    """
    fig, ax = plt.subplots(1, 4)
    ax[0].set_title('Input image')
    ax[0].imshow(in_img.permute(1,2,0))
    
    ax[1].set_title('Prob_class_0_image')
    ax[1].imshow(prob_img)
    
    ax[2].set_title('Segmented_Output image')
    ax[2].imshow(out_img.permute(1,2,0))
    
    ax[3].set_title('Label image')
    ax[3].imshow(label_img)
    
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    
    
            
    
def train(model, device, train_loader, loss_fn, optimizer, epoch, visual=False, treshold=0.5):
    model.train()
    
    loss_metric = []
    pixel_acc_metric = 0
    iou_metric       = 0
    dice_metric      = 0
    
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data   = data.float()
        target = target.float() 
        data, target = data.to(device), target.to(device=device, dtype=torch.int64)
        
        # Compute prediction and loss

        output = model(data)

        loss = loss_fn(output, target)
        
  
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        out = F.softmax(output, dim=1)
        out = out > treshold
        pixel_acc = pixel_accuracy(out[:,1,:,:], target)
        iou       = IoU(      out[:,1,:,:], target.unsqueeze(1))
        dice      = Dice_coef(out[:,1,:,:], target.unsqueeze(1))
        
        pixel_acc_metric += pixel_acc
        iou_metric       += iou
        dice_metric      += dice

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            loss_metric.append(loss.item())
            
            
            
            print('Average pixel Accuracy:{:.4f}, Average IoU: {:.4f}, Average Dice coef: {:.4f}'.format(
                pixel_acc, iou, dice))
            
            if visual :
                input_img  = format_input_img(data)
                output_img, prob_img = segmented_img(output, treshold)
                label_img  = format_label_img(target)
                
                plot_images(input_img, prob_img, output_img, label_img)
                
        torch.cuda.empty_cache()
        
    dataset_size = len(train_loader)
    
    pixel_acc_metric /= dataset_size
    iou_metric       /= dataset_size
    dice_metric      /= dataset_size
    
    return loss_metric, pixel_acc_metric, iou_metric, dice_metric


def test(model, device, test_loader, loss_fn, treshold=0.5, visual=False):
    model.eval()
    test_loss = 0
    pixel_acc = 0
    iou       = 0
    dice      = 0
    
    with torch.no_grad():
        
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device=device, dtype=torch.int64)
            output = model(data)
            
            test_loss += loss_fn(output, target).item()
            # correct += (output.argmax(1) == target).type(torch.float).sum().item()
            
            out = F.softmax(output, dim=1)
            out = out > treshold
            pixel_acc += pixel_accuracy(out[:,1,:,:], target)
            iou       += IoU(      out[:,1,:,:], target.unsqueeze(1))
            dice      += Dice_coef(out[:,1,:,:], target.unsqueeze(1))

    dataset_size = len(test_loader)
    test_loss /= dataset_size
    pixel_acc /= dataset_size
    iou       /= dataset_size
    dice      /= dataset_size

    print('\nTest set: Average loss: {:.4f}, Average pixel Accuracy:{:.4f}, Average IoU: {:.4f}, Average Dice coef: {:.4f}\n'.format(
        test_loss, pixel_acc, iou, dice))