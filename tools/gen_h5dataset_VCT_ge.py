#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:26:31 2022

@author: Rodrigo
"""

import numpy as np
import pydicom as dicom
import h5py
import random

from pathlib import Path

#%%

def get_img_bounds(img):
    '''Get image bounds of the segmented breast'''
    
    # Segment the breast
    mask = img < 10000
    
    # Height bounds
    mask_h = np.sum(mask, 1) > 0
    res = np.where(mask_h == True)
    h_min, h_max = res[0][0], res[0][-1]
    
    # Weight bounds
    mask_w = np.sum(mask, 0) > 0
    res = np.where(mask_w == True)
    w_min, w_max = res[0][0], res[0][-1]
        
    return w_min, h_min, w_max, h_max


def extract_rois(img_wc, img_nc):
    '''Extract Independent and Correlatedrois'''
    
    # Check if images are the same size
    assert img_wc.shape == img_nc.shape, "image sizes differ"
    
    global trow_away
    
    # Get image bounds of the segmented breast from the GT
    w_min, h_min, w_max, h_max = get_img_bounds(img_nc)
    
    # Crop all images
    img_wc = img_wc[h_min:h_max, w_min:w_max]
    img_nc = img_nc[h_min:h_max, w_min:w_max]
    
    # Get updated image shape
    w, h = img_wc.shape
    
    rois = []
    
    # Non-overlaping roi extraction
    for i in range(0, w-64, 64):
        for j in range(0, h-64, 64):
            
            # Extract roi
            roi_tuple = (img_wc[i:i+64, j:j+64], img_nc[i:i+64, j:j+64])
            
            # Am I geting at least one pixel from the breast?
            if np.sum(roi_tuple[1] > 10000) != 64*64:
                rois.append(roi_tuple)
            else:
                trow_away += 1                

    return rois


def process_each_folder(folder_name, num_proj=9):
    '''Process DBT folder to extract Independent and Correlated rois'''
        
    noisy_path = path2read + '/noisy/' + folder_name.split('/')[-1]
    
    rlz = 1
    
    global nt_imgs
    
    rois = []
    
    # Loop on each projection
    for proj in range(num_proj):
        
        # Independent image
        nc_file_name = noisy_path + '-{}mAs-rlz{}-nc/_{}.dcm'.format(lowDosemAs,rlz,proj)

        # Correlated image
        wc_file_name = noisy_path + '-{}mAs-rlz{}-wc/_{}.dcm'.format(dosemAs,rlz,proj)
    
        img_nc = dicom.read_file(nc_file_name).pixel_array
        img_wc = dicom.read_file(wc_file_name).pixel_array
    
        rois += extract_rois(img_wc, img_nc)
                    
    return rois

#%%

if __name__ == '__main__':
    
    path2read = '/media/rodrigo/Data/images/VCT_PEN/VCT_Bruno_500/GE-projs'
    path2write = '../data/'
    
    random.seed(0)
    np.random.seed(0)
    
    folder_names = [str(item) for item in Path(path2read).glob("*-proj") if Path(item).is_dir()]
    
    random.shuffle(folder_names)
    
    dosemAs = 90
    lowDosemAs = 360
    
    nROIs_total = 256000
    
    trow_away = 0
    flag_final = 0
    nROIs = 0
    
    # Create h5 file
    f = h5py.File('{}DBT_Decor_training_{}mAs.h5'.format(path2write, dosemAs), 'a')
    
    # Loop on each DBT folder (projections)
    for idX, folder_name in enumerate(folder_names):
        
        # Get low-dose and full-dose rois
        rois = process_each_folder(folder_name)        
                
        data = np.stack([x[0] for x in rois])
        target = np.stack([x[1] for x in rois])
        
        data = np.expand_dims(data, axis=1) 
        target = np.expand_dims(target, axis=1) 
        
        nROIs += data.shape[0]
        
        # Did I reach the expected size (nROIs_total)?
        if  nROIs >= nROIs_total:
            flag_final = 1
            diff = nROIs_total - nROIs
            data = data[:diff,:,:,:]
            target = target[:diff,:,:,:]
                            
        if idX == 0:
            f.create_dataset('data', data=data, chunks=True, maxshape=(None,1,64,64))
            f.create_dataset('target', data=target, chunks=True, maxshape=(None,1,64,64)) 
        else:
            f['data'].resize((f['data'].shape[0] + data.shape[0]), axis=0)
            f['data'][-data.shape[0]:] = data
            
            f['target'].resize((f['target'].shape[0] + target.shape[0]), axis=0)
            f['target'][-target.shape[0]:] = target
            
        print("Iter {} and 'data' chunk has shape:{} and 'target':{}".format(idX,f['data'].shape,f['target'].shape))

        if flag_final:
            break

    f.close()       
     
    
    