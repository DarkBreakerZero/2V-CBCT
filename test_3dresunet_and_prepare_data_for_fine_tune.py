# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from models import ResUNet3d
from utils import *
import time
from pytools import *
import nibabel as nib

trian_list = ['TrainCase1', 'TrainCase2', 'TrainCase3', ...]

validation_list = ['ValidationCase1', 'ValidationCase2', 'ValidationCase3', ...]

patient_list = validation_list + trian_list

gpu = 'cuda:0'

epoch = 27
model_name = '3DResUNet'
model_dir = './runs/' + model_name + '/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir, map_location=gpu)

save_dir = f'/Data/Stage1Results/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

model = ResUNet3d()
model = load_model(model, checkpoint).to(gpu)
model.eval()

input_data_dir = '/Data/input_data_dir/'

patchX = 128
patchY = 128
patchZ = 64

for index, patient in enumerate(patient_list):


    tic = time.time()

    inputFile = nib.load(f'{input_data_dir}/' + patient)
    inputImgData = inputFile.get_fdata()
    inputImgData = np.float32(inputImgData)
    inputImgData = np.transpose(inputImgData, [2, 1, 0])

    inputData = inputImgData[np.newaxis, np.newaxis, ...]
    inputData = torch.FloatTensor(inputData).to(gpu)

    outputData = np.zeros_like(inputImgData)

    indexZ_list = np.arange(0, np.size(inputImgData, 0)-patchZ, patchZ)
    if indexZ_list[-1] + patchZ < np.size(inputImgData, 0): indexZ_list = np.append(indexZ_list, np.size(inputImgData, 0)-patchZ)

    indexX_list = np.arange(0, np.size(inputImgData, 1)-patchX, patchX)
    if indexX_list[-1] + patchX < np.size(inputImgData, 1): indexX_list = np.append(indexX_list, np.size(inputImgData, 1)-patchX)

    indexY_list = indexX_list

    with torch.no_grad():

        for indexZ in indexZ_list:
            for indexX in indexX_list:
                for indexY in indexY_list:
                    inputPatch = inputData[:, :, indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY]
                    predictPatch = model(inputPatch)
                    predictPatch = np.squeeze(predictPatch.data.cpu().numpy())
                    outputData[indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY] = predictPatch

    outputData[outputData<0] = 0
    np.save(save_dir + patient[:-11], outputData)

    toc = time.time()

    print('Patient {0}, finished {1}/{2}, costs {3} seconds.'.format(patient, index + 1, len(patient_list), int(toc-tic)))