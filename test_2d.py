# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from models_2d import DenseUNet2d
from utils import *
import time
import nibabel as nib
from pytools import *

gpu = 'cuda:0'

test_list = ["TestCase1", "TestCase2", "TestCase3", ...]

epoch = 24

model_name = f'DenseUNet2DVGG'

model_dir = './runs/' + model_name + '/checkpoints/model_at_epoch_' + str(epoch).rjust(3, '0') + '.dat'
checkpoint = torch.load(model_dir, map_location=torch.device(gpu))
model = DenseUNet2d()
model = load_model(model, checkpoint).to(gpu)
model.eval()

save_dir = f'./Stage2Results/{model_name}/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

input_data_dir = f'/Data/Stage1Results/'

patient_list = test_list

for patient_index, patient in enumerate(patient_list):

    labelFile = nib.load('/Data/labelFileDir/' + patient)
    labelImgData = labelFile.get_fdata()
    labelImgData = np.float32(labelImgData)
    labelImgData = np.transpose(labelImgData, [2, 1, 0])
    labelImgData.astype(np.float32).tofile(save_dir + patient[:-4] + '_label.raw')

    patient = patient[:-11] + '.npy'

    inputData = np.load(f'/Data/Stage1Results/' + patient)
    inputData[inputData<0] = 0

    outData = np.zeros(np.shape(inputData), dtype=np.float32)

    tic = time.time()

    for index in range(np.size(inputData, 0)):

        inputDataSlice = inputData[index, :, :]
        inputDataSlice = inputDataSlice[np.newaxis, np.newaxis, ...]
        inputDataSlice = torch.FloatTensor(inputDataSlice).to(gpu)

        with torch.no_grad():

            predictImg = model(inputDataSlice)

        predict = np.squeeze(predictImg.data.cpu().numpy())

        predict[predict < 0] = 0
        predict = np.around(predict)
        outData[index, :, :] = predict

    toc = time.time()

    outData.astype(np.float32).tofile(save_dir + patient[:-4] + '_' + model_name + '_E' + str(epoch) + '.raw')

    print('Patent {0}, total {1}/3, cost {2} seconds.'.format(patient, patient_index+1, toc-tic))