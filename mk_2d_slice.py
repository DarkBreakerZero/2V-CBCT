import os
import matplotlib.pyplot as plt
import numpy as np
from pytools import *
# import pylab
import time
# from scipy.ndimage import zoom
import os
import nibabel as nib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

trian_list = ['TrainCase1', 'TrainCase2', 'TrainCase3', ...]

validation_list = ['ValidationCase1', 'ValidationCase2', 'ValidationCase3', ...]

root_dir = '/Data/'

TrainNPZSaveDir = f'{root_dir}TrainSlice2D/'
make_dirs(TrainNPZSaveDir)

for index, patient in enumerate(trian_list):

    startTime = time.time()

    labelFile = nib.load(f'{root_dir}labelFileDir/' + patient)
    labelImgData = labelFile.get_fdata()
    labelImgData = np.float32(labelImgData)
    labelImgData = np.transpose(labelImgData, [2, 1, 0])

    patient = patient[:-11] + '.npy'

    inputImgData = np.load(f'/Data/Stage1Results/' + patient)

    for indexZ in range(0, np.size(labelImgData, 0)):

        labelImgSlice = labelImgData[indexZ, :, :]
        inputImgSlice = inputImgData[indexZ, :, :]

        np.savez(TrainNPZSaveDir + patient[:-4] + '_slice' + str(indexZ), input=inputImgSlice, label=labelImgSlice)
        with open(f'./txt/train_2d_img_list.txt', 'a') as f:
            f.write(TrainNPZSaveDir + patient[:-4] + '_slice' + str(indexZ) + '.npz\n')
        f.close()

    endTime = time.time()

    print('Patient {0}, finished {1}/{2}, costs {3} seconds.'.format(patient[:-4], index + 1, len(trian_list), int(endTime-startTime)))

import random
with open('./txt/train_2d_img_list.txt', 'r') as infile:
    lines = infile.readlines()
random.shuffle(lines)
with open('./txt/train_2d_img_list.txt', 'w') as outfile:
    outfile.writelines(lines)

ValidNPZSaveDir = f'{root_dir}ValidSlice2D/'
make_dirs(ValidNPZSaveDir)

cnt = 0

for index, patient in enumerate(validation_list):

    startTime = time.time()

    labelFile = nib.load(f'{root_dir}labelFileDir/' + patient)
    labelImgData = labelFile.get_fdata()
    labelImgData = np.float32(labelImgData)
    labelImgData = np.transpose(labelImgData, [2, 1, 0])

    patient = patient[:-11] + '.npy'

    inputImgData = np.load(f'/Data/Stage1Results/' + patient)

    for indexZ in range(0, np.size(labelImgData, 0)):

        labelImgSlice = labelImgData[indexZ, :, :]
        inputImgSlice = inputImgData[indexZ, :, :]

        np.savez(ValidNPZSaveDir + patient[:-4] + '_slice' + str(indexZ), input=inputImgSlice, label=labelImgSlice)
        with open(f'./txt/valid_2d_img_list.txt', 'a') as f:
            f.write(ValidNPZSaveDir + patient[:-4] + '_slice' + str(indexZ) + '.npz\n')
        f.close()

        cnt += 1

    endTime = time.time()

    print('Patient {0}, finished {1}/{2}, costs {3} seconds.'.format(patient[:-4], index + 1, len(validation_list), int(endTime-startTime)))