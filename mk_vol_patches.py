import os
import matplotlib.pyplot as plt
import numpy as np
from pytools import *
# import pylab
import time
# from scipy.ndimage import zoom
import os
import nibabel as nib

trian_list = ['TrainCase1', 'TrainCase2', 'TrainCase3', ...]

validation_list = ['ValidationCase1', 'ValidationCase2', 'ValidationCase3', ...]

root_dir = '/Data/'

TrainNPZSaveDir = f'{root_dir}TrainPatches3D/'
make_dirs(TrainNPZSaveDir)

make_dirs('./txt/')

patchX = 128
patchY = 128
patchZ = 64

cnt = 0

for index, patient in enumerate(trian_list):

    startTime = time.time()
    
    labelFile = nib.load(f'{root_dir}labelFileDir/' + patient)
    labelImgData = labelFile.get_fdata()
    labelImgData = np.float32(labelImgData)
    labelImgData = np.transpose(labelImgData, [2, 1, 0])

    inputFile = nib.load(f'{root_dir}inputFileDir/' + patient)
    inputImgData = inputFile.get_fdata()
    inputImgData = np.float32(inputImgData)
    inputImgData = np.transpose(inputImgData, [2, 1, 0])

    indexZ_list = np.arange(0, np.size(labelImgData, 0)-patchZ, patchZ//2)
    if indexZ_list[-1] + patchZ < np.size(labelImgData, 0): indexZ_list = np.append(indexZ_list, np.size(labelImgData, 0)-patchZ)

    indexX_list = np.arange(0, np.size(labelImgData, 1)-patchX, patchX//2)
    if indexX_list[-1] + patchX < np.size(labelImgData, 1): indexX_list = np.append(indexX_list, np.size(labelImgData, 1)-patchX)

    indexY_list = indexX_list

    for indexZ in indexZ_list:
        for indexX in indexX_list:
            for indexY in indexY_list:
                imgPatch = labelImgData[indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY]
                scoutPatch = inputImgData[indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY]

                np.savez(TrainNPZSaveDir + 'patient' + str(index) + 'z' + str(indexZ) + 'x' + str(indexX) + 'y' + str(indexY), input=scoutPatch, label=imgPatch)
                with open('./txt/train_3d_patches_list.txt', 'a') as f:
                    f.write(TrainNPZSaveDir + 'patient' + str(index) + 'z' + str(indexZ) + 'x' + str(indexX) + 'y' + str(indexY) + '.npz\n')
                f.close()

                cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds, finished {3}/{4}.'.format(patient, cnt, int(endTime-startTime), index+1, len(trian_list)))

import random
with open('./txt/train_3d_patches_list.txt', 'r') as infile:
    lines = infile.readlines()
random.shuffle(lines)
with open('./txt/train_3d_patches_list.txt', 'w') as outfile:
    outfile.writelines(lines)

ValidNPZSaveDir = f'{root_dir}ValidVol3D/'
make_dirs(ValidNPZSaveDir)

cnt = 0

# patchX = 128
# patchY = 128
# patchZ = 64

for index, patient in enumerate(validation_list):

    startTime = time.time()

    labelFile = nib.load(f'{root_dir}labelFileDir/' + patient)
    labelImgData = labelFile.get_fdata()
    labelImgData = np.float32(labelImgData)
    labelImgData = np.transpose(labelImgData, [2, 1, 0])

    inputFile = nib.load(f'{root_dir}inputFileDir/' + patient)
    inputImgData = inputFile.get_fdata()
    inputImgData = np.float32(inputImgData)
    inputImgData = np.transpose(inputImgData, [2, 1, 0])

    np.savez(ValidNPZSaveDir + 'patient' + str(index), input=inputImgData, label=labelImgData)

    cnt += 1

    with open('./txt/valid_3d_volume_list.txt', 'a') as f:
        f.write(ValidNPZSaveDir + 'patient' + str(index) + '.npz\n')
    f.close()

    # indexZ_list = np.arange(0, np.size(labelImgData, 0)-patchZ, patchZ)
    # if indexZ_list[-1] + patchZ < np.size(labelImgData, 0): indexZ_list = np.append(indexZ_list, np.size(labelImgData, 0)-patchZ)

    # indexX_list = np.arange(0, np.size(labelImgData, 1)-patchX, patchX)
    # if indexX_list[-1] + patchX < np.size(labelImgData, 1): indexX_list = np.append(indexX_list, np.size(labelImgData, 1)-patchX)

    # indexY_list = indexX_list

    # for indexZ in indexZ_list:
    #     for indexX in indexX_list:
    #         for indexY in indexY_list:

    #             imgPatch = labelImgData[indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY]
    #             scoutPatch = inputImgData[indexZ: indexZ+patchZ, indexX: indexX+patchX, indexY: indexY+patchY]

    #             np.savez(ValidNPZSaveDir + 'patient' + str(index) + 'z' + str(indexZ) + 'x' + str(indexX) + 'y' + str(indexY), input=scoutPatch, label=imgPatch)
    #             with open('./txt/valid_3d_patches_list.txt', 'a') as f:
    #                 f.write(ValidNPZSaveDir + 'patient' + str(index) + 'z' + str(indexZ) + 'x' + str(indexX) + 'y' + str(indexY) + '.npz\n')
    #             f.close()

    #             cnt += 1

    endTime = time.time()

    print('Patient {0} finished, totally got {1} samples, cost {2} seconds, finished {3}/{4}.'.format(patient, cnt, int(endTime-startTime), index+1, len(validation_list)))