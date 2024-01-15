import numpy as np
import os
import pydicom
import matplotlib
import matplotlib.animation as animation
from matplotlib import pyplot as plt

def read_raw_data(file_name, w, h):

    file_temp = np.fromfile(file_name, dtype='float32', sep="")
    slice = int(np.size(file_temp) / w / h)
    file_temp = np.reshape(file_temp, [slice, w, h])

    return slice, file_temp

def read_raw_data_all(dir, w=512, h=512, start_index=8, end_index=-4):

    file_list = os.listdir(dir)
    file_list.sort(key=lambda x: int(x[start_index: end_index]))
    slice = len(file_list)

    file_vol = np.zeros([slice, w, h], dtype=np.float32)

    for index in range(slice):

        file_temp = np.fromfile(dir + file_list[index], dtype='float32', sep="")
        file_temp = file_temp.reshape([w, h])
        file_vol[index, :, :] = file_temp

    return file_vol

def dicomreader(filename):
    info = pydicom.read_file(filename)
    img = np.float32(info.pixel_array)
    return info, img

def listsorter(dir, strat_index, end_index):
    list = os.listdir(dir)
    # print(list)
    list.sort(key=lambda x: int(x[strat_index: end_index]))
    return list

def read_dicom_all(file_dir, sort_start, sort_end, w=512, h=512):

    file_names = listsorter(file_dir, strat_index=sort_start, end_index=sort_end)
    slice_number = len(file_names)
    volume = np.zeros([slice_number, w, h], dtype=np.float32)
    for index in range(slice_number):
        _, img = dicomreader(file_dir + file_names[index])
        # img = np.flipud(img)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()
        volume[index, :, :] = img

    return volume


def make_dirs(dir_path):

    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

def addPossionNoisy(CleanProj, I0=5e4):

    TempProj = I0 * np.exp(-CleanProj)
    NoiseProj = np.random.poisson(TempProj) + 1
    NoiseProj = -np.log(NoiseProj/I0)

    return NoiseProj

def genMask(imgX=512, imgY=512, imgZ=256, maskR=230):

    maskSlice = np.zeros([imgX, imgY, 1], dtype=np.float32)

    for indexX in range(imgX):
        for indexY in range(imgY):
            if (indexX - imgX/2)**2 + (indexY - imgY/2)**2 <= maskR**2:
                maskSlice[indexX, indexY, :] = 1

    maskVol = np.tile(maskSlice, imgZ).transpose()
    # print(maskVol.shape)
    return maskVol