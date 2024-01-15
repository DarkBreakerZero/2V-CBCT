import torch
from torch.utils.data import Dataset
import numpy as np

class npzFileReader(torch.utils.data.Dataset):
    def __init__(self, paired_data_txt):
        super(npzFileReader, self).__init__()
        self.paired_files = open(paired_data_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
            # pass
        elif data.ndim == 3:
            data = data[np.newaxis, ...]

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        # if data.ndim == 3: data = data.transpose(1, 2, 0)
        return data

    def get(self, paired_files):

        npzData = np.load(paired_files[:-1])

        imgData = npzData['label']
        scoutData = npzData['input']
        imgData[imgData < 0] = 0
        scoutData[scoutData < 0] = 0
        imgData = np.array(imgData).astype(np.float32)
        scoutData = np.array(scoutData).astype(np.float32)
        imgData = self.to_tensor(imgData)
        scoutData = self.to_tensor(scoutData)

        return {"label": imgData, "input": scoutData}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)

# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt

#     dataset = NpzFile(paired_data_txt='/home/yikun/10T/Code/AirTorch/txtDir/valid.txt')
#     data = dataset[4]

    # Phi = dataset.to_numpy(data["Phi"]).astype(np.float32)
    # print(Phi)

    # Proj = dataset.to_numpy(data["Proj"]).astype(np.float32)
    # print(np.shape(Proj))
    # plt.imshow(Proj, cmap=plt.cm.gray)
    # plt.show()

    # Img = dataset.to_numpy(data["Img"]).astype(np.float32)
    # print(np.shape(Img))
    # plt.imshow(np.concatenate([Img[80, :, :], Img[:, 150, :], Img[:, :, 150]], axis=0), cmap=plt.cm.gray)
    # plt.show()