import os
import torch
import numpy as np
from collections import OrderedDict
from scipy import ndimage
import cv2
import torch.nn.functional as F

def load_model(net, checkpoint):
    net.load_state_dict(checkpoint['state_dict'])
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net).cuda()
    # elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    #     net = net.cuda()
    return net

def save_model(net, optimizer, epoch, save_dir, scheduler=None):
    '''save model'''

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    if 'module' in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()

    if scheduler is None:
        torch.save({
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    else:
        torch.save({
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict()},
            os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))

    print(os.path.join(save_dir, 'model_at_epoch_%03d.dat' % (epoch)))
    
def find_lastest_file(file_dir):

    lists = os.listdir(file_dir)
    lists.sort(key=lambda x: os.path.getmtime((file_dir + x)))
    file_latest = os.path.join(file_dir, lists[-1])

    return file_latest

def normalization(tensor):

    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

def gaussian_smooth(input, kernel_size=9, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    if kernel.shape[1] != input.shape[1]:
            kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    low = F.conv2d(input, kernel, padding=(kernel_size-1)//2, groups=input.shape[1])
    high = input - low

    return torch.cat([input, low, high], 1)

def get_edges(input, kernel_size=9, sigma=3.5):
    # inputs: batch, channel, width, height

    filter = np.float32(np.multiply(cv2.getGaussianKernel(kernel_size, sigma), np.transpose(cv2.getGaussianKernel(kernel_size, sigma))))
    filter = filter[np.newaxis, np.newaxis, ...]
    kernel = torch.FloatTensor(filter).cuda(input.get_device())
    if kernel.shape[1] != input.shape[1]:
            kernel = kernel.repeat(input.shape[1], 1, 1, 1)
    low = F.conv2d(input, kernel, padding=(kernel_size-1)//2, groups=input.shape[1])
    high = input - low

    return high

def extract_patches_online(tensor, num=2):

    if tensor.ndim == 5:

        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]])
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]//num])

        return stack_h

    elif tensor.ndim == 4:

        split_w = torch.chunk(tensor, chunks=num, dim=2)
        # print('split_w', split_w.size())
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]])
        # print('stack_w', stack_w.size())
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        # print('split_h', split_h.size())
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]//num])
        # print('stack_h', stack_h.size())

        return stack_h

    else:
        print('Expect for the tensor with dim==5 or 4, other cases are not yet implemented.')

### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]

        if rgb_gt.ndim == 4:

            lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda(rgb_gt.get_device())

        elif rgb_gt.ndim == 5:

            lam = self.dist.rsample((bs,1)).view(-1,1,1,1,1).cuda(rgb_gt.get_device())

        else:

            print('Dim is not implemented for MixUp_AUG!')

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2

        return rgb_gt, rgb_noisy

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
