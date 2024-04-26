import os
from datasets import npz_file_reader
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
from models import ResUNet3d
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(train_loader, model, criterion, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    step = 0

    for data in train_loader:

        fullImg = data["label"]
        fullImg = fullImg.cuda()
        _, _, indexZ, _, indexXY = fullImg.size()
        indexZ = indexZ // 2
        indexXY = indexXY // 2

        scoutImg = data["input"]
        scoutImg = scoutImg.cuda()

        if epoch > 4: fullImg, scoutImg = MixUp_AUG().aug(fullImg, scoutImg)

        predictImg = model(scoutImg)

        loss = criterion(predictImg, fullImg)

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('losses', {'train_mae_loss': losses.avg}, epoch + 1)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)
    writer.add_image('train img/reference img', normalization(torch.cat([fullImg[0, :, indexZ, :, :], fullImg[0, :, :, indexXY, :], fullImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('train img/predict img', normalization(torch.cat([predictImg[0, :, indexZ, :, :], predictImg[0, :, :, indexXY, :], predictImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('train img/fdk img', normalization(torch.cat([scoutImg[0, :, indexZ, :, :], scoutImg[0, :, :, indexXY, :], scoutImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(fullImg[0, :, indexZ, :, :] - predictImg[0, :, indexZ, :, :])), epoch + 1)

    scheduler.step()
    print('Train Epoch: {}\t train_mae_loss: {:.6f}\t'.format(epoch + 1, losses.avg))

def valid(valid_loader, model, criterion, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()

    step = 0

    for data in valid_loader:

        fullImg = data["label"]
        fullImg = fullImg.cuda()
        _, _, indexZ, _, indexXY = fullImg.size()
        indexZ = indexZ // 2
        indexXY = indexXY // 2

        scoutImg = data["input"]
        scoutImg = scoutImg.cuda()

        with torch.no_grad():

            predictImg = model(scoutImg)
            loss = criterion(predictImg, fullImg)

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('losses', {'valid_mae_loss': losses.avg}, epoch+1)
    writer.add_image('valid img/reference img', normalization(torch.cat([fullImg[0, :, indexZ, :, :], fullImg[0, :, :, indexXY, :], fullImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('valid img/predict img', normalization(torch.cat([predictImg[0, :, indexZ, :, :], predictImg[0, :, :, indexXY, :], predictImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('valid img/fdk img', normalization(torch.cat([scoutImg[0, :, indexZ, :, :], scoutImg[0, :, :, indexXY, :], scoutImg[0, :, :, :, indexXY]], 1)), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(fullImg[0, :, indexZ, :, :] - predictImg[0, :, indexZ, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mae_loss: {:.6f}\t'.format(epoch + 1, losses.avg))


if __name__ == "__main__":

    cudnn.benchmark = True

    result_path = f'./runs/3DResUNet/logs/'
    save_dir = f'./runs/3DResUNet/checkpoints/'

    train_dataset = npz_file_reader.npzFileReader(paired_data_txt=f'./txt/train_3d_patches_list.txt')
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=128, shuffle=True)

    valid_dataset = npz_file_reader.npzFileReader(paired_data_txt=f'./txt/valid_3d_volume_list.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    model = ResUNet3d()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.85)

    if os.path.exists(save_dir) is False:

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, criterion, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, criterion, writer, epoch)

        save_model(model, optimizer, epoch + 1, save_dir, scheduler)
