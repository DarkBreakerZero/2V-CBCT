import os
from datasets import npz_file_reader
import scipy
import numpy as np
import torch.nn
# from datasets import dicom_file
from torch.utils.data import DataLoader
from torch.backends import cudnn
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
import time
import torch.optim as optim
from losses import *
from models_2d import DenseUNet2d
from utils import *
from datetime import datetime
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def train(train_loader, model, vgg_feature_extractor, loss_mae, loss_ssim, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    loss_mae_scalar = AverageMeter()
    loss_vgg_scalar = AverageMeter()
    model.train()
    end = time.time()

    step = 0

    for data in train_loader:

        RDCTImg = data["label"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["input"]
        LDCTImg = LDCTImg.cuda()

        if epoch > 4:
            RDCTImg, LDCTImg = MixUp_AUG().aug(RDCTImg, LDCTImg)

        predictImg = model(LDCTImg)
        loss1 = loss_mae(predictImg, RDCTImg)
        loss2 = 1-loss_ssim(predictImg, RDCTImg)
        loss3 = 0.02 * vgg_loss_calc(gaussian_smooth(predictImg, kernel_size=9), gaussian_smooth(RDCTImg, kernel_size=9), vgg_feature_extractor)
        # loss3 = 0.1 * vgg_loss_calc(predictImg, RDCTImg, vgg_feature_extractor)
        # loss = loss1 + loss2 * 300
        loss = loss3 + loss1

        loss_mae_scalar.update(loss1.item())
        loss_vgg_scalar.update(loss3.item())
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

    writer.add_scalars('loss/mae', {'train_mae_loss': loss_mae_scalar.avg}, epoch + 1)
    writer.add_scalars('loss/vgg', {'train_vgg_loss': loss_vgg_scalar.avg}, epoch + 1)
    writer.add_image('train img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :] - predictImg[0, :, :, :])), epoch + 1)

    scheduler.step()
    print('Train Epoch: {}\t train_mae_loss: {:.6f}\t'.format(epoch + 1, loss_mae_scalar.avg))

def valid(valid_loader, model, loss_mae, loss_ssim, writer, epoch):

    batch_time = AverageMeter()
    loss_mae_scalar = AverageMeter()
    loss_ssim_scalar = AverageMeter()
    model.eval()
    end = time.time()

    step = 0

    for data in valid_loader:


        RDCTImg = data["label"]
        RDCTImg = RDCTImg.cuda()

        LDCTImg = data["input"]
        LDCTImg = LDCTImg.cuda()

        with torch.no_grad():

            predictImg = model(LDCTImg)
            loss1 = loss_mae(predictImg, RDCTImg)
            loss2 = loss_ssim(predictImg, RDCTImg)

        loss_mae_scalar.update(loss1.item())
        loss_ssim_scalar.update(loss2.item())
        batch_time.update(time.time() - end)
        end = time.time()

        step += 1

    writer.add_scalars('loss/mae', {'valid_mae_loss': loss_mae_scalar.avg}, epoch+1)
    writer.add_scalars('loss/ssim', {'valid_ssim_loss': loss_ssim_scalar.avg}, epoch+1) 
    writer.add_image('valid img/reference img', normalization(RDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/predict img', normalization(predictImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/ldct img', normalization(LDCTImg[0, :, :, :]), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(RDCTImg[0, :, :, :]- predictImg[0, :, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mae_loss: {:.6f}\t'.format(epoch + 1, loss_mae_scalar.avg))


if __name__ == "__main__":


    cudnn.benchmark = True

    result_path = f'./runs/DenseUNet2DVGG/logs/'
    save_dir = f'./runs/DenseUNet2DVGG/checkpoints/'

    train_dataset = npz_file_reader.npzFileReader(paired_data_txt=f'./txt/train_2d_img_list.txt')
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=256, shuffle=True)

    valid_dataset = npz_file_reader.npzFileReader(paired_data_txt=f'./txt/valid_2d_img_list.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=64, shuffle=True)

    vgg = vgg_feature_extractor()
    vgg = torch.nn.DataParallel(vgg).cuda()
    vgg.eval()

    model = DenseUNet2d()
    # print(model)
    loss_mae = torch.nn.L1Loss()
    loss_ssim = SSIM()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    if os.path.exists(save_dir) is False:

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = torch.nn.DataParallel(load_model(model, checkpoint_latest)).cuda()
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

    for epoch in range(10, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, vgg, loss_mae, loss_ssim, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, loss_mae, loss_ssim, writer, epoch)
        save_model(model, optimizer, epoch + 1, save_dir, scheduler)