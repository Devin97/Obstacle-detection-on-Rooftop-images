from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from misc import *
import pdb
import dehaze22  as net
import torchvision.models as models
import h5py
import torch.nn.functional as F
from skimage import measure
import numpy as np
import cv2
import time
import math

# // Importing Modules // #
from generate_testsample import generate_h5

def dehaze(imgdir, workers=0):
    """
    Parameters :-
    imgdir - path to image(s) dir
    val_imgdir - path to validation images
    workers - number of data loading workers
    """

    opt = {
        'imgdir': imgdir,
        'exp': 'sample',
        'dataset': 'pix2pix',
        'dataroot': '',
        'originalSize': 1024,
        'imageSize': 1024,
        'batchSize': 1,
        'workers': workers,
        'valDataroot': './facades/test_cvpr',
        'valBatchSize': 1,
        'inputChannelSize': 3,
        'outputChannelSize': 3,
        'netG': 'netG_epoch_8.pth',
        'poolSize': 50,
        'lambdaGAN': 0.01,
        'lambdaIMG': 1,
        'mode': 'B2A',
        'ngf': 64,
        'ndf': 64
    }

    #__________Scaling Images for Dehazing (512 x 512)___________________
    scaled_img_dir = './scaled_imgs' # directory to store 512x512 images
    if not os.path.exists(scaled_img_dir):
        os.makedirs(scaled_img_dir)
    for i in os.listdir(imgdir):
        img = cv2.imread(os.path.join(imgdir, i))
        scaled_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(scaled_img_dir, i), scaled_image)

    generate_h5(scaled_img_dir) # Generating h5 files for the scaled images
    create_exp_dir(opt['exp'])
    opt['manualSeed'] = random.randint(1, 10000)
    # opt.manualSeed = 101
    random.seed(opt['manualSeed'])
    torch.manual_seed(opt['manualSeed'])
    torch.cuda.manual_seed_all(opt['manualSeed'])
    print("Random Seed: ", opt['manualSeed'])

    # get dataloader
    dataloader = getLoader(opt['dataset'],
                           opt['dataroot'],
                           opt['originalSize'],
                           opt['imageSize'],
                           opt['batchSize'],
                           opt['workers'],
                           mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           split='train',
                           shuffle=True,
                           seed=opt['manualSeed'])
    opt['dataset']='pix2pix_val'

    valDataloader = getLoader(opt['dataset'],
                              opt['valDataroot'],
                              opt['imageSize'], #opt.originalSize,
                              opt['imageSize'],
                              opt['valBatchSize'],
                              opt['workers'],
                              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                              split='Train',
                              shuffle=False,
                              seed=opt['manualSeed'])

    # get logger
    trainLogger = open('%s/train.log' % opt['exp'], 'w')

    def gradient(y):
        gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

        return gradient_h, gradient_y


    ngf = opt['ngf']
    ndf = opt['ndf']
    inputChannelSize = opt['inputChannelSize']
    outputChannelSize= opt['outputChannelSize']


    netG = net.dehaze(inputChannelSize, outputChannelSize, ngf)




    if opt['netG'] != '':
      netG.load_state_dict(torch.load(opt['netG']), strict=False)
    print(netG)



    netG.train()


    target= torch.FloatTensor(opt['batchSize'], outputChannelSize, opt['imageSize'], opt['imageSize'])
    input = torch.FloatTensor(opt['batchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])




    val_target= torch.FloatTensor(opt['valBatchSize'], outputChannelSize, opt['imageSize'], opt['imageSize'])
    val_input = torch.FloatTensor(opt['valBatchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])
    label_d = torch.FloatTensor(opt['batchSize'])


    target = torch.FloatTensor(opt['batchSize'], outputChannelSize, opt['imageSize'], opt['imageSize'])
    input = torch.FloatTensor(opt['batchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])
    depth = torch.FloatTensor(opt['batchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])
    ato = torch.FloatTensor(opt['batchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])


    val_target = torch.FloatTensor(opt['valBatchSize'], outputChannelSize, opt['imageSize'], opt['imageSize'])
    val_input = torch.FloatTensor(opt['valBatchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])
    val_depth = torch.FloatTensor(opt['valBatchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])
    val_ato = torch.FloatTensor(opt['valBatchSize'], inputChannelSize, opt['imageSize'], opt['imageSize'])




    # NOTE: size of 2D output maps in the discriminator
    sizePatchGAN = 30
    real_label = 1
    fake_label = 0

    # image pool storing previously generated samples from G
    imagePool = ImagePool(opt['poolSize'])

    # NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
    lambdaGAN = opt['lambdaGAN']
    lambdaIMG = opt['lambdaIMG']

    netG.cuda()

    target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
    val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

    target = Variable(target, volatile=True)
    input = Variable(input,volatile=True)
    depth = Variable(depth,volatile=True)
    ato = Variable(ato,volatile=True)

    label_d = Variable(label_d.cuda())



    def psnr(img1, img2):
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        # PIXEL_MAX = 1

        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    # NOTE training loop
    ganIterations = 0
    index=0
    psnrall = 0
    ssimall=0
    iteration = 0
    # print(1)
    for epoch in range(1):
      for i, data in enumerate(valDataloader, 0):
        t0 = time.time()

        if opt['mode'] == 'B2A':
            input_cpu, target_cpu, depth_cpu, ato_cpu = data
        elif opt['mode'] == 'A2B' :
            input_cpu, target_cpu, depth_cpu, ato_cpu = data
        batch_size = target_cpu.size(0)
        # print(i)
        target_cpu, input_cpu, depth_cpu, ato_cpu = target_cpu.float().cuda(), input_cpu.float().cuda(), depth_cpu.float().cuda(), ato_cpu.float().cuda()
        # get paired data
        target.data.resize_as_(target_cpu).copy_(target_cpu)
        input.data.resize_as_(input_cpu).copy_(input_cpu)
        depth.data.resize_as_(depth_cpu).copy_(depth_cpu)
        ato.data.resize_as_(ato_cpu).copy_(ato_cpu)
        #


        x_hat, tran_hat, atp_hat, dehaze2= netG(input)

        zz=x_hat.data

        iteration=iteration+1

        index2 = 0
        directory='./result_cvpr/Dehazed'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for i in range(opt['valBatchSize']):
            index=index+1
            print(index)
            zz1=zz[index2,:,:,:]

            #zz1 = cv2.resize(zz1, (300, 300), interpolation=cv2.INTER_CUBIC)
            vutils.save_image(zz1, os.path.join(directory, str(index-1)+'_DCPCN.png'), normalize=True, scale_each=False)
        for i in os.listdir('./result_cvpr/Dehazed'):
            #print(i)
            img = cv2.imread(os.path.join('./result_cvpr/Dehazed', i))
            rescaled_img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join('./result_cvpr/Dehazed', i), rescaled_img)
    trainLogger.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
    parser.add_argument('--imgdir', required=True, default='', help='directory of image(s)')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    opt = parser.parse_args()
    print(vars(opt))

    dehaze(vars(opt))
