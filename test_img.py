import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd
import random as rd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.find_net import find_net
from models.rate_distortion import RateDistortionLoss
from datasets import Datasets_withName
from utils import AverageMeter, parse_json_param, concat_images
from utils import compute_psnr, compute_msssim, compute_bpp

pd_columns = ['bpp', 'mse', 'psnr', 'ms-ssim']

def test(net, test_loader):
    """
    Eval model for one epoch
    """
    net.eval()  # Set model to training mode
    device = next(net.parameters()).device

    level, alpha = 10, 1.0
    lmbda = alpha * net.lmbda[level] + (1 - alpha) * net.lmbda[level + 1]

    for img, im_name in test_loader:
        img = img.to(device)
        pix_num = img.shape[0] * img.shape[2] * img.shape[3]
        out_forward = net.forward(img, level, alpha)
        out_compress = net.compress(img, level, alpha)
        img_hat = net.decompress(out_compress['strings'], out_compress['shape'], level, alpha)

        est_bpp = compute_bpp(out_forward)
        y_bits = len(out_compress['strings'][0][0]) * 8
        z_bits = len(out_compress['strings'][1][0]) * 8
        compress_bits = y_bits + z_bits
        real_bpp = compress_bits / pix_num

        mse, psnr = compute_psnr(img, img_hat['x_hat'])
        ms_ssim = compute_msssim(img, img_hat['x_hat'])


def main(args):
    device = args["device"]

    test_data = Datasets_withName(args["test_set"]) 
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # define net
    net_name = args["net"]
    net = find_net(net_name)(N=256, M=320, metric=args["metric"]).to(device)
    print(f"[Preparing] Using Net: {net_name}")

    # load model
    checkpoint = torch.load(args["test_checkpoint"])
    net.load_state_dict(checkpoint['state_dict'])
    net.update()

    result_pd = test(net, test_dataloader)
    result_pd.to_csv(args["rd_out"])
    print(result_pd)


if __name__ == "__main__":
    args = parse_json_param(sys.argv[1] if len(sys.argv) - 1 else "")

    global_start_time = time.time()
    main(args)

    print("Done. Time usage: {}".format(time.time() - global_start_time))



