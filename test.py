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

from models.net import find_net
from models.rate_distortion import RateDistortionLoss
from datasets import Datasets
from utils import AverageMeter, parse_json_param, concat_images
from utils import compute_psnr, compute_msssim, compute_bpp

pd_columns = ['bpp', 'mse', 'psnr', 'ms-ssim']

def test(net, test_loader):
    """
    Eval model for one epoch
    """
    net.eval()  # Set model to training mode
    device = next(net.parameters()).device

    result = pd.DataFrame([],columns=pd_columns)

    with torch.no_grad():
        # loop over all lambdas
        for level, alpha in iter(net.train_qualities):
            lmbda = alpha * net.lmbda[level] + (1 - alpha) * net.lmbda[level + 1]
            
            mse_thisLevel = AverageMeter()
            psnr_thisLevel = AverageMeter()
            ms_ssim_thisLevel = AverageMeter()
            bpp_thisLevel = AverageMeter()

            # loop over all images
            for img in test_loader:
                img = img.to(device)
                out_net = net(img, level, alpha)
                mse, psnr = compute_psnr(img, out_net['x_hat'])
                ms_ssim = compute_msssim(img, out_net['x_hat'])
                bpp = compute_bpp(out_net)

                mse_thisLevel.update(mse.item())
                psnr_thisLevel.update(psnr.item())
                ms_ssim_thisLevel.update(ms_ssim)
                bpp_thisLevel.update(bpp.item())

            result.loc[lmbda] = [
                bpp_thisLevel.avg, mse_thisLevel.avg, 
                psnr_thisLevel.avg, ms_ssim_thisLevel.avg
                ]

            print(f"[Testing] finish lambda {lmbda}")

    return result


def main(args):
    device = args["device"]

    test_data = Datasets(args["test_set"], 512, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # define net
    net_name = args["net"]
    net = find_net(net_name)(metric=args["metric"]).to(device)
    print(f"[Preparing] Using Net: {net_name}")

    # load model
    checkpoint = torch.load(args["test_checkpoint"], map_location=args["device"])
    net.load_state_dict(checkpoint['state_dict'])

    result_pd = test(net, test_dataloader)
    result_pd.to_csv(args["rd_out"])
    print(result_pd)


if __name__ == "__main__":
    args = parse_json_param(sys.argv[1] if len(sys.argv) - 1 else "")

    global_start_time = time.time()
    main(args)

    print("Done. Time usage: {}".format(time.time() - global_start_time))



