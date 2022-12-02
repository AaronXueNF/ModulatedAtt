import io
import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchvision import transforms
from torchvision.models import *
from pytorch_msssim import ms_ssim
from torch.utils.data import Dataset
from scipy.optimize import curve_fit

from models.net import find_net
from models.rate_distortion import RateDistortionLoss
from datasets import Datasets
from utils import AverageMeter, parse_json_param, concat_images
from utils import compute_psnr, compute_msssim, compute_bpp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

pd_columns = ['target0', 'real0', 'target1', 'real1', 'target2', 'real2']

def lmbda2level_alpha(lmbda, net):
    lmbda_list = net.lmbda

    # if over range
    if lmbda < lmbda_list[0]:
        return 0, 1.0, False
    elif lmbda > lmbda_list[-1]:
        return 8, 0.0, False

    for i in range(0, len(lmbda_list) - 1):
        lower = lmbda_list[i]
        upper = lmbda_list[i + 1]
        distance = upper - lower
        over = lmbda - lower
        if lower <= lmbda <= upper:
            a = i
            b = 1 - (over / distance)
            return int(a), float(b), True

    return -1, 0.0, False

def exp_bpp_lmbda(bpp, a, b, c):
    return a * np.exp(b * bpp) - c


def fit_rd(img, net):
    low, mid, high = 1, len(net.train_qualities) // 2, len(net.train_qualities) - 2
    sample_rate = [
        net.train_qualities[low],
        net.train_qualities[mid],
        net.train_qualities[high]
    ]

    bpps = np.zeros((3))
    lmbdas = np.zeros((3))

    for i in range(0,len(sample_rate)):
        level, alpha = sample_rate[i]
        lmbdas[i] = alpha * net.lmbda[level] + (1 - alpha) * net.lmbda[level + 1]
        bpps[i] = compute_bpp(net.forward(img, level, alpha)).item()

    lmbdas = lmbdas * (255 ** 2)
    (a, b, c), _ = curve_fit(exp_bpp_lmbda, bpps, lmbdas)
    return a, b, c


def test(net, test_loader):
    """
    Eval model for one epoch
    """
    net.eval()  # Set model to training mode
    device = next(net.parameters()).device

    result = pd.DataFrame([], columns=pd_columns)

    i = 0
    with torch.no_grad():
        # loop over all images
        for img in test_loader:
            img = img.to(device)
            pix_num = img.shape[0] * img.shape[2] * img.shape[3]

            # generate 3 target bpp point
            max_level, max_alpha = net.train_qualities[-1]
            min_level, min_alpha = net.train_qualities[0]
            out_max = net(img, max_level, max_alpha)
            out_min = net(img, min_level, min_alpha)
            max_bpp = compute_bpp(out_max).item()
            min_bpp = compute_bpp(out_min).item()
            bpp_range = max_bpp - min_bpp
            target_bpp_list = [min_bpp + i * (bpp_range / 5) for i in range(1, 4)]

            print(f"In image {i}, max bpp: {max_bpp}, min bpp: {min_bpp}, "
                  f"target bpps: {str(target_bpp_list)}")

            # calculate RD parameter:
            a, b, c = fit_rd(img, net)
            print(f"image {i}, a {a}, b {b}, c {c}")

            record = []
            for target_bpp in target_bpp_list:
                target_lmbda = exp_bpp_lmbda(target_bpp, a, b, c) / (255 ** 2)
                level, alpha, _ = lmbda2level_alpha(target_lmbda, net)
                compress_out = net.compress(img, level, alpha)
                y_bits = len(compress_out['strings'][0][0]) * 8
                z_bits = len(compress_out['strings'][1][0]) * 8
                compress_bits = y_bits + z_bits
                real_bpp = compress_bits / pix_num
                record.extend([target_bpp, real_bpp])

            result.loc[len(result)] = record

            print(f"finish image {i}, result: {str(record)}")
            i += 1

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
    result_pd.to_csv(args["rc_out"])
    print(result_pd)


if __name__ == "__main__":
    args = parse_json_param(sys.argv[1] if len(sys.argv) - 1 else "./hyperparam/test.json")

    global_start_time = time.time()
    main(args)

    print("Done. Time usage: {}".format(time.time() - global_start_time))
