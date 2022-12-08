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
from datasets import Datasets
from utils import AverageMeter, parse_json_param, concat_images
from utils import configure_optimizers, adjust_lr, clip_gradient

print_interval = 500
output_interval = 2000


def val_epoch(net, criterion, val_loader, tensorboard_writer, epoch):
    """
    Eval model for one epoch
    """
    net.eval()  # Set model to training mode
    device = next(net.parameters()).device

    rd_loss_all = AverageMeter()
    r_loss_all = AverageMeter()
    d_loss_all = AverageMeter()
    aux_loss_all = AverageMeter()

    lmbda_list = []
    rd_loss_list = []
    r_loss_list = []
    d_loss_list = []

    with torch.no_grad():
        # loop over all lambdas
        for level, alpha in iter(net.train_qualities):
            lmbda = alpha * net.lmbda[level] + (1 - alpha) * net.lmbda[level + 1]
            rd_loss_this = AverageMeter()
            r_loss_this = AverageMeter()
            d_loss_this = AverageMeter()
            # loop over all images
            for img in val_loader:
                img = img.to(device)
                out_net = net(img, level, alpha)
                out_criterion = criterion(out_net, img, lmbda)

                rd_loss_this.update(out_criterion["loss"].item())
                r_loss_this.update(out_criterion["R_loss"].item())
                d_loss_this.update(out_criterion["D_loss"].item())
                aux_loss_all.update(net.aux_loss().item())
            
            lmbda_list.append(lmbda)
            rd_loss_list.append(rd_loss_this.avg)
            r_loss_list.append(r_loss_this.avg)
            d_loss_list.append(d_loss_this.avg)
            
            rd_loss_all.update(rd_loss_list[-1])
            r_loss_all.update(r_loss_list[-1])
            d_loss_all.update(d_loss_list[-1])

    # tensorboard
    tensorboard_writer.add_scalar('Val_rd_loss', rd_loss_all.avg, epoch)
    tensorboard_writer.add_scalar('Val_r_loss', r_loss_all.avg, epoch)
    tensorboard_writer.add_scalar('Val_d_loss', d_loss_all.avg, epoch)
    tensorboard_writer.add_scalar('Val_aux_loss', aux_loss_all.avg, epoch)

    quality_pd = pd.DataFrame([rd_loss_list, r_loss_list, d_loss_list], 
                    index=['rd_loss', 'r_loss', 'd_loss'], columns=lmbda_list)

    return rd_loss_all.avg, r_loss_all.avg, d_loss_all.avg, aux_loss_all.avg, quality_pd


def train_epoch(net, criterion, optimizers, schedulers, data_loaders, 
                best_val_loss, epoch, tensorboard_writer, args):
    """
    Train model for one epoch
    """
    optimizer, aux_optimizer = optimizers
    lr_scheduler, aux_lr_scheduler = schedulers
    train_loader, val_loader = data_loaders

    lr = optimizer.param_groups[0]['lr']
    aux_lr = aux_optimizer.param_groups[0]['lr']
    device = next(net.parameters()).device
    print(f"[training] In epoch {epoch}, lr: {lr}, Aux lr: {aux_lr}")

    rd_loss_avg = AverageMeter()
    d_loss_avg = AverageMeter()
    r_loss_avg = AverageMeter()
    aux_loss_avg = AverageMeter()
    batch_time_avg = AverageMeter()
    val_interval = len(train_loader) // args["val_per_epoch"]

    net.train()  # Set model to training mode   
    for batch, inputs in enumerate(train_loader):
        start_time = time.time()
        inputs = inputs.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        level, alpha = next(net.train_qualities_cycle)
        lmbda = alpha * net.lmbda[level] + (1 - alpha) * net.lmbda[level + 1]

        out = net(inputs, level, alpha)
        out_criterion = criterion(out, inputs, lmbda)

        # debug!
        if out_criterion["D_loss"] > 1e4 and epoch >= 1:
            torch.save({'img': inputs, 'level': level, 'alpha': alpha}, args["debug_img"])
            state = {'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'aux_optimizer': aux_optimizer.state_dict(),
                    'aux_lr_scheduler': aux_lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss}
            torch.save(state, args["debug_checkpoint"])
            if (args["skip_invalid_loss"]):
                print(f"[Training] epoch {epoch:3}: [{batch:4}/{len(train_loader):4}] | "
                      f" !!!WARNING!!! large loss detected, skip this iteration!")
                continue
            else:
                print(f"[Training] epoch {epoch:3}: [{batch:4}/{len(train_loader):4}] | "
                      f" !!!WARNING!!! large loss detected, need DEBUG!")
                exit(-114514)

        out_criterion["loss"].backward()

        # clip gradient and optimize compress net
        if args["grd_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args["grd_clip"])
            clip_gradient(optimizer, args["grd_clip"])
        optimizer.step()

        # optimize aux_loss
        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        # keep track of loss
        rd_loss_avg.update(out_criterion["loss"].item())
        r_loss_avg.update(out_criterion["R_loss"].item())
        d_loss_avg.update(out_criterion["D_loss"].item())
        aux_loss_avg.update(aux_loss.item())
        batch_time_avg.update(time.time() - start_time)

        # print trace information
        if batch % print_interval == 0:
            print(
                f"[Training] epoch {epoch:3}: [{batch:4}/{len(train_loader):4}] | "
                f"RD Loss: {rd_loss_avg.avg:.4f} | R:{r_loss_avg.avg:.4f}, D:{d_loss_avg.avg:.4f} | "
                f"Aux loss: {aux_loss.item():.2f} | Batch time: {batch_time_avg.avg:.4f} |"
            )

        # evaluation and save model
        # if not add 1, when len(dataloader) % 2 == 0, only run 1 times when interval = 2
        # if batch % val_interval == 0:     # only for debug
        if batch and (batch + 1) % val_interval == 0: 
            val_rd, val_r, val_d, val_aux, quality_pd= val_epoch(
                net, criterion, val_loader, tensorboard_writer, epoch)

            state = {'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'aux_optimizer': aux_optimizer.state_dict(),
                    'aux_lr_scheduler': aux_lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss}
            torch.save(state, args["current_checkpoint"])
            
            if val_rd < best_val_loss:
                print("[Evaluating] update best loss and save best model...")
                best_val_loss = val_rd
                torch.save(state, args["best_checkpoint"])
    
            # adjust lr according to iteration
            if epoch >= args["lr_down_afterEpoch"]:
                lr_scheduler.step(val_rd)
            if epoch >= args["lr_aux_down_afterEpoch"]:
                aux_lr_scheduler.step(val_aux)

            print(
                f"[Evaluating] epoch {epoch:3}: "
                f"RD Loss: {val_rd:.4f} | best RD loss: {best_val_loss:.4f} | "
                f"R loss: {val_r:.4f} | D Loss: {val_d:.4f} | Aux loss: {val_aux:.4f}\n"
                f"[Evaluating] detailed RD performance:"
            )
            print(quality_pd)

            net.train()
    
    # tensorboard
    tensorboard_writer.add_scalar('Train_rd_loss', rd_loss_avg.avg, epoch)
    tensorboard_writer.add_scalar('Train_r_loss', r_loss_avg.avg, epoch)
    tensorboard_writer.add_scalar('Train_d_loss', d_loss_avg.avg, epoch)
    tensorboard_writer.add_scalar('Train_aux_loss', aux_loss_avg.avg, epoch)
    tensorboard_writer.add_scalar("lr", lr, epoch)
    tensorboard_writer.add_scalar("aux_lr", aux_lr, epoch)

    return rd_loss_avg.avg, best_val_loss


def main(args):
    if args["seed"] is not None:
        torch.manual_seed(args["seed"])
        rd.seed(args["seed"])
    device = args["device"]

    # define dataloader
    train_data = Datasets(args["train_set"], 256, train=True)      # load dataset and create data loader
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=8,
                                                   batch_size=args["batch_size"], shuffle=True)

    val_data = Datasets(args["val_set"], 512, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=1, shuffle=True)

    # define net
    net_name = args["net"]
    net = find_net(net_name)(
        metric=args["metric"], N=args["N"], M=args["M"], grad_proxy=args["grad_proxy"]).to(device)
    print(f"[Preparing] Using Net: {net_name}")

    # define optimizer
    optimizer, aux_optimizer = configure_optimizers(net, args["lr_init"], args["lr_aux_init"])

    # define scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args["lr_down_factor"], patience=args["lr_down_patience"], min_lr=1e-6)
    aux_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        aux_optimizer, mode='min', factor=args["lr_aux_down_factor"], patience=args["lr_aux_down_patience"], min_lr=1e-6)
        
    # define RD loss
    criterion = RateDistortionLoss(metric=args["metric"])


    # Tensorboard
    tensorboard_writer = SummaryWriter(args["summary"])

    # load pretrain model
    if args["use_pretrain"]:
        checkpoint = torch.load(args["pretrain_checkpoint"], map_location=args["device"])
        net.load_state_dict(checkpoint['state_dict'])
        print(f"[Preparing] Load pretrain model.")

    # load model and continue training
    if args["continue_training"]:
        checkpoint = torch.load(args["continue_checkpoint"], map_location=args["device"])
        last_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        aux_lr_scheduler.load_state_dict(checkpoint["aux_lr_scheduler"])
        print(f"[Preparing] Load last model and continue training.")
        if args["continue_optimizer"]:
            optimizer.load_state_dict(checkpoint['optimizer'])
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        if args["clear_best"]:
            best_val_loss = 999999.0
            print(f"[Preparing] clear best loss!")
    else:
        last_epoch = 0
        best_val_loss = 999999.0
        print(f"[Preparing] start new training.")

    print(f"[Training] Start from epoch {last_epoch}")
    data_loaders = (train_dataloader, val_dataloader)
    optimizers = (optimizer, aux_optimizer)
    schedulers = (lr_scheduler, aux_lr_scheduler)

    for epoch in range(last_epoch, args["epochs"]):
        train_loss, best_val_loss = train_epoch(
            net, criterion, optimizers, schedulers, data_loaders, 
            best_val_loss, epoch, tensorboard_writer, args
        )

    tensorboard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--json_args', type=str, default="")
    args = parse_json_param(parser.parse_args().json_args)

    global_start_time = time.time()
    main(args)

    print("Done. Time usage: {}".format(time.time() - global_start_time))



