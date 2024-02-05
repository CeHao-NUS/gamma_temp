import pdb
import os
from visual_model.object_articulation_part import gamma_model_net
import argparse
from datasets.gamma_dataset import GammaDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from typing import List, Tuple, Dict, Optional, Any, Union
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

def collate_fn(batch: List[Dict]):
    keys = batch[0].keys()
    return {
        key: default_collate([item[key]for item in batch])
        if batch[0][key] is not None
        else None
        for key in keys
    }

def get_train_loader(args):
    train_dataset = GammaDataset(args.train_data_path, point_num=args.num_points)
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return loader

def get_test_loader(args):
    dataset = GammaDataset(args.test_data_path, point_num=args.num_points)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args, writer):
    model = gamma_model_net(pointnet_type=args.pointnet_type, num_point=int(args.num_points), device=args.device).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=args.lr / 100.0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    optimizer.zero_grad()
    optimizer.step()
    scheduler_warmup.step()
    train_step = 0
    min_loss = 20
    train_load = get_train_loader(args)
    test_load = get_test_loader(args)
    for epoch in tqdm(range(int(args.train_epoch))):
        model.train()
        epoch_loss = 0
        sem_acc = 0
        ious = 0
        epoch_step = 0
        for sample in tqdm(train_load):
            train_step = train_step + 1
            losses, result_dict = model.get_loss(sample)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            epoch_loss += losses["total_loss"].item()
            sem_acc += result_dict["sem_all_accu"].item()
            ious += result_dict["iou"].item()
            epoch_step += 1
            loss_keys = losses.keys()
            for loss_key in loss_keys:
                writer.add_scalar("train/" + loss_key, losses[loss_key].item(), train_step)
            if train_step % 100 == 0:
                for loss_key in loss_keys:
                    print("train epoch: {}, step: {}  {}: {}".format(epoch, train_step, loss_key, losses[loss_key].item()))
                for result_key in result_dict:
                    print("train epoch: {}, step: {} {}: {}".format(epoch, train_step, result_key, result_dict[result_key].item()))
        print("train epoch:{} loss:{} sem_acc:{} iou:{}".format(epoch, epoch_loss/epoch_step, sem_acc/epoch_step, ious/epoch_step))
        writer.add_scalar("train/epoch_loss", epoch_loss / epoch_step, epoch)
        writer.add_scalar("train/epoch_sem_accu", sem_acc / epoch_step, epoch)
        scheduler_warmup.step()
        if epoch > 0 and epoch % 5 == 0:
            model.eval()
            epoch_loss = 0
            sem_acc = 0
            epoch_step = 0
            for sample in tqdm(test_load):
                losses, result_dict = model.get_loss(sample)
                optimizer.zero_grad()
                losses["total_loss"].backward()
                optimizer.step()
                epoch_loss += losses["total_loss"].item()
                sem_acc += result_dict["sem_all_accu"].item()
                epoch_step += 1
            writer.add_scalar("test/epoch_loss", epoch_loss / epoch_step, epoch)
            writer.add_scalar("test/epoch_sem_accu", sem_acc / epoch_step, epoch)
            print("test epoch:{} loss:{} sem_acc:{}".format(epoch, epoch_loss/epoch_step, sem_acc/epoch_step))
            if epoch_loss/epoch_step < min_loss:
                print("save fine-tune model, loss: ", epoch_loss/epoch_step)
                min_loss = epoch_loss/epoch_step
                save_path = args.log_dir + "/best.pth"
                print("best model: ", save_path, " save path: ", save_path)
                torch.save(model.state_dict(), save_path)
            if epoch % 10 == 0 and epoch > 0:
                save_path = args.log_dir + "/epoch" + str(epoch) + "_.pth"
                print("best model: ", save_path, " save path: ", save_path)
                torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bbox object')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_data_path', type=str, default=["/aidata/qiaojun/train_data/gamma/train/"])
    parser.add_argument('--test_data_path', type=str, default=["/aidata/qiaojun/train_data/gamma/test"])
    parser.add_argument('--pointnet_type', type=str, default="pointnet2_msg")
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument("--train_epoch", type=int, help="instructions file path", default=201)
    parser.add_argument("--batch_size", type=int, help="instructions file path", default=16)
    parser.add_argument("--num_workers", type=int, help="instructions file path", default=8)
    parser.add_argument("--lr", type=float, help="lrearning rate", default=0.001)
    parser.add_argument("--device", type=str, help="cuda or cpu", default="cuda")
    parser.add_argument("--log_dir", type=str, help="dataset list", default="/aidata/qiaojun/train_process/gamma/pointnet_articulation_model/")
    args = parser.parse_args()
    args.log_dir = args.log_dir + time.strftime("%Y-%m-%d-%H", time.localtime()) + "_" + args.pointnet_type + "_" + str(args.batch_size)
    os.makedirs(args.log_dir, exist_ok=True)
    print("Logging:", args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir, )
    setup_seed(args.seed)
    main(args, writer)