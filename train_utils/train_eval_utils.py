import math
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import train_utils as utils
import train_utils.train_skills as skills

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False):
    model.train()

    lr_scheduler = None
    if epoch == 0 and warmup is None:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = skills.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mean_loss = torch.zeros(1).to(device)
    enable_amp = True if "cuda" in device.type else False
    for i, [images, targets] in enumerate(data_loader):

        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_log = losses.item()

            mean_loss = (mean_loss * i + loss_log) / (i + 1)

            if not math.isfinite(loss_log):
                print(f"Loss is {loss_log}, stopping training")
                sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        now_lr = optimizer.param_groups[0]["lr"]

    return mean_loss, now_lr


