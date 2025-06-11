import os
import math
import argparse
import random
import numpy as np

# torch
import torch
from torch.optim import AdamW
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.model_selection import train_test_split
from datetime import datetime
# vision imports

from torch.utils.data import DataLoader

from mmcv.utils import Config, DictAction

# dalle classes and utils

from utils.train_api import save_model, build_data_set, vis_signal, eval_KNN, collate_fn, calculate_module_size_in_gb, group_by_subject
from utils.logger import get_logger
from models.hybrid_tome import Hybrid_ToMe_Masked_Modeling, Hybrid_ToMe_ClS_convmerge
from utils.metrics import accuracy
import time

def train(model, train_data_loader, eval_data_loader, opt, sched, cfg, logger, output_dir):
    opt_params = cfg.train_setting.opt_params
    EPOCHS = opt_params.epochs

    total_step = 0
    model.train()
    max_acc = [0 for i in range(cfg.train_setting.sub_count)]
    best_epochs = [0 for i in range(cfg.train_setting.sub_count)]
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_iteration = 0
        epoch_acc = [0 for i in range(cfg.train_setting.sub_count)]
        for iter, data in enumerate(train_data_loader):
            model.train()
            loss, recon_list, label_eeg_list, mask_list = model(data=data,)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if iter % 1 == 0:
                lr = opt.param_groups[0]['lr']

            if isinstance(sched, torch.optim.lr_scheduler._LRScheduler):
                sched.step()
            else:
                sched.step_update(total_step)
            # torch.cuda.synchronize()
            total_step += 1
            epoch_iteration += 1
            epoch_loss += loss.mean().item()
        # log training info
        losses = {'loss': epoch_loss / epoch_iteration}
        log_messages = {
                'epoch': epoch,
                'lr': lr,
            }
        log_messages.update(losses)
        logger.info(log_messages)


        if epoch % 100 == 0:
            save_model(f'{output_dir}/vae_{epoch}.pt', model)

    save_model(f'{output_dir}/vae-final.pt', model)


def main(cfg, cfg_name):
    if(cfg.model.model_type == 'MaskedModeling'):
        train_dataset = build_data_set(cfg.train_setting.data, split='train', seed=cfg.train_setting.seed, mode=2)
        test_dataset = build_data_set(cfg.test_setting.data, split='test', seed=cfg.train_setting.seed, mode=2)
    else:
        train_dataset = build_data_set(cfg.train_setting.data, split='train', seed=cfg.train_setting.seed, mode=1)
        test_dataset = build_data_set(cfg.test_setting.data, split='test', seed=cfg.train_setting.seed, mode=1)
    cfg.test_lengths = test_dataset.lengths

    if(cfg.model.model_type == 'MergeConv'):
        model = Hybrid_ToMe_ClS_convmerge(cfg, train_dataset.channels, train_dataset.region_indeces, train_dataset.region_counts, train_dataset.class_weights).cuda()
    elif(cfg.model.model_type == 'MaskedModeling'):
        model = Hybrid_ToMe_Masked_Modeling(cfg, train_dataset.channels, train_dataset.region_indeces, train_dataset.region_counts, train_dataset.class_weights).cuda()
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # Only calculate for modules with parameters
            size_gb = calculate_module_size_in_gb(module)
            print(f"Module {name} size: {size_gb:.4f} GB")
        # Optional: Calculate total model size
    total_size_gb = calculate_module_size_in_gb(model)
    print(f"Total model size: {total_size_gb:.4f} GB")
    base_dir = cfg.train_setting.output_dir
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = f"{current_time}_{cfg_name}"
    output_dir = os.path.join(base_dir, directory_name)
    logger = None
    print("creating logger")
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(
        path="{0}/{1}.log".format(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
    logger.info("begin log")
    logger.info(cfg)

    opt_params = cfg.train_setting.opt_params
    train_data_loader = DataLoader(
        train_dataset, opt_params.batch_size, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(
        test_dataset, opt_params.batch_size, shuffle=False, num_workers=0)
    assert len(train_dataset) > 0, 'folder does not contain any images'

    # optimizer
    opt = AdamW(model.parameters(), lr=opt_params.learning_rate,
                weight_decay=opt_params.weight_decay)
    if opt_params.schedule_type == 'exp':
        sched = ExponentialLR(optimizer=opt, gamma=opt_params.lr_decay_rate)
    elif opt_params.schedule_type == 'cosine':
        sched = CosineLRScheduler(
            opt,
            t_initial=math.ceil(opt_params.epochs *
                                len(train_dataset) / opt_params.batch_size),
            lr_min=cfg.train_setting.opt_params.min_lr,
            warmup_lr_init=opt_params.warmup_ratio * opt_params.learning_rate,
            warmup_t=opt_params.warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    if len(args.eval) == 0:
        train(model, train_data_loader, test_data_loader, opt, sched, cfg, logger, output_dir)
    else:
        ckpt = torch.load(args.eval)['weights']
        if 'module' not in list(ckpt.keys())[0]:
            new_ckpt = {}
            for key in list(ckpt.keys()):
                new_ckpt[f'module.{key}'] = ckpt[key]
            model.load_state_dict(
                new_ckpt,
            )
        else:
            model.load_state_dict(
                ckpt,
            )


if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file. If the value to '
                        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                        'Note that the quotation marks are necessary and that no white space '
                        'is allowed.')
    parser.add_argument('--eval', type=str, default='')
    parser.add_argument('--save_interval', type=int, default=50)
    args = parser.parse_args()
    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    main(cfg, cfg_name)
