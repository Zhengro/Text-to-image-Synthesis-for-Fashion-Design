from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a StackGAN network')
    parser.add_argument('--conf', dest='config_name', default='StackGAN', type=str)
    parser.add_argument('--embtype', dest='embedding_type', default='lstm', type=str)
    parser.add_argument('--emb', dest='emb_dim', default=256, type=int)
    parser.add_argument('--imsize', dest='img_size', default=64, type=int)
    parser.add_argument('--nstage', dest='num_stage', default=1, type=int)
    parser.add_argument('--rnum', dest='r_num', default=4, type=int)
    parser.add_argument('--flaggg', dest='train_flag', default=True, type=bool)
    parser.add_argument('--batch', dest='batch_size', default=256, type=int)
    parser.add_argument('--mepoch', dest='max_epoch', default=200, type=int)
    parser.add_argument('--dlr', dest='discriminator_lr', default=2e-4, type=float)
    parser.add_argument('--glr', dest='generator_lr', default=2e-4, type=float)
    parser.add_argument('--pepoch', dest='pretrained_epoch', default=0, type=int)
    parser.add_argument('--pmodel', dest='pretrained_model', default='', type=str)
    parser.add_argument('--stage1g', dest='stage1_g', default='', type=str)
    parser.add_argument('--netg', dest='net_g', default='', type=str)
    parser.add_argument('--netd', dest='net_d', default='', type=str)
    parser.add_argument('--w', dest='workers', default=16, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')           
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg.CONFIG_NAME = args.config_name
    cfg.EMBEDDING_TYPE = args.embedding_type
    cfg.TEXT.DIMENSION = args.emb_dim
    cfg.IMSIZE = args.img_size
    cfg.STAGE = args.num_stage
    cfg.GAN.R_NUM = args.r_num
    cfg.TRAIN.FLAG = args.train_flag
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.MAX_EPOCH = args.max_epoch
    cfg.TRAIN.DISCRIMINATOR_LR = args.discriminator_lr
    cfg.TRAIN.GENERATOR_LR = args.generator_lr
    cfg.TRAIN.PRETRAINED_EPOCH = args.pretrained_epoch
    cfg.TRAIN.PRETRAINED_MODEL = args.pretrained_model
    cfg.STAGE1_G = args.stage1_g
    cfg.NET_G = args.net_g
    cfg.NET_D = args.net_d
    cfg.WORKERS = args.workers
        
    print('Using config:')
    pprint.pprint(cfg)
    
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
        
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.IMSIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              embedding_type='lstm',
                              imsize=cfg.IMSIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir)
        algo.train(dataloader, cfg.STAGE)
