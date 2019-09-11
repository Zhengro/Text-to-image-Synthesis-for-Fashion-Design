from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--conf', dest='config_name', default='AttnGAN', type=str)
    parser.add_argument('--val', dest='b_validation', default=False, type=bool)
    parser.add_argument('--df', dest='df_dim', default=96, type=int)
    parser.add_argument('--gf', dest='gf_dim', default=48, type=int)
    parser.add_argument('--emb', dest='emb_dim', default=256, type=int)
    parser.add_argument('--rnum', dest='r_num', default=3, type=int)
    parser.add_argument('--rnn', dest='rnn_type', default='LSTM', type=str)
    parser.add_argument('--wnum', dest='words_num', default=15, type=int)
    parser.add_argument('--batch', dest='batch_size', default=48, type=int)
    parser.add_argument('--flaggg', dest='train_flag', default=True, type=bool)
    parser.add_argument('--dlr', dest='discriminator_lr', default=1e-4, type=float)
    parser.add_argument('--glr', dest='generator_lr', default=1e-4, type=float)
    parser.add_argument('--elr', dest='encoder_lr', default=1e-3, type=float)
    parser.add_argument('--nete', dest='net_e', default='', type=str)
    parser.add_argument('--netg', dest='net_g', default='', type=str)
    parser.add_argument('--lam', dest='lambda_value', default=1, type=int)
    parser.add_argument('--gam1', dest='gamma1', default=5, type=int)
    parser.add_argument('--gam2', dest='gamma2', default=5, type=int)
    parser.add_argument('--gam3', dest='gamma3', default=10, type=int)
    parser.add_argument('--base', dest='base_size', default=64, type=int)
    parser.add_argument('--bnum', dest='branch_num', default=2, type=int)
    parser.add_argument('--w', dest='workers', default=16, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')           
    args = parser.parse_args()
    return args


def gen_example(wordtoix, algo):                                                
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '%s/example_sentences.txt' % (cfg.DATA_DIR) 
    data_dic = {}
    with open(filepath, "r") as f:
        print('Load from:', filepath)
        sentences = f.read().decode('utf8').split('\n')
        # a list of indices for a sentence
        captions = []
        cap_lens = []
        for sent in sentences:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(sent.lower())
            if len(tokens) == 0:
                print('sent', sent)
                continue

            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))
    max_len = np.max(cap_lens)

    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for j in range(len(captions)):
      for i in range(len(captions)):
          idx = sorted_indices[i]
          cap = captions[idx]
          c_len = len(cap)
          cap_array[i, :c_len] = cap
      data_dic[j] = [cap_array, cap_lens, sorted_indices]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    cfg.CONFIG_NAME = args.config_name
    cfg.B_VALIDATION = args.b_validation
    cfg.GAN.DF_DIM = args.df_dim
    cfg.GAN.GF_DIM = args.gf_dim
    cfg.TEXT.EMBEDDING_DIM = args.emb_dim
    cfg.GAN.R_NUM = args.r_num
    cfg.RNN_TYPE = args.rnn_type
    cfg.TEXT.WORDS_NUM = args.words_num
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.FLAG = args.train_flag
    cfg.TRAIN.DISCRIMINATOR_LR = args.discriminator_lr
    cfg.TRAIN.GENERATOR_LR = args.generator_lr
    cfg.TRAIN.ENCODER_LR = args.encoder_lr
    cfg.TRAIN.NET_E = args.net_e
    cfg.TRAIN.NET_G = args.net_g
    cfg.TRAIN.SMOOTH.GAMMA1 = args.gamma1
    cfg.TRAIN.SMOOTH.GAMMA2 = args.gamma2
    cfg.TRAIN.SMOOTH.GAMMA3 = args.gamma3
    cfg.TRAIN.SMOOTH.LAMBDA = args.lambda_value
    cfg.TREE.BASE_SIZE = args.base_size
    cfg.TREE.BRANCH_NUM = args.branch_num
    cfg.WORKERS = args.workers

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle, drop_last = 'train', True, True
        
    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=drop_last, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
