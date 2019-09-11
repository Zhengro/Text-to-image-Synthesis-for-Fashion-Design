from __future__ import print_function
from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from model import RNN_ENCODER
from torch.autograd import Variable

import os
import sys
import random
import pprint
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
if sys.version_info[0] == 2:
   import cPickle as pickle
else:
   import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def prepare_data(data):
    imgs, captions, captions_lens, class_ids = data
    # sort data by the captions_lens in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
           real_imgs.append(Variable(imgs[i]).cuda())
        else:
           real_imgs.append(Variable(imgs[i]))
    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    if cfg.CUDA:
      captions = Variable(captions).cuda()
      sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
      captions = Variable(captions)
      sorted_cap_lens = Variable(sorted_cap_lens)
    return [real_imgs, captions, sorted_cap_lens, class_ids, sorted_cap_indices.numpy()]
  

def parse_args():
    parser = argparse.ArgumentParser(description='Extract text embeddings')
    parser.add_argument('--conf', dest='config_name', default='embeddings', type=str)
    parser.add_argument('--val', dest='b_validation', default=False, type=bool)
    parser.add_argument('--df', dest='df_dim', default=96, type=int)
    parser.add_argument('--gf', dest='gf_dim', default=48, type=int)
    parser.add_argument('--emb', dest='emb_dim', default=256, type=int)
    parser.add_argument('--rnum', dest='r_num', default=3, type=int)
    parser.add_argument('--rnn', dest='rnn_type', default='LSTM', type=str)
    parser.add_argument('--wnum', dest='words_num', default=42, type=int)
    parser.add_argument('--batch', dest='batch_size', default=48, type=int)
    parser.add_argument('--flaggg', dest='flaggg', default=False, type=bool)
    parser.add_argument('--dlr', dest='discriminator_lr', type=float)
    parser.add_argument('--glr', dest='generator_lr', type=float)
    parser.add_argument('--elr', dest='encoder_lr', default=0.001, type=float)
    parser.add_argument('--nete', dest='net_e', default='', type=str)
    parser.add_argument('--netg', dest='net_g', default='', type=str)
    parser.add_argument('--lam', dest='lambda_value', default=50, type=int)
    parser.add_argument('--base', dest='base_size', default=299, type=int)
    parser.add_argument('--bnum', dest='branch_num', default=1, type=int)
    parser.add_argument('--w', dest='workers', default=16, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def extractor(dataloader, rnn_model, batch_size, len_dataset, save_dir):
    embeddings = np.zeros((len_dataset, cfg.TEXT.EMBEDDING_DIM))
    num_batches = len(dataloader)
    rnn_model.eval()
    for step, data in enumerate(dataloader, 0):
        _, captions, cap_lens, _, sorted_cap_indices = prepare_data(data) 
        if step < num_batches-1:
          hidden = rnn_model.init_hidden(batch_size)
        else:
          hidden = rnn_model.init_hidden(len_dataset%batch_size)   
        _, sent_emb = rnn_model(captions, cap_lens, hidden)          
        if step == num_batches - 1:
          lb, ub = step * batch_size, len_dataset 
        else:
          lb, ub = step * batch_size, (step + 1) * batch_size
          
        text_embeddings = np.zeros((int(ub-lb), cfg.TEXT.EMBEDDING_DIM))
        right_embeddings = np.zeros((int(ub-lb), cfg.TEXT.EMBEDDING_DIM))
        text_embeddings = sent_emb.detach().cpu()
        for i in range(int(ub-lb)):
          right_embeddings[sorted_cap_indices[i], :] = text_embeddings[i, :]
        embeddings[lb:ub, :] = right_embeddings
        
        if (step+1)%100 == 0:
          print('{:1d}/{:1d} batches'.format(step, num_batches))
    np.save(save_dir, embeddings)
    print('Save text embeddings successfully.')
    return None


def build_text_encoder(ntokens):
    text_encoder = RNN_ENCODER(ntokens, nhidden=cfg.TEXT.EMBEDDING_DIM)
    if cfg.TRAIN.NET_E != '':
      state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage) 
      text_encoder.load_state_dict(state_dict)
      print('Load ', cfg.TRAIN.NET_E)
    if cfg.CUDA:
      text_encoder = text_encoder.cuda()
    return text_encoder
  
#####################################__main__###################################
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
  cfg.TRAIN.FLAG = args.flaggg
  cfg.TRAIN.DISCRIMINATOR_LR = args.discriminator_lr
  cfg.TRAIN.GENERATOR_LR = args.generator_lr
  cfg.TRAIN.ENCODER_LR = args.encoder_lr
  cfg.TRAIN.NET_E = args.net_e
  cfg.TRAIN.NET_G = args.net_g
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

  os.environ['CUDA_VISIBLE_DEVICES']='0'
#   torch.cuda.set_device(cfg.GPU_ID)
  cudnn.benchmark = True
  ###########################TextDataset & loader###############################
  imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
  batch_size = cfg.TRAIN.BATCH_SIZE
  image_transform = transforms.Compose([
      transforms.Resize(int(imsize * 76 / 64)),
      transforms.RandomCrop(imsize),
      transforms.RandomHorizontalFlip()])
  # dataset = TextDataset(cfg.DATA_DIR, 'train',
  #                           base_size=cfg.TREE.BASE_SIZE,
  #                           transform=image_transform)
  # assert dataset
  # dataloader = torch.utils.data.DataLoader(
  #   dataset, batch_size=batch_size, drop_last=False,
  #   shuffle=False, num_workers=int(cfg.WORKERS))
  
  dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
  assert dataset_val
  dataloader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=batch_size, drop_last=False,
    shuffle=False, num_workers=int(cfg.WORKERS))
  ntokens = 501+1  # 6872+1
  ##############################################################################
  text_encoder = build_text_encoder(ntokens)
  # save_dir = '/.local/AttnGAN/data/FashionSynthesis/train/embeddings/final'
  save_dir = '/.local/AttnGAN/data/FashionSynthesis/test/embeddings/final'
  # extractor(dataloader, text_encoder, batch_size, len(dataset), save_dir)
  extractor(dataloader_val, text_encoder, batch_size, len(dataset_val), save_dir)  
