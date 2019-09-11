from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

UPDATE_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(description='Train and validate a DAMSM')
    parser.add_argument('--conf', dest='config_name', default='DAMSM', type=str)
    parser.add_argument('--val', dest='b_validation', default=False, type=bool)
    parser.add_argument('--df', dest='df_dim', default=96, type=int)
    parser.add_argument('--gf', dest='gf_dim', default=48, type=int)
    parser.add_argument('--emb', dest='emb_dim', default=256, type=int)
    parser.add_argument('--rnum', dest='r_num', default=3, type=int)
    parser.add_argument('--rnn', dest='rnn_type', default='LSTM', type=str)
    parser.add_argument('--wnum', dest='words_num', default=42, type=int)
    parser.add_argument('--batch', dest='batch_size', default=48, type=int)
    parser.add_argument('--flaggg', dest='flaggg', default=True, type=bool)
    parser.add_argument('--dlr', dest='discriminator_lr', default=1e-4, type=float)
    parser.add_argument('--glr', dest='generator_lr', default=1e-4, type=float)
    parser.add_argument('--elr', dest='encoder_lr', default=1e-3, type=float)
    parser.add_argument('--nete', dest='net_e', default='', type=str)
    parser.add_argument('--netg', dest='net_g', default='', type=str)
    parser.add_argument('--lam', dest='lambda_value', default=50, type=int)
    parser.add_argument('--gam1', dest='gamma1', default=4, type=int)
    parser.add_argument('--gam2', dest='gamma2', default=5, type=int)
    parser.add_argument('--gam3', dest='gamma3', default=20, type=int)
    parser.add_argument('--base', dest='base_size', default=299, type=int)
    parser.add_argument('--bnum', dest='branch_num', default=1, type=int)
    parser.add_argument('--w', dest='workers', default=12, type=int)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def train(dataloader, cnn_model, rnn_model, batch_size,
          labels, optimizer, epoch, ixtoword, image_dir):
    cnn_model.train()
    rnn_model.train()
    s_total_loss0 = 0
    s_total_loss1 = 0
    w_total_loss0 = 0
    w_total_loss1 = 0
    s_total_loss = 0
    w_total_loss = 0
    count = (epoch + 1) * len(dataloader)
    start_time = time.time()
    for step, data in enumerate(dataloader, 0):
        rnn_model.zero_grad()
        cnn_model.zero_grad()
        imgs, captions, cap_lens, class_ids = prepare_data(data)
        # words_features: batch_size x nef x 17 x 17
        # sent_code: batch_size x nef
        words_features, sent_code = cnn_model(imgs[-1])
        nef, att_sze = words_features.size(1), words_features.size(2)
        hidden = rnn_model.init_hidden(batch_size)
        # words_emb: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        w_loss0, w_loss1, attn_maps = words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss0 += w_loss0.data
        w_total_loss1 += w_loss1.data
        loss = w_loss0 + w_loss1
        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        loss += s_loss0 + s_loss1
        s_total_loss0 += s_loss0.data
        s_total_loss1 += s_loss1.data     
        loss.backward()
        w_total_loss += (w_loss0 + w_loss1).data
        s_total_loss += (s_loss0 + s_loss1).data
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()
        if step % UPDATE_INTERVAL == 0:
          count = epoch * len(dataloader) + step
          s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
          s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL
          w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
          w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL
          elapsed = time.time() - start_time
          print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                's_loss {:5.2f} {:5.2f} | '
                'w_loss {:5.2f} {:5.2f}'
                .format(epoch, step, len(dataloader),
                        elapsed * 1000. / UPDATE_INTERVAL,
                        s_cur_loss0, s_cur_loss1,
                        w_cur_loss0, w_cur_loss1))
          s_total_loss0 = 0
          s_total_loss1 = 0
          w_total_loss0 = 0
          w_total_loss1 = 0
          start_time = time.time()
          # attention maps
          img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze, batch_size, cfg.TEXT.WORDS_NUM, lr_imgs=None)
          if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/attention_maps%d.png' % (image_dir, step)
            im.save(fullpath)
    s_cur_loss = s_total_loss.item() / len(dataloader)
    w_cur_loss = w_total_loss.item() / len(dataloader)
    print('-' * 90)
    print('| end epoch {:3d} | train loss {:5.2f} {:5.2f} |'
                     .format(epoch, s_cur_loss, w_cur_loss))
    print('-' * 90)
    return count

def evaluate(dataloader, cnn_model, rnn_model, batch_size, labels):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss = 0
    w_total_loss = 0
    for step, data in enumerate(dataloader, 0):
        real_imgs, captions, cap_lens, class_ids = prepare_data(data)
        words_features, sent_code = cnn_model(real_imgs[-1])
        if step == len(dataloader)-1:
          batch_size = len(subset_val)-(len(dataloader)-1)*batch_size
          labels = Variable(torch.LongTensor(range(batch_size)))
          labels = labels.cuda()          
        hidden = rnn_model.init_hidden(batch_size)
        words_emb, sent_emb = rnn_model(captions, cap_lens, hidden)
        w_loss0, w_loss1, attn = words_loss(words_features, words_emb, labels, cap_lens, class_ids, batch_size)
        w_total_loss += (w_loss0 + w_loss1).data
        s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        s_total_loss += (s_loss0 + s_loss1).data
    s_cur_loss = s_total_loss.item() / len(dataloader)
    w_cur_loss = w_total_loss.item() / len(dataloader)
    return s_cur_loss, w_cur_loss

def build_models():
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    lr = cfg.TRAIN.ENCODER_LR
    if cfg.TRAIN.NET_E != '':
      state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage) 
      text_encoder.load_state_dict(state_dict)
      print('Load {}'.format(cfg.TRAIN.NET_E))
      name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
      state_dict = torch.load(name, map_location=lambda storage, loc: storage)
      image_encoder.load_state_dict(state_dict)
      print('Load {}'.format(name))
      istart = cfg.TRAIN.NET_E.rfind('_') + 8
      iend = cfg.TRAIN.NET_E.rfind('.')
      start_epoch = cfg.TRAIN.NET_E[istart:iend]
      start_epoch = int(start_epoch) + 1
      print('start_epoch', start_epoch)
      # initial lr with the right value
      # note that the turning point is always epoch 114
      if start_epoch < 114:
        lr = cfg.TRAIN.ENCODER_LR * (0.98 ** start_epoch)
      else:
        lr = cfg.TRAIN.ENCODER_LR / 10   
    if cfg.CUDA:
      text_encoder = text_encoder.cuda()
      image_encoder = image_encoder.cuda()
      labels = labels.cuda()
    return text_encoder, image_encoder, labels, start_epoch, lr

  
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
  ##################################time & dir##################################
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
  output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
  model_dir = os.path.join(output_dir, 'Model')
  image_dir = os.path.join(output_dir, 'Image')
  mkdir_p(model_dir)
  mkdir_p(image_dir)
  os.environ['CUDA_VISIBLE_DEVICES']='0'
  cudnn.benchmark = True
  ###########################TextDataset & loader###############################
  imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
  batch_size = cfg.TRAIN.BATCH_SIZE
  image_transform = transforms.Compose([
      transforms.Resize(int(imsize * 76 / 64)),
      transforms.RandomCrop(imsize),
      transforms.RandomHorizontalFlip()])
  dataset = TextDataset(cfg.DATA_DIR, 'train',
                        base_size=cfg.TREE.BASE_SIZE,
                        transform=image_transform)
  assert dataset
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, drop_last=True,
      shuffle=True, num_workers=int(cfg.WORKERS))
  dataset_val = TextDataset(cfg.DATA_DIR, 'test',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
  val_idx = np.load('/.local/AttnGAN/data/FashionGen/test/val_idx.npy')
  subset_val = torch.utils.data.Subset(dataset_val, val_idx)
  dataloader_val = torch.utils.data.DataLoader(
    subset_val, batch_size=batch_size, drop_last=False,
    shuffle=True, num_workers=int(cfg.WORKERS))
  #####################################Train####################################
  text_encoder, image_encoder, labels, start_epoch, lr = build_models()
  para = list(text_encoder.parameters())
  for v in image_encoder.parameters():
      if v.requires_grad:
         para.append(v)
  # At any point you can hit Ctrl + C to break out of training early.
  try:
      for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
          optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
          epoch_start_time = time.time()
          count = train(dataloader, image_encoder, text_encoder,
                        batch_size, labels, optimizer, epoch,
                        dataset.ixtoword, image_dir)
          if len(dataloader_val) > 0:
            s_loss, w_loss = evaluate(dataloader_val, image_encoder,
                                      text_encoder, batch_size, labels)
            print('| end epoch {:3d} | valid loss {:5.2f} {:5.2f} | lr {:.5f} | train + val {:5.2f} min |'
                     .format(epoch, s_loss, w_loss, lr, (time.time()-epoch_start_time)/60))
          print('-' * 90)
          if lr > cfg.TRAIN.ENCODER_LR/10.: 
             lr *= 0.98

          if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
              epoch == cfg.TRAIN.MAX_EPOCH):
              torch.save(image_encoder.state_dict(),
                         '%s/image_encoder%d.pth' % (model_dir, epoch))
              torch.save(text_encoder.state_dict(),
                         '%s/text_encoder%d.pth' % (model_dir, epoch))
              print('Save image encoder and text encoder.')
  except KeyboardInterrupt:
    print('-' * 90)
    print('Exiting from training early')
