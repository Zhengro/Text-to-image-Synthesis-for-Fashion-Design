from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict, OrderedDict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib 
matplotlib.use('agg')
import numpy.random as random
import scipy.io
if sys.version_info[0] == 2:
   import cPickle as pickle
else:
   import pickle

   
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
    return [real_imgs, captions, sorted_cap_lens, class_ids]


def get_imgs(img, imsize, transform=None, normalize=None):
    img = Image.fromarray(img, 'RGB')
    width, height = img.size
    if transform is not None:
      img = transform(img)
    ret = []
    if cfg.GAN.B_DCGAN:
      ret = [normalize(img)]
    else:
       for i in range(cfg.TREE.BRANCH_NUM):
          if i < (cfg.TREE.BRANCH_NUM - 1):
             re_img = transforms.Resize(imsize[i])(img)
          else:
             re_img = img
          ret.append(normalize(re_img))
    return ret


def captions_statistics(filepath, option): # option -- 'train', 'test', 'all'
    with open(filepath, 'rb') as f:
        captions = pickle.load(f)
    f.close()
    length = []
    if option=='all':
      longest = len(captions[0][0])
      captions_all = captions[0]+captions[1]
    elif option=='train':
      longest = len(captions[0][0])
      captions_all = captions[0]
    else:
      longest = len(captions[1][0])
      captions_all = captions[1]
    for cap in captions_all:
        if len(cap) > longest:
          longest = len(cap)
        length.append(len(cap))
    hist, _ = np.histogram(length, bins=np.arange(1, longest+1))
    hist = hist/len(captions_all)
    proportion = np.ones((1, longest-1))*hist[0]
    for i in range(1, longest-1):
        proportion[0, i] = proportion[0, i-1] + hist[i] 
    plt.hist(length, bins=np.arange(1, longest))
    plt.show()
    median = np.median(length)
    mean = np.mean(length)
    print('Median is {}; Mean is {}'.format(median, mean))
    return proportion
  
  
def h5py_batch(file, batch_idx, batch_size, num_batch):
    if batch_idx == num_batch - 1:
      lb, ub = batch_idx * batch_size, len(file['input_image']) 
    else:
      lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
    batch_data = file['input_image'][lb: ub]
    return batch_data
  
   
class TextDataset(data.Dataset):
  def __init__(self, data_dir, split='train',
               base_size=64,
               transform=None, target_transform=None): 
      self.data_dir = data_dir
      self.split = split
      self.split_dir = os.path.join(self.data_dir, self.split)
      self.imgpath = '%s/%s/FashionSynthesis_128_128.h5' % (self.data_dir, self.split)
      with h5py.File(self.imgpath, 'r') as file:
        self.number_example = len(file['input_image'])
      self.imgset = self.load_imgset()
      # not needed when files exist
#       self.mat1path = '%s/language_original.mat' % (data_dir)
#       self.mat2path = '%s/ind.mat' % (data_dir)
#       self.mat1 = scipy.io.loadmat(self.mat1path)
#       self.mat2 = scipy.io.loadmat(self.mat2path)
#       self.textset = self.mat1['engJ']
#       self.classset = self.mat1['cate_new']
#       self.train_ids = self.mat2['train_ind']  # 70000 x 1
#       self.test_ids = self.mat2['test_ind']  # 8979 x 1
#       self.train_len = self.train_ids.shape[0]
#       self.test_len = self.test_ids.shape[0]

      self.transform = transform
      self.norm = transforms.Compose([
         transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
      self.target_transform = target_transform
      self.imsize = []
      for i in range(cfg.TREE.BRANCH_NUM):
        self.imsize.append(base_size)
        base_size = base_size * 2
      self.embeddings_num = 1
      
      self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data()
      self.class_id = self.load_class_id()
      
      
  def load_imgset(self):
      file = h5py.File(self.imgpath, 'r')
      batch_size = 256
      num_batch = np.ceil(self.number_example / batch_size)
      imgset = np.zeros((self.number_example, 128, 128, 3))
      for i in range(int(num_batch)): 
        batch_data = h5py_batch(file, i, batch_size, num_batch)
        if i == num_batch - 1:
            lb, ub = i * batch_size, self.number_example
        else:
            lb, ub = i * batch_size, (i + 1) * batch_size
        imgset[lb:ub, :, :, :] = batch_data
        
      file.close()
      return imgset
   
   
  def load_captions(self):
      train_captions = []
      test_captions = []
      for i in range(self.train_len):
          cap = self.textset[self.train_ids[i, :]-1][0][0][0]
          cap = cap.replace("\ufffd\ufffd", " ")
          tokenizer = RegexpTokenizer(r'\w+')
          tokens = tokenizer.tokenize(cap.lower())
          tokens_new = []
          for t in tokens:
              t = t.encode('ascii', 'ignore').decode('ascii')
              if len(t) > 0:
                tokens_new.append(t)
          train_captions.append(tokens_new)
      for i in range(self.test_len):
          cap = self.textset[self.test_ids[i, :]-1][0][0][0]
          cap = cap.replace("\ufffd\ufffd", " ")
          tokenizer = RegexpTokenizer(r'\w+')
          tokens = tokenizer.tokenize(cap.lower())
          tokens_new = []
          for t in tokens:
              t = t.encode('ascii', 'ignore').decode('ascii')
              if len(t) > 0:
                tokens_new.append(t)
          test_captions.append(tokens_new)
      return train_captions, test_captions      
       
     
  def build_dictionary(self, train_captions, test_captions):
      word_counts = defaultdict(float)
      captions = train_captions + test_captions
      for sent in captions:
          for word in sent:
              word_counts[word] += 1
      sorted_word_counts = OrderedDict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
      print('word counts in fashion dataset: \n', sorted_word_counts)
      vocab = [w for w in word_counts if word_counts[w] >= 0]
      print('{} vocabularies in fashion dataset'.format(len(vocab)))
      # len(vocab) = len(ixtoword) - 1
      ixtoword = {}
      ixtoword[0] = '<end>'
      wordtoix = {}
      wordtoix['<end>'] = 0
      ix = 1
      for w in vocab:
          wordtoix[w] = ix
          ixtoword[ix] = w
          ix += 1        
      train_captions_new = []
      for t in train_captions:
          rev = []
          for w in t:
              if w in wordtoix:
                rev.append(wordtoix[w])
          train_captions_new.append(rev)
      test_captions_new = []
      for t in test_captions:
          rev = []
          for w in t:
              if w in wordtoix:
                rev.append(wordtoix[w])
          test_captions_new.append(rev)
      return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]
     
       
  def load_text_data(self):
      filepath = os.path.join(self.data_dir, 'captions.pickle')
      if not os.path.isfile(filepath):
        train_captions, test_captions = self.load_captions()
        train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions)
        with open(filepath, 'wb') as f:
          pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
        print('Save to: {}'.format(filepath))
      else:
        with open(filepath, 'rb') as f:
          x = pickle.load(f)
          train_captions, test_captions = x[0], x[1]
          ixtoword, wordtoix = x[2], x[3]
          del x
          n_words = len(ixtoword)
        print('Load from: {}'.format(filepath))
      if self.split == 'train':
        captions = train_captions
      else:
        captions = test_captions
      return captions, ixtoword, wordtoix, n_words
       
       
  def load_class_id(self):
      filepath = os.path.join(self.split_dir, 'Class_info.pickle')
      if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            class_id = pickle.load(f)
        print('Load from: {}'.format(filepath))
      else:
        trainclass = []
        testclass = []
        if self.split == 'train':
          for i in range(self.train_len):
            trainclass.append(self.classset[self.train_ids[i, :]-1][0])
          with open(filepath, 'wb') as f:
              pickle.dump(trainclass, f, protocol=2)
          print('Save to: {}'.format(filepath))
        else:
          for i in range(self.test_len):
            testclass.append(self.classset[self.test_ids[i, :]-1][0])
          with open(filepath, 'wb') as f:
              pickle.dump(testclass, f, protocol=2)
          print('Save to: {}'.format(filepath))
      return class_id
 

  def get_caption(self, cap_ix):
      caption = np.asarray(self.captions[cap_ix]).astype('int64')
      if (caption == 0).sum() > 0:
         print('ERROR: do not need END (0) token', caption)
      num_words = len(caption)
      # pad with 0s (i.e., '<end>')
      x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
      x_len = num_words
      if num_words <= cfg.TEXT.WORDS_NUM:
         x[:num_words, 0] = caption
      else:
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:cfg.TEXT.WORDS_NUM]
        ix = np.sort(ix)
        x[:, 0] = caption[ix]
        x_len = cfg.TEXT.WORDS_NUM
      return x, x_len
       

  def __getitem__(self, index):  # call -- dataset[index]
      imgs = get_imgs(self.imgset[index, :, :, :].astype(np.uint8), self.imsize, self.transform, self.norm)
      cls_id = self.class_id[index]
      cap, cap_len = self.get_caption(index)
      return imgs, cap, cap_len, cls_id
  
     
  def __len__(self):
      return self.number_example  # call -- len(dataset)
