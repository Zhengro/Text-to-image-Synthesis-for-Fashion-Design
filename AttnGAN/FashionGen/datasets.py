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
import numpy.random as random
if sys.version_info[0] == 2:
   import cPickle as pickle
else:
   import pickle

################################## New Version #################################
# 1. prepare_data(data) returns no keys; 
# 2. class TextDataset(data.Dataset): self.data = []; self.embeddings_num = 1; self.bbox = None
# 3. batch_size = 256 when retrieve a batch in h5.file
# 4. class_id refers to the categories instead of subcategories
# 5. captions_statistics calculated median, mean and proportion
################################################################################
   
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


def get_imgs(img, imsize, bbox=None, transform=None, normalize=None):
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
 
   
class TextDataset(data.Dataset):
  def __init__(self, data_dir, split='train',
               base_size=64,
               transform=None, target_transform=None): 
      self.filepath = '%s/%s/Fashiongen_256_256.h5' % (data_dir, split)
      self.imgfile = None
      self.textfile = None
      self.classfile = None
      self.data = []
      self.data_dir = data_dir
      self.split = split
      self.split_dir = os.path.join(data_dir, split)
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
      self.bbox = None        
      self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data()
      self.class_id = self.load_class_id()
      with h5py.File(self.filepath, 'r') as file:
        self.number_example = len(file['input_image'])

         
  def h5py_batch(self, file, feature, batch_idx, batch_size, num_batch):
      if batch_idx == num_batch - 1:
        lb, ub = batch_idx * batch_size, len(file['input_image']) 
      else:
        lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
      batch_data = file[feature][lb: ub]
      return batch_data

       
  def load_captions(self, dir):
      file = h5py.File(dir, 'r')
      all_captions = []
      batch_size = 256
      num_batch = np.ceil(len(file['input_image'])/ batch_size)
      for i in range(int(num_batch)):
          batch_data = self.h5py_batch(file, 'input_description', i, batch_size, num_batch)
          for j in range(batch_data.shape[0]):
              cap = batch_data[j, 0].decode('latin-1')
              cap = cap.replace("\ufffd\ufffd", " ")
              tokenizer = RegexpTokenizer(r'\w+')
              tokens = tokenizer.tokenize(cap.lower())
              tokens_new = []
              for t in tokens:
                  t = t.encode('ascii', 'ignore').decode('ascii')
                  if len(t) > 0:
                    tokens_new.append(t)
              all_captions.append(tokens_new)
      file.close()
      return all_captions      
       
       
  def load_subcas(self, dir):
      file = h5py.File(dir, 'r')
      all_subcas = []
      batch_size = 256
      num_batch = np.ceil(len(file['input_image'])/ batch_size)
      list_of_subs = [u'TOPS', u'SWEATERS', u'JACKETS & COATS', u'PANTS', u'JEANS', u'SHORTS', u'SHIRTS', u'DRESSES', u'SKIRTS',  u'SUITS & BLAZERS']
      for i in range(int(num_batch)):
          batch_data = self.h5py_batch(file, 'input_category', i, batch_size, num_batch)
          for j in range(batch_data.shape[0]):
              sub = batch_data[j, 0].decode('latin-1')
              all_subcas.append(sub)
#               if sub in list_of_subs:
#                 all_subcas.append(sub)
      file.close()
      print('{}data'.format(len(all_subcas)))
      return all_subcas
     
     
  def build_dictionary(self, train_captions, test_captions):
      word_counts = defaultdict(float)
      captions = train_captions + test_captions
      for sent in captions:
          for word in sent:
              word_counts[word] += 1
      sorted_word_counts = OrderedDict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True))
      print('word counts in fashion dataset: \n', sorted_word_counts)
      vocab = [w for w in word_counts if word_counts[w] >= 0]
      print('{} vocabularies in fashion dataset'.format(len(vocab)))  # len(vocab) = len(ixtoword) - 1
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
     
     
  def build_dictionary2(self, train_subcas, test_subcas):
      subca_counts = defaultdict(float)
      subcas = train_subcas + test_subcas
      for subca in subcas:
          subca_counts[subca] += 1
      sorted_subca_counts = OrderedDict(sorted(subca_counts.items(), key=lambda x: x[1], reverse=True))
      print('subcategory counts in fashion dataset: \n', sorted_subca_counts)
      sub = [s for s in subca_counts if subca_counts[s] >= 0]
      print('{} subcategories in fashion dataset'.format(len(sub)))
      ixtosub = {}
      subtoix = {}
      ix = 1
      for s in sub:
          subtoix[s] = ix
          ixtosub[ix] = s
          ix += 1        
      train_subcas_new = []
      for s in train_subcas:
         if s in subtoix:
            train_subcas_new.append(subtoix[s])
      test_subcas_new = []
      for s in test_subcas:
         if s in subtoix:
            test_subcas_new.append(subtoix[s])
      return [train_subcas_new, test_subcas_new]
       
     
  def load_text_data(self):
      filepath = os.path.join(self.data_dir, 'captions.pickle')
      if not os.path.isfile(filepath):
        train_dir = os.path.join(self.data_dir, 'train/Fashiongen_256_256.h5')
        test_dir = os.path.join(self.data_dir, 'test/Fashiongen_256_256.h5')
        train_captions = self.load_captions(train_dir)
        test_captions = self.load_captions(test_dir)
        train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions)
        with open(filepath, 'wb') as f:
          pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
        print('Save to: ', filepath)
      else:
        with open(filepath, 'rb') as f:
          x = pickle.load(f)
          train_captions, test_captions = x[0], x[1]
          ixtoword, wordtoix = x[2], x[3]
          del x
          n_words = len(ixtoword)
        print('Load from: ', filepath)
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
        print('Load from: ', filepath)
      else:
        train_dir = os.path.join(self.data_dir, 'train/Fashiongen_256_256.h5')
        test_dir = os.path.join(self.data_dir, 'test/Fashiongen_256_256.h5')
        train_subcas = self.load_subcas(train_dir)
        test_subcas = self.load_subcas(test_dir)
        train_subcas, test_subcas = self.build_dictionary2(train_subcas, test_subcas)  
        if self.split == 'train':
          class_id = train_subcas
          with open(filepath, 'wb') as f:
              pickle.dump(train_subcas, f, protocol=2)
          print('Save to: ', filepath)
        else:
          class_id = test_subcas
          with open(filepath, 'wb') as f:
              pickle.dump(test_subcas, f, protocol=2)
          print('Save to: ', filepath)
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
      if self.imgfile is None and self.textfile is None and self.classfile is None:
        self.imgfile = h5py.File(self.filepath, 'r')['input_image']
#         self.textfile = h5py.File(self.filepath, 'r')['input_description']
#         self.classfile = h5py.File(self.filepath, 'r')['input_category']
      cls_id = self.class_id[index]
      imgs = get_imgs(self.imgfile[index], self.imsize, self.bbox, self.transform, self.norm)
      cap, cap_len = self.get_caption(index)
      return imgs, cap, cap_len, cls_id
  
     
  def __len__(self):
      return self.number_example  # call -- len(dataset)
