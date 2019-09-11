from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import random
import numpy as np
import pandas as pd
import h5py

from miscc.config import cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='lstm',
                 imsize=64, transform=None, target_transform=None):
        self.filepath = '%s/%s/FashionSynthesis_128_128.h5' % (data_dir, split)
        self.path2 = '/.local/AttnGAN/data/FashionSynthesis'
        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.imgfile = None
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.embeddings = self.load_embedding(split, embedding_type)
        self.class_id = self.load_class_id()
        with h5py.File(self.filepath, 'r') as file:
          self.number_example = len(file['input_image'])

    def get_img(self, img):
        img = Image.fromarray(img, 'RGB')
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img    

    def load_class_id(self):
        filepath = os.path.join(self.split_dir, 'Class_info.pickle')
        if os.path.isfile(filepath):
          with open(filepath, 'rb') as f:
              class_id = pickle.load(f)
          print('Load from: ', filepath)
        else:
          raise NotImplementedError
        return class_id
      
    def load_embedding(self, split, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/embeddings/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'lstm':
            embedding_filename = '/embeddings/final.npy'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/embeddings/skip-thought-embeddings.pickle'
        path = self.path2 + '/{}'.format(split)
        embeddings = np.load(path + embedding_filename)
        print('embeddings: ', embeddings.shape)
        return embeddings

    def __getitem__(self, index):
        self.imgfile = h5py.File(self.filepath, 'r')['input_image']
        img = self.get_img(self.imgfile[index])
        embedding = self.embeddings[index, :]
        if self.target_transform is not None:
          embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return self.number_example
