from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import h5py
import numpy as np
from PIL import Image
import os
import sys
if sys.version_info[0] == 2:
   import cPickle as pickle
else:
   import pickle


def h5py_batch(file, feature, batch_idx, batch_size, num_batch):
    if batch_idx == num_batch - 1:
      lb, ub = batch_idx * batch_size, len(file['input_image']) 
    else:
      lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
    batch_data = file[feature][lb: ub]
    return batch_data   

def print_imgs(data_dir, output_dir, size):
  file = h5py.File(data_dir, 'r')
  batch_size = 256
  num_batch = np.ceil(len(file)/ batch_size)
  for i in range(int(num_batch)):
      batch_data = h5py_batch(file, 'input_image', i, batch_size, num_batch)
      print('Print %d/%d' %(i+1, num_batch))
      for j in range(batch_data.shape[0]):
        img = batch_data[j]
        Img = Image.fromarray(img, 'RGB')
        file_name = '/test128_%d.png' %(j+1+batch_size*i)
        Img.save(output_dir + file_name)
  file.close()
  return None

data_dir = '/local_storage/zhyi/data/FashionSynthesis/test/FashionSynthesis_128_128.h5'
output_dir = '/.local/AttnGAN/data/FashionSynthesis/test/original'
size = (128, 128)
print_imgs(data_dir, output_dir, size)
