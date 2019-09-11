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
#         newimg = Img.resize(size, Image.ANTIALIAS)
#         file_name = '/test128_%d.png' %(j+1+batch_size*i)
        file_name = '/test256_%d.png' %(j+1+batch_size*i)
#         newimg.save(output_dir + file_name)
        Img.save(output_dir + file_name)
  file.close()
  return None

def print_text(data_dir, output_dir):
  file = h5py.File(data_dir, 'r')
  batch_size = 256
  num_batch = np.ceil(len(file)/ batch_size)
  f = open(output_dir + '/test_input_desc.txt', 'a')
  for i in range(int(num_batch)):
      batch_data = h5py_batch(file, 'input_description', i, batch_size, num_batch)
      print('Print %d/%d' %(i+1, num_batch))
      for j in range(batch_data.shape[0]):
        desc = batch_data[j, 0].decode('UTF-8', errors="ignore")
        pre = '%d. ' %(j+1+batch_size*i)
        desc = ''.join([pre, desc])
        f.write(desc.encode('utf8') + '\n')
  f.close()
  file.close()
  return None

data_dir = '/Fashiongen_256_256.h5'
output_dir = '/.local/AttnGAN/data/fashion/test/original'
# size = (128, 128)
size = (256, 256)
print_imgs(data_dir, output_dir, size)
# output_dir = '/.local/AttnGAN/data/fashion/test'
# print_text(data_dir, output_dir)
