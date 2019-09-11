import h5py
import numpy as np
import scipy.io
import pickle


def h5py_batch(file, feature, batch_idx, batch_size, num_batch):
    if batch_idx == num_batch - 1:
      lb, ub = batch_idx * batch_size, len(file['images']) 
    else:
      lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
    batch_data = file[feature][lb: ub]
    return batch_data
  
  
imgfile = h5py.File('/.local/AttnGAN/data/FashionSynthesis/imgset.h5', 'r')
batch_size = 256
num_batch = np.ceil(len(imgfile['images'])/ batch_size)
dataset_len = len(imgfile['images'])  # 78979
imgset = np.zeros((dataset_len, 128, 128, 3))
for i in range(int(num_batch)):
  batch_data = h5py_batch(imgfile, 'images', i, batch_size, num_batch)
  if i == num_batch - 1:
      lb, ub = i * batch_size, dataset_len
  else:
      lb, ub = i * batch_size, (i + 1) * batch_size
  imgset[lb:ub, :, :, :] = batch_data  
imgfile.close()

mat1path = '/.local/AttnGAN/data/FashionSynthesis/language_original.mat'
mat2path = '/.local/AttnGAN/data/FashionSynthesis/ind.mat'
mat1 = scipy.io.loadmat(mat1path)
mat2 = scipy.io.loadmat(mat2path)
textset = mat1['engJ']
classset = mat1['cate_new']
train_ids = mat2['train_ind']  # 70000 x 1
test_ids = mat2['test_ind']  # 8979 x 1
train_len = train_ids.shape[0]
test_len = test_ids.shape[0]

trainfile = h5py.File('/.local/AttnGAN/data/FashionSynthesis/train/FashionSynthesis_128_128.h5', 'a')
trainset = trainfile.create_dataset('input_image', (train_len, 128, 128, 3), dtype='uint8', compression='gzip')
idx = []
for i in range(train_len): 
  idx.append(train_ids[i, :][0]-1)
num_batch = np.ceil(train_len / batch_size)
for i in range(int(num_batch)):
  if i == num_batch - 1:
      lb, ub = i * batch_size, train_len
  else:
      lb, ub = i * batch_size, (i + 1) * batch_size
  trainset[lb:ub, :, :, :] = imgset[idx[lb:ub], :, :, :]
  
trainfile.close()
  
testfile = h5py.File('/.local/AttnGAN/data/FashionSynthesis/test/FashionSynthesis_128_128.h5', 'a')
testset = testfile.create_dataset('input_image', (test_len, 128, 128, 3), dtype='uint8', compression='gzip')
idx = []
for i in range(test_len): 
  idx.append(test_ids[i, :][0]-1)
num_batch = np.ceil(test_len / batch_size)
for i in range(int(num_batch)):
  if i == num_batch - 1:
      lb, ub = i * batch_size, test_len
  else:
      lb, ub = i * batch_size, (i + 1) * batch_size
  testset[lb:ub, :, :, :] = imgset[idx[lb:ub], :, :, :]
  
testfile.close()
