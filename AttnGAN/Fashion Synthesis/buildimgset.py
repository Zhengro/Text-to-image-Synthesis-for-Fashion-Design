import h5py
import numpy as np
import scipy.io
import pickle


def h5py_batch(file, features, batch_idx, batch_size, num_batch):
    if batch_idx == num_batch - 1:
        lb, ub = batch_idx * batch_size, len(file['ih'])
    else:
        lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
    imgs = file[features[0]][lb: ub]
    img_mean = file[features[1]]
    new_mean = np.tile(img_mean, (int(ub-lb), 1, 1, 1))
    batch_data = imgs + new_mean
    batch_data = np.moveaxis(batch_data, [0, 1, 2, 3], [0, -1, -2, -3])
    for i in range(ub-lb):
        batch_data[i, :, :, :] -= batch_data[i, :, :, :].min()
        batch_data[i, :, :, :] /= batch_data[i, :, :, :].max()
        batch_data[i, :, :, :] = batch_data[i, :, :, :]*255
    return batch_data.astype(np.uint8)

file = h5py.File('/.local/AttnGAN/data/FashionSynthesis/G2.h5', mode='r')
list_of_features = ['ih', 'ih_mean']
batch_size = 256
dataset_len = len(file['ih'])  # 78979
num_batch = np.ceil(dataset_len / batch_size)

imgfile = h5py.File('/.local/AttnGAN/data/FashionSynthesis/imgset.h5', 'a')
imgset = imgfile.create_dataset('images', (dataset_len, 128, 128, 3), dtype='uint8', compression='gzip')

for i in range(int(num_batch)): 
    batch_data = h5py_batch(file, list_of_features, i, batch_size, num_batch)
    if i == num_batch - 1:
        lb, ub = i * batch_size, len(file['ih'])
    else:
        lb, ub = i * batch_size, (i + 1) * batch_size
    imgset[lb:ub, :, :, :] = batch_data
    
file.close()
imgfile.close()
