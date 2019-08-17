import h5py
import numpy as np
from collections import defaultdict, OrderedDict
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *


def h5py_batch(file, feature, batch_idx, batch_size, num_batch):
    if batch_idx == num_batch - 1:
      lb, ub = batch_idx * batch_size, len(file['input_image']) 
    else:
      lb, ub = batch_idx * batch_size, (batch_idx + 1) * batch_size
    batch_data = file[feature][lb: ub]
    return batch_data

def load_subcas(dir):
    file = h5py.File(dir, 'r')
    all_subcas = []
    batch_size = 256
    num_batch = np.ceil(len(file['input_image'])/ batch_size)
    for i in range(int(num_batch)):
        batch_data = h5py_batch(file, 'input_category', i, batch_size, num_batch)
        for j in range(batch_data.shape[0]):
            sub = batch_data[j, 0].decode('latin-1')
            all_subcas.append(sub)
    file.close()
    print('{} data'.format(len(all_subcas)))
    return all_subcas

def build_dictionary(subcas):
    subca_counts = defaultdict(float)
    for subca in subcas:
        subca_counts[subca] += 1
    sorted_subca_counts = OrderedDict(sorted(subca_counts.items(), key=lambda x: x[1], reverse=True))
    print('subcategory counts in fashion dataset: \n{}'.format(sorted_subca_counts))
    sub = [s for s in subca_counts if subca_counts[s] >= 0]
    print('{} subcategories in fashion dataset'.format(len(sub)))
    return sorted_subca_counts, sub

dir = '/Datasets/FashionGen/train/Fashiongen_256_256.h5'
train_subcas = load_subcas(dir)
sorted_subca_counts, sub = build_dictionary(train_subcas)
keys = []
values = []
for k, v in sorted_subca_counts.items():
  keys.append(k)
  values.append(v)

dir = '/Datasets/FashionGen/test/Fashiongen_256_256.h5'
test_subcas = load_subcas(dir)
sorted_subca_counts, sub = build_dictionary(test_subcas)
ks = []
vs = []
for k, v in sorted_subca_counts.items():
  ks.append(k)
  vs.append(v)

new_vs = []
for k in keys:
  if k not in ks:
    new_vs.append(0)
  else:
    idk = ks.index(k)
    new_vs.append(vs[idk])
 
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(5, 8))

height_train = values
height_val = new_vs
bars = keys
y_pos = np.arange(len(bars))

ax.barh(y_pos, height_train, 0.5, left=0, align='center', color='lightblue')
ax.barh(y_pos+0.5, height_val, 0.5, align='center', color='lavender')
ax.set_yticks(y_pos+0.1)
ax.set_yticklabels(bars)
ax.set_xscale('log')
plt.ylim([-1, len(bars)])
ax.invert_yaxis()
ax.set_xlabel('Number of samples')
ax.grid(True, axis='x', c='silver', linestyle='--', linewidth=0.58)
minorticks_off()
tick_params(top=False, bottom=False, left=False, right=False)

split = ['train','validation']
plt.legend(split, loc=4, prop={'size': 7})

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() ):
    item.set_fontsize(7)
plt.savefig('distribution_cate1.png', bbox_inches = 'tight')
