import sys
import numpy as np
if sys.version_info[0] == 2:
   import cPickle as pickle
else:
   import pickle
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *


def distribution_nw(captions):
  set_len = len(captions)
  nw = []
  for i in range(set_len):
    nw.append(len(captions[i]))
  nwdic = {}
  for i in range(min(nw), max(nw)+1):
    key = i
    nwdic['{}'.format(key)] = 0
  for i in range(set_len):
    nwdic['{}'.format(nw[i])] += 1
  keys = range(min(nw), max(nw)+1)
  values = []
  for i in range(min(nw), max(nw)+1):
    values.append(nwdic['{}'.format(i)])
  return keys, values

filepath = '/.local/AttnGAN/data/FashionSynthesis/captions.pickle'
# filepath = '/.local/AttnGAN/data/FashionGen/captions.pickle'
with open(filepath, 'rb') as f:
  x = pickle.load(f)
  train_captions, test_captions = x[0], x[1]
  del x
keys, values = distribution_nw(train_captions)
ks, vs = distribution_nw(test_captions)
final_keys = list(set(keys + ks)) 

new_values = []
for k in final_keys:
  if k not in keys:
    new_values.append(0)
  else:
    idk = keys.index(k)
    new_values.append(values[idk])
new_vs = []
for k in final_keys:
  if k not in ks:
    new_vs.append(0)
  else:
    idk = ks.index(k)
    new_vs.append(vs[idk])

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(4, 2.5))

height_train = new_values
height_val = new_vs
bars = final_keys

x_pos = np.arange(len(bars))
ax.bar(x_pos, height_train, 0.5, align='center', color='lightblue')
ax.bar(x_pos+0.5, height_val, 0.5, align='center', color='lavender')
ax.set_xticks(x_pos+0.1)
ax.set_xticklabels(bars)
plt.xlim([-1, len(bars)])
ax.set_ylabel('Number of samples')
ax.grid(True, axis='y', c='silver', linestyle='--', linewidth=0.58)
minorticks_off()
tick_params(top=False, bottom=False, left=False, right=False)

split = ['train', 'validation']
plt.legend(split, loc=1, prop={'size': 7})
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() ):
    item.set_fontsize(7)
plt.savefig('/.local/AttnGAN/distribution_nw2.png', bbox_inches = 'tight')
# plt.savefig('/.local/AttnGAN/distribution_nw1.png', bbox_inches = 'tight')
