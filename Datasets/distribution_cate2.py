import scipy.io
import numpy as np
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pylab import *

def fashionsynthesis_distribution_cate(set_len, ids, nameset):
  keys = []
  for i in range(set_len):
    idx = ids[i][0]-1
    i1 = nameset[idx][0][0].find('/')
    i2 = nameset[idx][0][0].rfind('/')
    phrase = nameset[idx][0][0][i1+1:i2]
    i3 = phrase.rfind('_')
    key = phrase[i3+1:]
    if key not in keys:
      keys.append(key)

  classdic = {}
  for i in range(len(keys)):
    key = keys[i]
    classdic['{}'.format(key)] = 0
  for i in range(set_len):
    idx = ids[i][0]-1
    i1 = nameset[idx][0][0].find('/')
    i2 = nameset[idx][0][0].rfind('/')
    phrase = nameset[idx][0][0][i1+1:i2]
    i3 = phrase.rfind('_')
    key = phrase[i3+1:]
    classdic['{}'.format(key)] += 1

  values = []
  for i in range(len(keys)):
    v = classdic['{}'.format(keys[i])]
    values.append(v)
  order = sorted(range(len(values)), key=lambda k: values[k])

  ks = []
  vs = []
  for i in reversed(order):
    ks.append(keys[i])
    vs.append(values[i])
  return ks, vs

mat1 = scipy.io.loadmat('/Datasets/FashionSynthesis/language_original.mat')
mat2 = scipy.io.loadmat('/Datasets/FashionSynthesis/ind.mat')
nameset = mat1['nameList']
train_ids = mat2['train_ind']  # 70000 x 1
test_ids = mat2['test_ind']  # 8979 x 1
train_len = train_ids.shape[0]
test_len = test_ids.shape[0]

train_ks, train_vs = fashionsynthesis_distribution_cate(train_len, train_ids, nameset)
test_ks, test_vs = fashionsynthesis_distribution_cate(test_len, test_ids, nameset)

new_vs = []
for k in train_ks:
  if k not in test_ks:
    new_vs.append(0)
  else:
    idk = test_ks.index(k)
    new_vs.append(test_vs[idk])
    
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(2, 3.2))

height_train = train_vs
height_val = new_vs
bars = train_ks
y_pos = np.arange(len(bars))

ax.barh(y_pos, height_train, 0.5, left=0, align='center', color='lightblue')
ax.barh(y_pos+0.5, height_val, 0.5, align='center', color='lavender')
ax.set_yticks(y_pos+0.1)
ax.set_yticklabels(bars)
ax.set_xscale('log')
plt.ylim([-0.5, len(bars)])
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
plt.savefig('distribution_cate2.png', bbox_inches = 'tight')
