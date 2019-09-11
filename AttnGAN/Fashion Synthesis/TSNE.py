import numpy as np
from sklearn.manifold import TSNE
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

filepath = '/.local/AttnGAN/data/FashionSynthesis/test/Class_info.pickle'
with open(filepath, 'rb') as f:
    labels = pickle.load(f)
print('Load from: ', filepath)

X = np.load('/.local/AttnGAN/data/FashionSynthesis/test/embeddings/final.npy')
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = TSNE(n_components=2, perplexity=5.0, early_exaggeration=8.0, 
         learning_rate=10.0, n_iter=4000, n_iter_without_progress=300, 
         min_grad_norm=1e-07, metric='euclidean', init='pca', 
         verbose=1, random_state=None, method='barnes_hut').fit_transform(X)

categories = np.asarray(labels)
colormap = np.array(['#0048BA',
                     '#7CB9E8',
                     '#C46210',
                     '#9F2B68',
                     '#F19CBB',
                     '#D3212D',
                     '#3B7A57',
                     '#FF7E00',
                     '#9966CC',
                     '#665D1E',
                     '#8DB600',
                     '#00FFFF',
                     '#4B5320',
                     '#8F9779',
                     '#FDEE00',
                     '#EEDC82',
                     '#534B4F',
                     '#7B3F00',
                     '#000000'])
scatterplot = []
names = ['Blazer',
          'Blouse',
          'Bomber',
          'Cardigan',
          'Henley',
          'Hoodie',
          'Jacket',
          'Jersey',
          'Parka',               
          'Poncho',           
          'Sweater',
          'Tank', 
          'Tee',
          'Top',
          'Coat',
          'Dress',
          'Jumpsuit',
          'Kimono',       
          'Romper']

existingidx = np.unique(categories)
existingnames = []
for i in existingidx:
  existingnames.append(names[int(i)-1])
for i in range(1, 20):
    idx = np.argwhere(categories == i)
    if len(idx)==0:
      continue
    else:
      scatterplot.append(plt.scatter(Y[idx, 0], Y[idx, 1], 2, colormap[i-1], marker='.'))    
plt.figure(figsize=(512, 450))
plt.legend(scatterplot,
           existingnames,
           bbox_to_anchor=(0.5,0),
           loc='upper center',
           fontsize=4,
           ncol=8)
plt.axis('off')
plt.savefig('/.local/AttnGAN/data/FashionSynthesis/test/embeddings/tsne_p5_early8_lr10_epoch4000_barnes.png', bbox_inches='tight')  
plt.close()
print('Save the image successfully.')
