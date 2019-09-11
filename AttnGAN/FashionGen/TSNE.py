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

filepath = '/.local/AttnGAN/data/FashionGen/test/Class_info.pickle'
with open(filepath, 'rb') as f:
    labels = pickle.load(f)
print('Load from: ', filepath)

X = np.load('/.local/AttnGAN/data/FashionGen/test/embeddings/final.npy')
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, 
         learning_rate=200.0, n_iter=20000, n_iter_without_progress=300, 
         min_grad_norm=1e-07, metric='euclidean', init='pca', 
         verbose=1, random_state=None, method='barnes_hut').fit_transform(X)

categories = np.asarray(labels)
colormap = np.array(['#ff0000',
                     '#ff5400',
                     '#ff7200',
                     '#ffa100',
                     '#ffd000',
                     '#fff600',
                     '#e5ff00',
                     '#b2ff00',
                     '#72ff00',
                     '#00ff6a',
                     '#00ffa5',
                     '#00ffe9',
                     '#00f2ff',
                     '#00ddff',
                     '#00c3ff',
                     '#00a1ff',
                     '#006eff',
                     '#0048ff',
                     '#6a00ff',
                     '#9000ff',
                     '#bf00ff',
                     '#e900ff',
                     '#ff00ee',
                     '#ff00d0',
                     '#ff00a5',
                     '#ff007b',
                     '#ff0054',
                     '#96312a',
                     '#964f2a',
                     '#96692a',
                     '#96852a',
                     '#ff5656',
                     '#fc7b7b',
                     '#af5454',
                     '#fcb480',
                     '#fcdd88',
                     '#f4fc88',
                     '#165431',
                     '#00893b',
                     '#66a581',
                     '#829b8d',
                     '#ace6f9',
                     '#0a5872',
                     '#6776b2',
                     '#30285b',
                     '#f496ff',
                     '#e8c2dc',
                     '#000000'])
scatterplot = []
names = ['POCKET SQUARES & TIE BARS', 'FINE JEWELRY',
         'JACKETS & COATS', 'HATS',
         'TOPS', 'SOCKS',
         'SHOULDER BAGS', 'LOAFERS',
         'SHIRTS', 'TIES',
         'BRIEFCASES', 'BELTS & SUSPENDERS',
         'TOTE BAGS', 'TRAVEL BAGS',
         'DUFFLE & TOP HANDLE BAGS', 'BAG ACCESSORIES',
         'KEYCHAINS', 'DUFFLE BAGS',
         'SNEAKERS', 'PANTS',
         'SWEATERS', 'JEWELRY',
         'SHORTS', 'ESPADRILLES',
         'MESSENGER BAGS', 'EYEWEAR',
         'HEELS', 'MONKSTRAPS',
         'MESSENGER BAGS & SATCHELS', 'FLATS',
         'BLANKETS', 'POUCHES & DOCUMENT HOLDERS',
         'DRESSES', 'JUMPSUITS',
         'UNDERWEAR & LOUNGEWEAR', 'BOAT SHOES & MOCCASINS',
         'CLUTCHES & POUCHES', 'JEANS',
         'SWIMWEAR', 'SUITS & BLAZERS',
         'LINGERIE', 'GLOVES',
         'BOOTS', 'LACE UPS',
         'SCARVES', 'SANDALS',
         'BACKPACKS', 'SKIRTS']

existingidx = np.unique(categories)
existingnames = []
for i in existingidx:
  existingnames.append(names[int(i)-1])
for i in range(1, 49):
    idx = np.argwhere(categories == i)
    if len(idx)==0:
      continue
    else:
      scatterplot.append(plt.scatter(Y[idx, 0], Y[idx, 1], 2, colormap[i-1], marker='.'))
plt.legend(scatterplot,
           existingnames,
           bbox_to_anchor=(0.5,0),
           loc='upper center',
           fontsize=4,
           ncol=8)
plt.axis('off')
plt.savefig('/.local/AttnGAN/data/FashionGen/test/embeddings/tsne_p30_early12_lr200_epoch20000_barnes.png', bbox_inches='tight')  
plt.close()
print('Save the image successfully.')
