# Select a subset from the validation set of FashionGen
# The resulted idx sequence is saved as 'val_idx.npy'

from random import seed
from random import sample
import numpy as np
# seed random number generator
seed(1)

num_batch = 50
batch_size = 48
# prepare a sequence
sequence = [i for i in range(32528)]
print(sequence)
# select a subset without replacement
subset = sample(sequence, num_batch*batch_size)
print(subset)
np.save('val_idx.npy', subset)
