from model import RNN_ENCODER
import torch
import numpy as np

ntokens = 6872+1
nhidden = 256
TRAIN_NET_E = '/.local/AttnGAN/output/FashionGen_damsm_final_2019_06_21_20_53_23/Model/text_encoder130.pth'
text_encoder = RNN_ENCODER(ntokens, nhidden=nhidden)
state_dict = torch.load(TRAIN_NET_E, map_location=lambda storage, loc: storage) 
text_encoder.load_state_dict(state_dict)
weights = text_encoder.encoder.weight
np.save('/.local/AttnGAN/data/FashionGen/word embeddings/wordsfinal.npy', weights.detach().numpy())
