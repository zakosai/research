import tensorflow as tf
import numpy as np
from dae_im import vanilla_vae
import scipy.io as sio
from scipy.sparse import load_npz
import argparse

np.random.seed(0)
tf.set_random_seed(0)
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
args = parser.parse_args()
ckpt = args.ckpt_folder
dir = args.data_dir

images = np.fromfile(dir +"images.bin", dtype=np.uint8)
data = images.reshape((13791, 32, 32, 3))
data = data.astype(np.float32)/255

idx = np.random.rand(data.shape[0]) < 0.8
train_X = data[idx]
test_X = data[~idx]
print(len(train_X), len(train_X[0]))
print(len(test_X), len(test_X[0]))
#


model = vanilla_vae(width=32, height=32, loss='l2', ckpt_folder=ckpt, z_dim=50)
# As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                    # lr=0.01, batch_size=128, print_step=50)
print('fitting data starts...')
model.fit(train_X, epochs=10000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope="image")
