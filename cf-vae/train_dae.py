import tensorflow as tf
import numpy as np
from dae import vanilla_vae
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

# variables = sio.loadmat("data/citeulike-a/mult_nor.mat")
# data = variables['X']
variables = load_npz(dir + "/mult-nor.npz")
data = variables.toarray()
idx = np.random.rand(data.shape[0]) < 0.8
train_X = data[idx]
test_X = data[~idx]

#
# images = np.fromfile("data/amazon/images.bin")
# images = images.reshape((16000, 3072))
# train_img = images[idx]
# test_img = images[~idx]

model = vanilla_vae(input_dim=8000, encoding_dims=[200, 100], z_dim=50, decoding_dims=[100, 200, 8000], loss='cross_entropy', ckpt_folder=ckpt)
# As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                    # lr=0.01, batch_size=128, print_step=50)
print('fitting data starts...')
model.fit(train_X, epochs=10000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope="text")
