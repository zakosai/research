import tensorflow as tf
from tensorbayes.layers import dense, placeholder
import os
from keras.backend import binary_crossentropy
from keras.layers import merge
import numpy as np
import time
from scipy.sparse import load_npz
import argparse

class vanilla_vae:
    """
    build a vanilla vae
    you can customize the activation functions pf each layer yourself.
    """

    def __init__(self, input_dim, encoding_dims, z_dim, decoding_dims, loss="cross_entropy", useTranse = False, eps = 1e-10, ckpt_folder="pre_model"):
        # useTranse: if we use trasposed weigths of inference nets
        # eps for numerical stability
        # structural info
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoding_dims = encoding_dims
        self.decoding_dims = decoding_dims
        self.loss = loss
        self.useTranse = useTranse
        self.eps = eps
        self.weights = []    # better in np form. first run, then append in
        self.bias = []
        self.ckpt = ckpt_folder



    def fit(self, x_input, epochs = 1000, learning_rate = 0.001, batch_size = 100, print_size = 50, train=True, scope="text"):
        # training setting
        self.DO_SHARE = False
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_size = print_size

        self.g = tf.Graph()
        # inference process
        ########TEXT###################
        with tf.variable_scope(scope):
            print(self.encoding_dims, self.decoding_dims)
            x_ = placeholder((None, self.input_dim))
            x = x_
            depth_inf = len(self.encoding_dims)

            # noisy_level = 1
            # x = x + noisy_level*tf.random_normal(tf.shape(x))
            for i in range(depth_inf):
                x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)

            h_encode = x
            z_mu = dense(h_encode, self.z_dim, scope="mu_layer")
            z_log_sigma_sq = dense(h_encode, self.z_dim, scope="sigma_layer")
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

            # generative process
            if self.useTranse == False:
                depth_gen = len(self.decoding_dims)
                y = z
                for i in range(depth_gen):
                    y = dense(y, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)


            x_recons = y

        if self.loss == "cross_entropy":
            loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x_, x_recons), axis=1))
        elif self.loss == "l2":
            loss_recons = tf.reduce_mean(tf.nn.l2_loss(x_- x_recons))
        loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
        # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
        loss = loss_recons + loss_kl
        # other cases not finished yet
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope))
        ckpt_file = os.path.join(self.ckpt,"vae_%s.ckpt" %scope)
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            start = time.time()
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                _, l, lr, lk = sess.run((train_op, loss, loss_recons, loss_kl), feed_dict={x_:x_batch})
                if i % self.print_size == 0:
                    print("epoches: %d\t loss: %f\t loss recons: %f\t loss kl: %f\t time: %d s"%(i, l,lr, lk, time.time()-start))

            saver.save(sess, ckpt_file)
        else:
            saver.restore(sess, ckpt_file)

        z_mu = sess.run(z_mu, feed_dict={x_:x_input})
        print(z_mu.shape)
        np.save(os.path.join(self.ckpt, scope), z_mu)

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--zdim',  type=int, default=50,
                   help='where model is stored')
parser.add_argument('--data_type',  type=str, default='5',
                   help='where model is stored')
parser.add_argument('--user_dim',  type=int, default=9975,
                   help='where model is stored')
parser.add_argument('--type',  type=str, default="text",
                   help='where model is stored')
args = parser.parse_args()
ckpt = args.ckpt_folder
dir = args.data_dir
zdim = args.zdim
data_type = args.data_type

np.random.seed(0)
tf.set_random_seed(0)

# variables = sio.loadmat("data/citeulike-a/mult_nor.mat")
# data = variables['X']
if args.type == "text":
    variables = load_npz(os.path.join(dir, "item.npz"))
    data = variables.toarray()
else:
    user_info = list(open(os.path.join(dir, "user_info_train.txt"))) + \
                list(open(os.path.join(dir, "user_info_test.txt")))
    user_info = [u.strip() for u in user_info]
    user_info = [u.split(",") for u in user_info]
    user_info = [u[1:] for u in user_info]
    data = np.array(user_info).astype(np.float32)
    # col = [0] + list(range(6, data.shape[1] - 1))
    # data = data[:, col]
# data = np.delete(data, [7,8,9,10,11], axis=1)

idx = np.random.rand(data.shape[0]) < 0.8
train_X = data
test_X = data[~idx]
# print(train_X[0])
#
# images = np.fromfile("data/amazon/images.bin")
# images = images.reshape((16000, 3072))
# train_img = images[idx]
# test_img = images[~idx]

# model = vanilla_vae(input_dim=args.user_dim, encoding_dims=[100], z_dim=zdim, decoding_dims=[100, args.user_dim], loss='cross_entropy', ckpt_folder=ckpt)
dim = train_X.shape[1]
print(dim)
if args.type =="text":
    model = vanilla_vae(input_dim=dim, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200, 400, dim],loss='cross_entropy', ckpt_folder=ckpt)
else:
    model = vanilla_vae(input_dim=dim, encoding_dims=[100], z_dim=zdim, decoding_dims=[100, dim], loss='cross_entropy', ckpt_folder=ckpt)
# As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                # lr=0.01, batch_size=128, print_step=50)
print('fitting data starts...')
model.fit(train_X, epochs=5000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope=args.type)
