__author__ = 'linh'
import tensorflow as tf
from tensorbayes.layers import dense, placeholder
from tensorbayes.utils import progbar
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits
from keras.backend import binary_crossentropy
import numpy as np
import time
import os
from scipy.sparse import load_npz

class vanilla_vae:
    """
    build a vanilla vae
    you can customize the activation functions pf each layer yourself.
    """

    def __init__(self, input_dim, encoding_dims, z_dim, decoding_dims, loss="cross_entropy", useTranse = False, eps = 1e-10, ckpt_folder="pre3/dae"):
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
            dropout_rate = 0.3
            x_ = placeholder((None, self.input_dim))
            x = x_
            x = tf.layers.dropout(x, dropout_rate, training=train )
            depth_inf = len(self.encoding_dims)
            for i in range(depth_inf):
                x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
            h_encode = x
            z = dense(h_encode, self.z_dim, scope="mu_layer", activation=tf.nn.sigmoid)


            # generative process
            if self.useTranse == False:
                depth_gen = len(self.decoding_dims)
                y = z
                for i in range(depth_gen):
                    y = dense(y, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # if last_layer_nonelinear: depth_gen -1

            else:
                depth_gen = depth_inf
                ## haven't finnished yet...

            x_recons = y
        sparsity_weight = 0.2
        sparsity_target = 0.1
        def kl_divergence(p,q):
            return  p*tf.log(p/q) + (1-p)*tf.log((1-p)/(1-q))

        if self.loss == "cross_entropy":
            loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(x_, x_recons), axis=1))
        elif self.loss == "l2":
            loss_recons = tf.reduce_mean(tf.nn.l2_loss(x_- x_recons))
        # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
        # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
        z_mean = tf.reduce_mean(z, axis=0)
        sparsity_loss = tf.reduce_sum(kl_divergence(z_mean, sparsity_target))
        # loss = loss_recons + sparsity_weight * sparsity_loss
        loss = loss_recons
        # other cases not finished yet
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope))
        ckpt_file = os.path.join(self.ckpt,"dae_%s.ckpt" %scope)
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            start = time.time()
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                _, l = sess.run((train_op, loss), feed_dict={x_:x_batch})
                if i % self.print_size == 0:
                    print("epoches: %d\t loss: %f\t time: %d s"%(i, l, time.time()-start))

            saver.save(sess, ckpt_file)
        else:
            saver.restore(sess, ckpt_file)

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
    variables = load_npz(os.path.join(dir, "mult_nor.npz"))
    data = variables.toarray()
else:
    data = np.load(os.path.join(dir, "user_info_%s.npy"%data_type))
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
model = vanilla_vae(input_dim=dim, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200, 400, dim],
                    loss='cross_entropy', ckpt_folder=ckpt)
# As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                # lr=0.01, batch_size=128, print_step=50)
print('fitting data starts...')
model.fit(train_X, epochs=5000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope="text")

