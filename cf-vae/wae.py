import tensorflow as tf
from tensorbayes.layers import dense, placeholder
import os
from keras.backend import binary_crossentropy
import numpy as np
import time
from scipy.sparse import load_npz
import argparse

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
            x_ = placeholder((None, self.input_dim))
            x = x_
            depth_inf = len(self.encoding_dims)

            # noisy_level = 1
            # x = x + noisy_level*tf.random_normal(tf.shape(x))
            with tf.variable_scope("encode"):
                for i in range(depth_inf):
                    x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)

                h_encode = x
                z_mu = dense(h_encode, self.z_dim, scope="mu_layer")
                z_log_sigma_sq = dense(h_encode, self.z_dim, scope = "sigma_layer")
                e = tf.random_normal(tf.shape(z_mu))
                z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

            # generative process
            y_true = self.decode(z)
            self.reconstructed = y_true

            z_fake = tf.random_uniform(tf.shape(z_mu), -1, 1)
            y_fake = self.decode(z_fake, reuse=True)

            self.wae_lambda = 0.5
            self.loss_gan, self.penalty = self.gan_penalty(z_fake, z)
            self.loss_reconstruct = self.reconstruction_loss(x_, self.reconstructed)
            self.wae_objective = self.loss_reconstruct + \
                                 self.wae_lambda * self.penalty

        z_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='z_adversary')
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encode')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decode')
        ae_vars = encoder_vars + decoder_vars

        ae_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.wae_objective,
                                   var_list=encoder_vars + decoder_vars)
        z_adv_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            loss=self.loss_gan[0], var_list=z_adv_vars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope))
        ckpt_file = os.path.join(self.ckpt,"wae_%s.ckpt" %scope)
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            start = time.time()
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                _, l, lr, lk = sess.run((ae_opt, self.wae_objective, self.penalty, self.loss_reconstruct), feed_dict={x_:x_batch})
                _,  = sess.run((z_adv_opt, self.loss_gan), feed_dict={x_:x_batch})
                if i % self.print_size == 0:
                    print("epoches: %d\t loss: %f\t loss penalty: %f\t loss res: %f\t loss gan: %f \t time: %d s"%(i, l,lr, lk, lg, time.time()-start))

            saver.save(sess, ckpt_file)
        else:
            saver.restore(sess, ckpt_file)

    def decode(self, z, reuse=False):
        with tf.variable_scope("decode", reuse=reuse):
            depth_gen = len(self.decoding_dims)
            y = z
            for i in range(depth_gen):
                y = dense(y, self.decoding_dims[i], scope="dec_layer" + "%s" % i, activation=tf.nn.sigmoid)
        return y

    def reconstruction_loss(self, real, reconstr):
        loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
        loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        return loss

    def gan_penalty(self, sample_qz, sample_pz):
        # Pz = Qz test based on GAN in the Z space
        logits_Pz = self.z_adversary(sample_pz)
        logits_Qz = self.z_adversary(sample_qz, reuse=True)
        loss_Pz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
        loss_Qz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
        loss_Qz_trick = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
        loss_adversary = self.wae_lambda * (loss_Pz + loss_Qz)
        # Non-saturating loss trick
        loss_match = loss_Qz_trick
        return (loss_adversary, logits_Pz, logits_Qz), loss_match

    def z_adversary(self, inputs, reuse=False):
        num_units = 100
        num_layers = 2
        nowozin_trick = 1
        # No convolutions as GAN happens in the latent space
        with tf.variable_scope('z_adversary', reuse=reuse):
            hi = inputs
            for i in xrange(num_layers):
                hi = dense(hi, num_units, scope='hi_%d'%i)
                hi = tf.nn.relu(hi)
            hi = dense(hi, 1, scope='hfinal_lin')
            # if nowozin_trick:
            #     # We are doing GAN between our model Qz and the true Pz.
            #     # Imagine we know analytical form of the true Pz.
            #     # The optimal discriminator for D_JS(Pz, Qz) is given by:
            #     # Dopt(x) = log dPz(x) - log dQz(x)
            #     # And we know exactly dPz(x). So add log dPz(x) explicitly
            #     # to the discriminator and let it learn only the remaining
            #     # dQz(x) term. This appeared in the AVB paper.
            #     assert opts['pz'] == 'normal', \
            #         'The GAN Pz trick is currently available only for Gaussian Pz'
            #     sigma2_p = float(opts['pz_scale']) ** 2
            #     normsq = tf.reduce_sum(tf.square(inputs), 1)
            #     hi = hi - normsq / 2. / sigma2_p \
            #          - 0.5 * tf.log(2. * np.pi) \
            #          - 0.5 * opts['zdim'] * np.log(sigma2_p)
        return hi


if __name__ == '__main__':

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
    train_X = data[idx]
    test_X = data[~idx]
    # print(train_X[0])
    #
    # images = np.fromfile("data/amazon/images.bin")
    # images = images.reshape((16000, 3072))
    # train_img = images[idx]
    # test_img = images[~idx]

    # model = vanilla_vae(input_dim=args.user_dim, encoding_dims=[100], z_dim=zdim, decoding_dims=[100, args.user_dim], loss='cross_entropy', ckpt_folder=ckpt)
    if args.type == "text":
        model = vanilla_vae(input_dim=8000, encoding_dims=[200, 100], z_dim=zdim, decoding_dims=[100, 200,8000], loss='cross_entropy', ckpt_folder=ckpt)
    # As there will be an additional layer from 100 to 50 in the encoder. in decoder, we also take this layer
                        # lr=0.01, batch_size=128, print_step=50)
        print('fitting data starts...')
        model.fit(train_X, epochs=10000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope="text")

    else:
        model = vanilla_vae(input_dim=args.user_dim, encoding_dims=[200], z_dim=zdim, decoding_dims=[200,args.user_dim], loss='cross_entropy', ckpt_folder=ckpt)
        print('fitting data starts...')
        model.fit(train_X, epochs=10000,learning_rate=0.001, batch_size=500, print_size=50, train=True, scope="user")
