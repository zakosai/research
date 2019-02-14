import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm, maxout
import numpy as np
import os
import argparse
import pickle
from dataset import Dataset
import sys


class Translation:
    def __init__(self, batch_size, dim_A, dim_B, encode_dim_A, decode_dim_A, encode_dim_B, decode_dim_B, adv_dim_A,
                 adv_dim_B, z_dim, share_dim, rating_layers, z_A=None, z_B=None, eps=1e-10, lambda_0=0.1, lambda_1=0.1,
                 lambda_2=100,lambda_3=0.01,lambda_4=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.encode_dim_A = encode_dim_A
        self.encode_dim_B = encode_dim_B
        self.decode_dim_A = decode_dim_A
        self.decode_dim_B = decode_dim_B
        self.adv_dim_A = adv_dim_A
        self.adv_dim_B = adv_dim_B
        self.rating_layers = rating_layers
        self.z_dim = z_dim
        self.eps = eps
        self.share_dim = share_dim
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.learning_rate = learning_rate
        self.active_function = tf.nn.tanh
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.freeze = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.5)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                # y = maxout(x_, encode_dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)

                print(x_.shape)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)-1):
                x_ = fully_connected(x_, decode_dim[i], scope="dec_%d" % i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)
            x_ = fully_connected(x_, decode_dim[-1], scope="last_dec",
                             weights_regularizer=self.regularizer, trainable=self.freeze)
        return x_

    def mlp(self, x, layers, scope="rating"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)-1):
                x_ = fully_connected(x_, layers[i], tf.nn.relu, weights_regularizer=self.regularizer)
            x_ = fully_connected(x_, layers[-1], weights_regularizer=self.regularizer)
        return x_

    def adversal(self, x, scope, adv_dim, reuse=False):
        x_ = x

        with tf.variable_scope(scope, reuse=reuse):
            # if self.train:
            # x_ = tf.nn.dropout(x_, 0.7)
            for i in range(len(adv_dim)-1):
                x_ = fully_connected(x_, adv_dim[i], self.active_function, scope="adv_%d" % i)
            x_ = fully_connected(x_, adv_dim[-1], scope="adv_last")
        return x_

    def share_layer(self, x, scope, dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(dim)):
                x_ = fully_connected(x_, dim[i],  scope="share_%d"%i,
                                     weights_regularizer=self.regularizer)
                # y = maxout(x_, dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)

        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.active_function, scope="z_mu", weights_regularizer=self.regularizer)
            z_sigma = fully_connected(h, self.z_dim,  self.active_function, scope="z_sigma",
                                      weights_regularizer=self.regularizer)
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def encode(self, x, scope, dim, reuse_enc, reuse_share, reuse_z=False):
        h = self.enc(x, "encode_%s"%scope, dim, reuse_enc)
        h = self.share_layer(h, "encode", self.share_dim, reuse_share)
        z, z_mu, z_sigma = self.gen_z(h, "encode", reuse=reuse_z)
        return z, z_mu, z_sigma

    def decode(self, x, scope, dim, reuse_dec, reuse_share):
        x = self.share_layer(x, "decode", self.share_dim[::-1], reuse_share)
        x = self.dec(x, "decode_%s"%scope, dim, reuse_dec)
        return x

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):

        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        return neg_ll


    # def loss_recsys(self, pred, label):
    #     return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(label, pred), axis=1))

    def loss_discriminator(self, x, x_fake):
        loss_real = tf.reduce_mean(tf.squared_difference(x, 1))
        loss_fake = tf.reduce_mean(tf.squared_difference(x_fake, 0))
        return loss_real + loss_fake
    def loss_generator(self, x):
        return tf.reduce_mean(tf.squared_difference(x, 1))


    def build_model(self):
        self.x_A = tf.placeholder(tf.float32, [None, self.dim_A], name='input_A')
        self.x_B = tf.placeholder(tf.float32, [None, self.dim_B], name='input_B')
        self.y = tf.placeholder(tf.float32, [None], name="rating")

        x_A = self.x_A
        x_B = self.x_B

        # VAE for domain A
        z_A, z_mu_A, z_sigma_A = self.encode(x_A, "A", self.encode_dim_A, False, False)
        y_AA = self.decode(z_A, "A", self.decode_dim_A, False, False)

        # VAE for domain B
        z_B, z_mu_B, z_sigma_B = self.encode(x_B, "B", self.encode_dim_B, False, True, True)
        y_BB = self.decode(z_B, "B", self.decode_dim_B, False, True)

        # Adversal
        y_BA = self.decode(z_B, "A", self.decode_dim_A, True, True)
        adv_AA = self.adversal(y_AA, "adv_A", self.adv_dim_A)
        adv_BA = self.adversal(y_BA, "adv_A", self.adv_dim_A, reuse=True)


        y_AB = self.decode(z_A, "B", self.decode_dim_B, True, True)
        adv_BB = self.adversal(y_BB, "adv_B", self.adv_dim_B)
        adv_AB = self.adversal(y_AB, "adv_B", self.adv_dim_B, reuse=True)

        # Cycle - Consistency
        z_ABA, z_mu_ABA, z_sigma_ABA = self.encode(y_AB, "B", self.encode_dim_B, True, True, True)
        y_ABA = self.decode(z_ABA, "A", self.decode_dim_A, True, True)
        z_BAB, z_mu_BAB, z_sigma_BAB = self.encode(y_BA, "A", self.encode_dim_A, True, True, True)
        y_BAB = self.decode(z_BAB, "B", self.decode_dim_B, True, True)


        # predict rating
        z = tf.concat([y_AB, y_BA], axis=1)
        rating_pred = self.mlp(z, self.rating_layers)
        rating_pred = tf.reshape(rating_pred, [-1])


        # Loss VAE
        loss_VAE_A = self.lambda_1 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_2 * self.loss_reconstruct(x_A, y_AA)
        loss_VAE_B = self.lambda_1 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_2 * self.loss_reconstruct(x_B, y_BB)
        self.loss_VAE = loss_VAE_A + loss_VAE_B

        # Loss GAN
        loss_d_A = self.lambda_0 * self.loss_discriminator(adv_AA, adv_BA)
        loss_d_B = self.lambda_0 * self.loss_discriminator(adv_BB, adv_AB)
        self.loss_d= loss_d_A + loss_d_B
        self.adv_AA = adv_AA
        self.adv_AB = adv_BA
        self.y_AA = y_AA
        self.y_BB = y_BB

        # Loss cycle - consistency (CC)
        loss_CC_A = self.lambda_3 * self.loss_kl(z_mu_ABA, z_sigma_ABA) + \
                    self.lambda_4 * self.loss_reconstruct(x_A,y_BA) + self.lambda_4 * self.loss_reconstruct(x_A, y_ABA)
        loss_CC_B = self.lambda_3 * self.loss_kl(z_mu_BAB, z_sigma_BAB) + self.lambda_4 * \
                    self.loss_reconstruct(x_B,y_AB) + self.lambda_4 * self.loss_reconstruct(x_B, y_BAB)



        self.loss_CC = loss_CC_A + loss_CC_B

        self.loss_val_a = self.lambda_4 * self.loss_reconstruct(x_A, y_BA)
        self.loss_val_b = self.lambda_4 * self.loss_reconstruct(x_B, y_AB)
        self.y_BA = y_BA
        self.y_AB = y_AB

        # loss rating
        print(self.y.get_shape, rating_pred.get_shape)
        self.loss_rating = tf.losses.mean_squared_error(self.y, rating_pred)

        self.loss_gen = self.loss_CC + 0.1 * tf.losses.get_regularization_loss() +\
                        self.loss_generator(y_AB) + self.loss_generator(y_ABA) + self.loss_generator(y_BAB) +\
                        self.loss_generator(y_BA) + self.loss_rating
        # self.loss_gen = drself.loss_CC + 0.1 * tf.losses.get_regularization_loss() - loss_d_A - loss_d_B



        self.loss_dis = loss_d_A + loss_d_B


        self.train_op_VAE_A = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_VAE_A)
        self.train_op_VAE_B = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_VAE_B)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_gen)

        adv_var_A = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv_A")
        adv_var_B = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv_B")
        print(adv_var_A, adv_var_B)
        self.train_op_dis_A = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_d_A,
                                                                                         var_list=adv_var_A)
        self.train_op_dis_B = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_d_B,
                                                                                  var_list=adv_var_B)

        self.pred = rating_pred


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare dataset'
    )
    parser.add_argument(
        '--data',
        default='data/Tool/dataset.pkl',
        dest='data',
        help='data file',
        type=str
    )
    parser.add_argument(
        '--output',
        default='experiment',
        dest='folder',
        help='where to experiment',
        type=str
    )
    parser.add_argument(
        '--attention',
        default=False,
        dest='attention',
        help='using attention or not',
        type=bool
    )
    parser.add_argument(
        '--deep',
        default=False,
        dest='deep',
        help='using deep model or not',
        type=bool
    )
    parser.add_argument(
        '--vae',
        default=False,
        dest='vae',
        help='using vae model or not',
        type=bool
    )
    parser.add_argument(
        '--k',
        default=2,
        dest='k',
        help='using k review',
        type=int
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main():
    args = parse_args()
    f = open(args.data, "rb")
    data = pickle.load(f)
    dataset = Dataset(data, max_sequence_length=1024)

    iter = 500
    batch_size = 500
    z_dim = 50
    adv_dim_A = adv_dim_B = rating_layers = [100, 1]
    encoding_dim_A = [200]
    encoding_dim_B = [200]
    share_dim = [100]
    decoding_dim_A = [200, data['item_no']]
    decoding_dim_B = [200, data['user_no']]

    model = Translation(batch_size, data['item_no'], data['user_no'], encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim, rating_layers)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    train_no = len(data['train'])
    test_no = len(data['test'])
    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_no)
        train_cost = 0
        for j in range(int(train_no/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            user, item, rating = dataset.create_implicit_batch(list_idx, "train")
            feed ={model.x_A: user,
                        model.x_B: item,
                        model.y: rating}
            if i < 20:
                _, loss_vae = sess.run([model.train_op_VAE_A, model.loss_VAE], feed_dict=feed)
                _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)

            else:
                model.freeze = False
                _, loss_gen, loss_vae, loss_cc, loss_rating = sess.run([model.train_op_gen, model.loss_gen,
                                                                      model.loss_VAE,
                                                        model.loss_CC, model.loss_rating], feed_dict=feed)



        if i%10 == 0 and i > 20:
            print("Loss last batch: %f" % loss_rating)
            for j in range(int(test_no / batch_size)+1):
                idx = list(range(j*batch_size, min(test_no, (j+1)*batch_size)))
                user, item, rating = dataset.create_implicit_batch(idx, "test")
                feed_dict = {model.x_A: user,
                             model.x_B: item,
                             model.y: rating}


                p = sess.run(model.pred, feed_dict=feed_dict)
                if j == 0:
                    error = p - rating
                else:
                    error = np.concatenate([error, p-rating], axis=0)
            mse = np.mean(error ** 2)
            print("rmse = %f"%mse)
        if i%100 == 0:
            model.learning_rate /= 10

if __name__ == '__main__':
    main()