import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from keras.backend import binary_crossentropy
from src.dataset import Dataset, recallK
import numpy as np
import argparse


class Translation:
    def __init__(self, batch_size, dim, user_info_dim, item_info_dim, encode_dim, decode_dim, z_dim, eps=1e-10,
                 lambda_1=0.1, lambda_2=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim = dim
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.z_dim = z_dim
        self.eps = eps
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.learning_rate = learning_rate
        self.active_function = tf.nn.tanh
        self.user_info_dim = user_info_dim
        self.item_info_dim = item_info_dim
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x

        # x_ = tf.nn.l2_normalize(x_, 1)
        x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
                x_ = batch_norm(x_, decay=0.995)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)):
                x_ = fully_connected(x_, decode_dim[i], self.active_function, scope="dec_%d" % i,
                                     weights_regularizer=self.regularizer)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.active_function, scope="z_mu")
            z_sigma = fully_connected(h, self.z_dim, self.active_function, scope="z_sigma")
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def vae(self, x, encode_dim, decode_dim, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            h = self.enc(x, "encode", encode_dim)
            # if scope == "CF":
            #     y = self.dec(h, "decode", decode_dim)
            #     return y
            z, z_mu, z_sigma = self.gen_z(h, "VAE")
            loss_kl = self.loss_kl(z_mu, z_sigma)
            y = self.dec(z_mu, "decode", decode_dim)
        return z, y, loss_kl

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = - tf.reduce_mean(tf.reduce_sum(log_softmax_var * x, axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        return neg_ll

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.dim], name='input')
        self.user_info = tf.placeholder(tf.float32, [None, self.user_info_dim], name='user_info')
        self.item_info = tf.placeholder(tf.float32, [None, self.item_info_dim], name='item_info')

        # VAE for user
        z_user, user_recon, loss_kl_user = self.vae(self.user_info, [200], [200, self.user_info_dim], "user")
        self.loss_user = self.lambda_2 * tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.user_info, user_recon), axis=1)) +\
             self.lambda_1 * loss_kl_user + self.lambda_1 * tf.losses.get_regularization_loss()

        # VAE for item
        z_item, item_recon, loss_kl_item = self.vae(self.item_info, [400, 200], [200, 400, self.item_info_dim], "item")
        self.loss_item = self.lambda_2 * tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.item_info, item_recon), axis=1)) +\
                         self.lambda_1 * loss_kl_item + self.lambda_1 * tf.losses.get_regularization_loss()

        content_matrix = tf.matmul(z_user, tf.transpose(z_item))
        content_matrix = tf.nn.l2_normalize(content_matrix)
        # min = tf.reduce_min(content_matrix, axis=1, keep_dims=True)
        # max = tf.reduce_max(content_matrix, axis=1, keep_dims=True)
        # content_matrix = (content_matrix - min) / (max - min)
        x = self.x * content_matrix
        # VAE for CF
        _, self.x_recon, loss_kl = self.vae(x, self.encode_dim, self.decode_dim, "CF")
        # Loss VAE
        self.loss = loss_kl + self.loss_reconstruct(self.x, self.x_recon) + \
                    2 * tf.losses.get_regularization_loss()
        # self.x_recon = self.vae(x, self.encode_dim, self.decode_dim, "CF")
        # self.loss = self.loss_reconstruct(self.x, self.x_recon) + 2 * tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.train_op_user = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_user)
        self.train_op_item = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_item)


def main(args):
    iter = args.iter
    batch_size = 500

    dataset = Dataset(args.data_dir, args.data_type)
    model = Translation(batch_size, dataset.no_item, dataset.user_size, dataset.item_size,
                        [100], [100, dataset.no_item], 50, learning_rate=args.learning_rate)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    best = 0
    iter_no = int(dataset.no_user / batch_size + 1)

    for i in range(1, 30):
        shuffle_idx = np.random.permutation(range(dataset.no_user))
        for j in range(iter_no):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            x = dataset.user_info[list_idx]
            feed = {model.user_info: x}
            _, loss_user = sess.run([model.train_op_user, model.loss_user], feed_dict=feed)

        shuffle_idx = np.random.permutation(range(dataset.no_item))
        for j in range(int(len(shuffle_idx) / batch_size + 1)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            x = dataset.item_info[list_idx]
            feed = {model.item_info: x}
            _, loss_item = sess.run([model.train_op_item, model.loss_item], feed_dict=feed)

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(range(dataset.no_user))
        for j in range(iter_no):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            x = dataset.user_info[list_idx]
            feed = {model.user_info: x}
            _, loss_user = sess.run([model.train_op_user, model.loss_user], feed_dict=feed)

        shuffle_idx = np.random.permutation(range(dataset.no_item))
        for j in range(int(len(shuffle_idx) / batch_size + 1)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            x = dataset.item_info[list_idx]
            feed = {model.item_info: x}
            _, loss_item = sess.run([model.train_op_item, model.loss_item], feed_dict=feed)

        shuffle_idx = np.random.permutation(range(len(dataset.transaction)))
        for j in range(iter_no):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x = dataset.transaction[list_idx]
            feed = {model.x: x,
                    model.user_info: dataset.user_info[list_idx],
                    model.item_info: dataset.item_info}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        print("loss user: %f, loss item: %f, loss pred: %f"%(loss_user, loss_item, loss))

        # Validation Process
        if i%1 == 0:
            model.train = False
            loss_val_a, y_b = sess.run([model.loss, model.x_recon],
                                              feed_dict={model.x: dataset.transaction,
                                                         model.user_info: dataset.user_info,
                                                         model.item_info: dataset.item_info})
            recall = recallK(dataset.train, dataset.test, y_b)
            print("recall: %f"%recall)
            model.train = True
            if recall > best:
                best = recall
        if (i%10 == 0) and (i < 30):
            model.learning_rate /= 10
    print(best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='5')
    parser.add_argument('--data_dir', type=str, default='data/amazon')
    parser.add_argument('--iter', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)



