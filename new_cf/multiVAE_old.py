import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from src.dataset import Dataset, recallK
import numpy as np
import argparse


class RSVAE:
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
        self.train = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
                x_ = batch_norm(x_, decay=0.995)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
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
            z, z_mu, z_sigma = self.gen_z(h, "VAE")
            loss_kl = self.loss_kl(z_mu, z_sigma)
            y = self.dec(z_mu, "decode", decode_dim)
        return z, y, loss_kl

    @staticmethod
    def loss_kl(mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    @staticmethod
    def loss_reconstruct(x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)
        neg_ll = -tf.reduce_mean(tf.reduce_sum(log_softmax_var * x, axis=-1))
        return neg_ll

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.dim], name='input')
        x = self.x

        # VAE for CF
        _, self.x_recon, loss_kl = self.vae(x, self.encode_dim, self.decode_dim, "CF")

        # Loss VAE
        self.loss = loss_kl + self.loss_reconstruct(self.x, self.x_recon) + \
                    tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def main(args):
    iter = args.iter
    batch_size = 500

    dataset = Dataset(args.data_dir, args.data_type)
    layers = [[200, 100]]
    for layer in layers:
        model = RSVAE(batch_size, dataset.no_item, dataset.user_size, dataset.item_size,
                            layer[:-1], layer[:-1][::-1]+[dataset.no_item], layer[-1], learning_rate=args.learning_rate)
        model.build_model()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        best = [0, 0, 0]
        best_ndcg, best_mAP = 0, 0

        for i in range(1, iter):
            # shuffle_idx = np.random.permutation(range(dataset.no_user))
            # for j in range(int(len(shuffle_idx) / batch_size)):
            #     list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            #     x = dataset.user_info[list_idx]
            #     feed = {model.user_info: x}
            #     _, loss_user = sess.run([model.train_op_user, model.loss_user], feed_dict=feed)
            #
            # shuffle_idx = np.random.permutation(range(dataset.no_item))
            # for j in range(int(len(shuffle_idx) / batch_size)):
            #     list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            #     x = dataset.item_info[list_idx]
            #     feed = {model.item_info: x}
            #     _, loss_item = sess.run([model.train_op_item, model.loss_item], feed_dict=feed)

            shuffle_idx = np.random.permutation(range(len(dataset.transaction)))
            for j in range(int(len(shuffle_idx)/batch_size + 1)):
                list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
                x = dataset.transaction[list_idx]
                feed = {model.x: x}

                _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

            # print("loss user: %f, loss item: %f, loss pred: %f"%(loss, loss, loss))

            # Validation Process
            if i%1 == 0:
                model.train = False
                loss_val_a, y_b = sess.run([model.loss, model.x_recon],
                                                  feed_dict={model.x: dataset.transaction})
                recall, ndcg, mAP = recallK(dataset.train, dataset.test, y_b, 50)
                # print("recall: %f, ndcg: %f" % (recall, ndcg))
                model.train = True
                if recall > best[0]:
                    best = [recall, ndcg, mAP]
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                if mAP > best_mAP:
                    best_mAP = mAP
            if (i % 10 == 0) and (model.learning_rate >= 1e-6):
                model.learning_rate /= 10
        print(layer, " : ", best, ", ", best_ndcg, ", ", best_mAP)
        tf.keras.backend.clear_session()

    # print(best, best_ndcg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='5')
    parser.add_argument('--data_dir', type=str, default='data/amazon')
    parser.add_argument('--iter', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)



