import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
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
        self.active_function = tf.nn.relu
        self.user_info_dim = user_info_dim
        self.item_info_dim = item_info_dim
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x

        # x_ = tf.nn.l2_normalize(x_, 1)
        x_ = tf.nn.dropout(x_, 0.5)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)):
                x_ = fully_connected(x_, decode_dim[i], self.active_function, scope="dec_%d" % i,
                                     weights_regularizer=self.regularizer)
            # x_ = tf.nn.sigmoid(x_)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.active_function, scope="z_mu")
            z_sigma = fully_connected(h, self.z_dim, self.active_function, scope="z_sigma")
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def vae(self, x, encode_dim, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            h = self.enc(x, "encode", encode_dim)
            z, z_mu, z_sigma = self.gen_z(h, "VAE")
        return z, z_mu, z_sigma

    def loss_kl(self, mu, log_sigma_sq):
        return -0.5 * tf.reduce_sum(1 + tf.clip_by_value(log_sigma_sq, -10.0, 10.0)
                                    - tf.clip_by_value(mu, -10.0, 10.0) ** 2
                                    - tf.exp(tf.clip_by_value(log_sigma_sq, -10.0, 10.0)), 1)

    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))

        return neg_ll

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.dim], name='input')
        self.user_info = tf.placeholder(tf.float32, [None, self.user_info_dim], name='user_info')

        h_x, h_x_mu, h_x_sigma = self.vae(self.user_info, self.encode_dim, "content")
        z_y, z_y_mu, z_y_sigma = self.vae(self.x, self.encode_dim, "rating")
        h_y, h_y_mu, h_y_sigma = self.vae(self.x, self.encode_dim, "added_kl")

        kl_z_y = tf.reduce_sum(self.loss_kl(z_y_mu, z_y_sigma))
        kl_h_x = tf.reduce_sum(self.loss_kl(h_x_mu, h_x_sigma))
        kl_h_xy = 0.5 * tf.reduce_sum(tf.reduce_sum(tf.exp(h_x_sigma)/tf.exp(h_y_sigma) + tf.square(h_y_mu-h_x_mu)/h_y_sigma +
                                                     h_y_sigma - h_x_sigma - 1, 1))

        self.y = self.dec(tf.concat((h_x, z_y), axis=-1), "decode", self.decode_dim)
        loss_recon = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=self.y))
        recon_h = self.dec(h_x, "decode_h", [200, self.user_info_dim])
        loss_recon_h = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.user_info, logits=recon_h))
        self.loss_enc = loss_recon + kl_z_y + 0.1 * kl_h_x + loss_recon_h

        self.train_op_enc = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_enc)
        self.loss_values = [self.loss_enc, loss_recon, kl_z_y, kl_h_x, kl_h_xy]


def main(args):
    iter = args.iter
    batch_size = 500

    dataset = Dataset(args.data_dir, args.data_type)
    model = Translation(batch_size, dataset.no_item, dataset.user_size, dataset.item_size,
                        [200], [200, dataset.no_item], 50, learning_rate=args.learning_rate)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    best = 0

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(range(len(dataset.transaction)))
        for j in range(int(len(shuffle_idx)/batch_size + 1)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x = dataset.transaction[list_idx]
            user_info = dataset.user_info[list_idx]
            feed = {model.x: x, model.user_info:user_info}

            _, loss = sess.run([model.train_op_enc, model.loss_values], feed_dict=feed)

            # print("loss user: %f, loss item: %f, loss pred: %f"%(loss, loss, loss))
        print(loss)

        # Validation Process
        if i%1 == 0:
            model.train = False
            loss_val_a, y_b = sess.run([model.loss_values, model.y],
                                              feed_dict={model.x: dataset.transaction, model.user_info:dataset.user_info})
            recall = recallK(dataset.train, dataset.test, y_b)
            print("recall: %f"%recall)
            model.train = True
            if recall > best:
                best = recall
        # if (i%50 == 0) :
        #     model.learning_rate /= 10
    print("[200]", best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='5')
    parser.add_argument('--data_dir', type=str, default='data/amazon')
    parser.add_argument('--iter', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    main(args)



