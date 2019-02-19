import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import argparse
import sys
import pickle
from dataset import Dataset




class Model(object):
    def __init__(self, tf_dim=20000, vae=False, deep=False):
        self.tfdim = tf_dim
        self.layers = [1000, 500, 100]
        self.z_dim = 50
        self.activation = None
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.learning_rate = 1e-4
        self.vae = vae
        self.deep = deep
        if not self.vae:
            self.layers.append(self.z_dim)

    def _enc(self, x, layers, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)):
                x_ = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="encode_%d"%i)
                x_ = tf.nn.leaky_relu(x_, 0.5)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.activation, scope="z_mu", weights_regularizer=self.regularizer)
            z_sigma = fully_connected(h, self.z_dim,  self.activation, scope="z_sigma",
                                      weights_regularizer=self.regularizer)
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), 1e-10)) * e
        return z, z_mu, z_sigma

    def _dec(self, x, layers, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)-1):
                x_ = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="decode_%d" % i)
                x_ = tf.nn.leaky_relu(x_, 0.5)
            x_ = fully_connected(x_, layers[-1],  weights_regularizer=self.regularizer)
        return x_

    def resnet(self, x, layers, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)):
                x_ = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="encode_%d" % i)
                net = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="deeper_%d" % i)
                # net = fully_connected(net, layers[i], activation_fn=self.activation,
                #                       weights_regularizer=self.regularizer,
                #                       scope="deeper2_%d" % i)
                x_ = tf.math.add(x_, net)
                if i != (len(layers) -1):
                    x_ = tf.nn.leaky_relu(x_, 0.5)
            return x_

    def mlp(self, x, layers, scope="rating"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)-1):
                x_ = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="encode_%d" % i)
                x_ = tf.nn.leaky_relu(x_, 0.5)
            x_ = fully_connected(x_, layers[-1], weights_regularizer=self.regularizer)
        return x_

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):

        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        return neg_ll

    def build_model(self):
        self.x_user = tf.placeholder(tf.float32, [None, self.tfdim])
        self.x_item = tf.placeholder(tf.float32, [None, self.tfdim])
        self.y = tf.placeholder(tf.float32, [None])


        if self.deep:
            z_user = self.resnet(self.x_user, self.layers, "user")
            z_item = self.resnet(self.x_item, self.layers, "item")
        else:
            z_user = self._enc(self.x_user, self.layers, "user")
            z_item = self._enc(self.x_item, self.layers, "item")

        if self.vae:
            layers = self.layers[::-1]
            layers.append(self.tfdim)
            user_h, user_mu, user_sigma = self.gen_z(z_user, "user")
            item_h, item_mu, item_sigma = self.gen_z(z_item, "item")
            user_gen = self._dec(user_h, layers, "user")
            item_gen = self._dec(item_h, layers, "item")
            z = tf.concat([user_mu, item_mu], axis=1)
        else:
            z = tf.concat([z_user, z_item], axis=1)

        self.pred = self.mlp(z, [20, 1], scope="rating")
        self.pred = tf.reshape(self.pred, [-1])
        self.loss = tf.losses.mean_squared_error(self.y, self.pred) + tf.losses.get_regularization_loss()
        if self.vae:
            self.loss += self.loss_reconstruct(self.x_user, user_gen) + self.loss_reconstruct(self.x_item, item_gen) +\
                            0.1 * self.loss_kl(user_mu, user_sigma) + 0.1 * self.loss_kl(item_mu, item_sigma)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

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

    batch_size = 500
    iter = 50

    model = Model(vae=args.vae, deep=args.deep)
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
            x_user, x_item, y_rating = dataset.create_tfidf(list_idx, k=args.k)
            feed_dict ={model.x_user: x_user,
                        model.x_item: x_item,
                        model.y: y_rating}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        print("Loss last batch: %f"%loss)

        if i%1 == 0:
            for j in range(int(test_no / batch_size)+1):
                idx = list(range(j*batch_size, min(test_no, (j+1)*batch_size)))
                x_user, x_item, y_rating = dataset.create_tfidf(idx, k=args.k, type="test")
                feed_dict = {model.x_user: x_user,
                             model.x_item: x_item,
                             model.y: y_rating}
                p = sess.run(model.pred, feed_dict=feed_dict)
                if j == 0:
                    error = p - y_rating
                else:
                    error = np.concatenate([error, p-y_rating], axis=0)
            mse = np.mean(error ** 2)
            print("rmse = %f"%mse)

        if i%30 == 0:
            model.learning_rate /= 10

if __name__ == '__main__':
    main()



