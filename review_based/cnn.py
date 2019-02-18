import tensorflow as tf
from tensorflow.layers import conv2d, dense, max_pooling2d, flatten, conv2d_transpose
from keras import metrics
import pickle
import argparse
import sys
from dataset import Dataset
import numpy as np
from attention import MultiHeadsAttModel, NormL

class Model(object):
    def __init__(self, filters, mlp_layers, vocab_size, embedding_dim, seq_dim=1000, learning_rate=1e-4,
                 attention=True, deep_model=True, vae=True):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.filters = filters
        self.mlp_layers = mlp_layers
        self.activation = tf.nn.relu
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.learning_rate = learning_rate
        self.attention = attention
        self.deep_model = deep_model
        self.vae = vae
        self.z_dim = 50

    def denseBlock(self, x, i, num_filters_per_size_i, cnn_filter_size_i=3, num_rep_block_i=4):
        with tf.variable_scope("dense_unit_%s" % i):
            nodes = []
            x_ = x
            x_ = conv2d(x_, num_filters_per_size_i, (cnn_filter_size_i, 1), padding='same')
            nodes.append(x_)
            print(x_.get_shape())
            for z in range(num_rep_block_i - 1):
                x_ = conv2d(tf.concat(nodes, 3), num_filters_per_size_i, (cnn_filter_size_i, 1), padding='same')
                nodes.append(x_)
                print(x_.get_shape())

            return x_

    def _enc(self, x, filters, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(filters)):
                x_ = conv2d(x_, filters[i], (2,1), padding='same')
                x_ = max_pooling2d(x_, (2, 1), (2, 1))
            x_ = max_pooling2d(x_, (8, 1), (8, 1))
            if self.attention:
                x_ = tf.reshape(x_, (-1, 8, 512))
                att = MultiHeadsAttModel(8, 512, 64, 512)
                x_ = att([x_, x_, x_])
                x_ = tf.reshape(x_, (-1, 8, 1, 512))
                # x_ = NormL()(x_)
            print(x_.get_shape())
            x_ = flatten(x_)
        return x_

    def _dec(self, x, filters, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            x_ = tf.reshape(x_, (-1, 8, 1, filters[0]))
            x_ = tf.image.resize_nearest_neighbor(x_, (64, 1))
            print(x_.get_shape())
            for i in range(1, len(filters)):
                x_ = conv2d_transpose(x_, filters[i], (3, 1), (2, 1), padding="same")
                print(x_.get_shape())
            x_ = conv2d_transpose(x_, self.embedding_dim, (3,1), (2, 1), padding="same")
            print(x_.get_shape())
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = dense(h, self.z_dim, kernel_regularizer=self.regularizer, activation=self.activation)
            z_sigma = dense(h, self.z_dim, kernel_regularizer=self.regularizer, activation=self.activation)
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def encode(self, x, filters, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            x_ = conv2d(x_, filters[0], (3, 1), padding='same')
            for i in range(len(filters)):
                x_ = self.denseBlock(x_, i, filters[i])
                x_ = max_pooling2d(x_, (2, 1), (2, 1))
            x_ = max_pooling2d(x_, (8, 1), (8, 1))
            if self.attention:
                x_ = tf.reshape(x_, (-1, 8, 512))
                att = MultiHeadsAttModel(8, 512, 64, 32)
                x_ = att([x_, x_, x_])
                x_ = tf.reshape(x_, (-1, 8, 1, 32))
                # x_ = NormL()(x_)
            print(x_.get_shape())
            x_ = flatten(x_)

        return x_


    # def decode(self, x, filters, scope="user"):
    #     x_ = x
    #     with tf.variable_scope(scope):
    #         for i in range(len(filters)):
    #             x_ = conv2d(x_, filters[i], (5, 1) )
    #             x_ = max_pooling2d(x_, (5, 1), (5, 1))
    #         x_ = flatten(x_)
    #
    #     return x_

    def mlp(self, x, layers, scope="mlp"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)-1):
                x_ = dense(x_, layers[i], kernel_regularizer=self.regularizer, activation=self.activation)
            x_ = dense(x_, layers[-1], kernel_regularizer=self.regularizer)
        return x_

    def rec_loss(self, x, x_rec):
        x = flatten(x)
        x_rec = flatten(x_rec)
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec)
        # print(loss.get_shape())
        # loss = tf.reduce_mean(loss, axis=0)
        # print(loss.get_shape())
        # return -loss
        loss = tf.reduce_mean(self.embedding_dim * self.seq_dim * metrics.binary_crossentropy(x, x_rec))
        return loss

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))


    def build_model(self):
        self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
                        trainable=False, name="embedding")
        embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])
        embedding_init = self.embedding.assign(embedding_placeholder)

        self.X_user_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "X_user_ids")
        self.X_item_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "X_item_ids")
        self.y_review_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "y_review_ids")
        self.y_rating = tf.placeholder(tf.int32, [None], "y_rating")
        y = tf.one_hot(self.y_rating, 5)


        X_user = tf.nn.embedding_lookup(self.embedding, self.X_user_ids)
        X_item = tf.nn.embedding_lookup(self.embedding, self.X_item_ids)
        X_user = tf.reshape(X_user, (-1, self.seq_dim, 1, self.embedding_dim))
        X_item = tf.reshape(X_item, (-1, self.seq_dim, 1, self.embedding_dim))

        if self.deep_model:
            X_user_z = self.encode(X_user, self.filters, "user")
            X_item_z = self.encode(X_item, self.filters, "item")
        else:
            X_user_z = self._enc(X_user, self.filters, "user")
            X_item_z = self._enc(X_item, self.filters, "item")

        if self.vae:
            h_user, X_user_mu, X_user_sigma = self.gen_z(X_user_z, "user")
            h_item, X_item_mu, X_item_sigma = self.gen_z(X_item_z, "item")
            h_user = dense(h_user, 4092,kernel_regularizer=self.regularizer, activation=self.activation)
            h_item = dense(h_item, 4092,kernel_regularizer=self.regularizer, activation=self.activation)
            X_user_rec = self._dec(h_user, self.filters[::-1], "dec_user")
            X_item_rec = self._dec(h_item, self.filters[::-1], "dec_item")

        if self.vae:
            X = tf.concat([X_user_mu, X_item_mu], axis=1)
        else:
            X = tf.concat([X_user_z, X_item_z], axis=1)

        X = self.mlp(X, self.mlp_layers)
        X = tf.reshape(X, [-1])
        print(y.shape, X.shape)
        # X = tf.clip_by_value(X, 1, 5)
        self.loss = tf.losses.mean_squared_error(self.y_rating, X) + tf.losses.get_regularization_loss()
        if self.vae:
            loss_rec = self.rec_loss(X_user, X_user_rec) + self.rec_loss(X_item, X_item_rec)
            self.loss += 0.1 * loss_rec
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.X = X


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

    filter = [64, 128, 256, 512]
    mlp_layers = [256, 1]
    batch_size = 256
    iter = 50

    model = Model(filter, mlp_layers, dataset.vocab_size, dataset.embedding_dim, seq_dim=dataset.max_sequence_length,
                  attention=args.attention, deep_model=args.deep, vae=args.vae)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    embedding = dataset.embedding_matrix
    train_no = len(data['train'])
    test_no = len(data['test'])
    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_no)
        train_cost = 0
        for j in range(int(train_no/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            X_user, X_item, y_review, y_rating = dataset.create_batch(list_idx, k=args.k)
            feed_dict ={model.embedding: embedding,
                        model.X_user_ids: X_user,
                        model.X_item_ids: X_item,
                        model.y_review_ids: y_review,
                        model.y_rating: y_rating}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
        print("Loss last batch: %f"%loss)

        if i%1 == 0:
            for j in range(int(test_no / batch_size)+1):
                idx = list(range(j*batch_size, min(test_no, (j+1)*batch_size)))
                X_user, X_item, y_review, y_rating = dataset.create_batch(idx, k=args.k, type="test")
                feed_dict = {model.embedding: embedding,
                             model.X_user_ids: X_user,
                             model.X_item_ids: X_item,
                             model.y_review_ids: y_review,
                             model.y_rating: y_rating}
                p = sess.run(model.X, feed_dict=feed_dict)
                if j == 0:
                    error = p - y_rating
                else:
                    error = np.concatenate([error, p-y_rating], axis=0)
            mse = np.mean(error ** 2)
            print("rmse = %f"%mse)



if __name__ == '__main__':
    main()