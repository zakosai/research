import tensorflow as tf
from tensorflow.layers import conv2d, dense, max_pooling2d, flatten
import tensorflow.contrib.slim as slim
import pickle
import argparse
import sys
from dataset import Dataset
import numpy as np

class Model(object):
    def __init__(self, filters, mlp_layers, vocab_size, embedding_dim, seq_dim=1000, learning_rate=1e-4):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.filters = filters
        self.mlp_layers = mlp_layers
        self.activation = tf.nn.relu
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.learning_rate = learning_rate

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

    def encode(self, x, filters, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            x_ = conv2d(x_, filters[0], (3, 1), padding='same')
            for i in range(len(filters)):
                x_ = self.denseBlock(x_, i, filters[i])
                x_ = max_pooling2d(x_, (2, 1), (2, 1))
            x_ = max_pooling2d(x_, (8, 1), (8, 1))
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

        X_user_z = self.encode(X_user, self.filters, "user")
        X_item_z = self.encode(X_item, self.filters, "item")
        X = tf.concat([X_user_z, X_item_z], axis=1)

        X = self.mlp(X, self.mlp_layers)
        X = tf.reshape(X, [-1])
        print(y.shape, X.shape)
        self.loss = tf.losses.mean_squared_error(self.y_rating, X) + 0.1* tf.losses.get_regularization_loss()
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
    mlp_layers = [20, 1]
    batch_size = 128
    iter = 20

    model = Model(filter, mlp_layers, dataset.vocab_size, dataset.embedding_dim, seq_dim=dataset.max_sequence_length)
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
            X_user, X_item, y_review, y_rating = dataset.create_batch(list_idx)
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
                X_user, X_item, y_review, y_rating = dataset.create_batch(idx, k=2, type="test")
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