import tensorflow as tf
from tensorflow.layers import conv1d, dense, max_pooling1d, flatten
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

    def encode(self, x, filters, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(filters)):
                x_ = conv1d(x_, filters[i], 5)
                x_ = max_pooling1d(x_, 5, 5)
            x_ = flatten(x_)

        return x_

    def mlp(self, x, layers, scope="mlp"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)):
                x_ = dense(x_, layers[i], kernel_regularizer=self.regularizer, activation=self.activation)
        return x_




    def build_model(self):
        self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_dim]),
                        trainable=False, name="embedding")
        embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])
        embedding_init = self.embedding.assign(embedding_placeholder)

        self.X_user_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "X_user_ids")
        self.X_item_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "X_item_ids")
        self.y_review_ids = tf.placeholder(tf.int32, [None, self.seq_dim], "y_review_ids")
        self.y_rating = tf.placeholder(tf.int32, [None, 1], "y_rating")
        y = tf.one_hot(self.y_rating, 5)


        X_user = tf.nn.embedding_lookup(self.embedding, self.X_user_ids)
        X_item = tf.nn.embedding_lookup(self.embedding, self.X_item_ids)

        X_user_z = self.encode(X_user, self.filters, "user")
        X_item_z = self.encode(X_item, self.filters, "item")
        X = tf.concat([X_user_z, X_item_z], axis=1)

        X = self.mlp(X, self.mlp_layers)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=X))
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
    data = pickle.load(f, encoding='latin1')
    dataset = Dataset(data)

    filter = [32, 64, 128]
    mlp_layers = [20, 5]
    batch_size = 500
    iter = 100

    model = Model(filter, mlp_layers, dataset.vocab_size, dataset.embedding_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    embedding = dataset.embedding_matrix
    train_no = len(data['train'])
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

        if i%10 == 0:
            X_user, X_item, y_review, y_rating = dataset.create_batch(range(data['test']), k=2, type="test")
            feed_dict = {model.embedding: embedding,
                         model.X_user_ids: X_user,
                         model.X_item_ids: X_item,
                         model.y_review_ids: y_review,
                         model.y_rating: y_rating}
            pred = sess.run(model.X, feed_dict=feed_dict)
            pred = np.argmax(pred, axis=1)
            mse = np.mean((pred - y_rating) ** 2)
            print("rmse = %f"%mse)



if __name__ == '__main__':
    main()