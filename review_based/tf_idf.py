import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import argparse
import sys
import pickle
from dataset import Dataset




class Model(object):
    def __init__(self, tf_dim=8000):
        self.tfdim = tf_dim
        self.layers = [600, 200, 50]
        self.activation = tf.nn.sigmoid
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.learning_rate = 1e-4

    def _enc(self, x, layers, scope="user"):
        x_ = x
        with tf.variable_scope(scope):
            for i in range(len(layers)):
                x_ = fully_connected(x_, layers[i], activation_fn=self.activation, weights_regularizer=self.regularizer,
                                     scope="encode_%d"%i)
                x_ = tf.nn.leaky_relu(x_, 0.5)
        return x_


    def build_model(self):
        self.x_user = tf.placeholder(tf.float32, [None, self.tfdim])
        self.x_item = tf.placeholder(tf.float32, [None, self.tfdim])
        self.y = tf.placeholder(tf.float32, [None])

        z_user = self._enc(self.x_user, self.layers, "user")
        z_item = self._enc(self.x_item, self.layers, "item")
        z = tf.concat([z_user, z_item], axis=1)

        self.pred = self._enc(z, [10, 1], scope="rating")
        self.pred = tf.reshape(self.pred, [-1])
        self.loss = tf.losses.mean_squared_error(self.y, self.pred) + tf.losses.get_regularization_loss()
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

    model = Model()
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

if __name__ == '__main__':
    main()



