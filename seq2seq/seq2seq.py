import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
import argparse
from dataset import Dataset, calc_recall
import os


class Seq2seq(object):
    def __init__(self):
        self.w_size = 10
        self.p_dim = 100
        self.n_products = 3706
        self.n_hidden = 256
        self.learning_rate = 1e-4
        self.train = True

    def prediction(self, x, y, reuse=False):
        with tf.variable_scope("last_layer", reuse=reuse):
            out = layers.fully_connected(x, self.n_products, tf.nn.relu)
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out, 100))

        return loss, out

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.w_size, self.p_dim])
        self.y = tf.placeholder(tf.float32, [None, self.n_products])

        self.seq_len = tf.fill([tf.shape(self.X)[0]], self.w_size)

        # with tf.variable_scope("cell_def_1"):
        #     f_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)
        #     b_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)
        #
        # with tf.variable_scope("cell_op_1"):
        #     outputs1, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, self.X, sequence_length=self.seq_len,
        #                                                   dtype=tf.float32)
        #
        # outputs = tf.concat(outputs1, 2)
        # print(outputs.shape)
        # last_state = tf.reshape(outputs[:, -1, :], (-1, self.n_hidden*2))
        cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
        # cell = tf.contrib.rnn.AttentionCellWrapper(
        #     cell, attn_length=24, state_is_tuple=True)
        if self.train:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

        cell1 = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
        # cell1 = tf.contrib.rnn.AttentionCellWrapper(
        #     cell1, attn_length=16, state_is_tuple=True)
        if self.train:
            cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

        # The second output is the last state and we will not use that
        outputs, _ = tf.nn.dynamic_rnn(stack, self.X, self.seq_len, dtype=tf.float32)
        # attention_output, alphas = self.attention(outputs, 256, return_alphas=True)
        last_state = tf.reshape(outputs[:, -1, :], (-1, self.n_hidden))
        # # last_state = attention_output
        # self.predict =layers.fully_connected(last_state, 24)
        if not self.train:
            self.predict = layers.fully_connected(outputs, self.n_products, tf.nn.relu)

        self.loss, self.predict = self.prediction(last_state, self.y)
        # self.loss *=10
        for i in range(self.w_size-1):
            x = tf.reshape(outputs[:, i, :], (-1, self.n_hidden))
            y = tf.reshape(self.X[:, i+1, -self.n_products:], (-1, self.n_products))
            loss, _ = self.prediction(x, y, True)
            self.loss += loss

        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.y, self.predict, 100))

        # self.loss = self.loss_reconstruct(self.y, self.predict)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



def main():
    iter = 3000
    batch_size = 500
    args = parser.parse_args()
    dataset = args.data
    type = args.type
    num_p = args.num_p
    checkpoint_dir = "experiment/%s/%s/" % (dataset, type)

    data = Dataset(num_p, "data/%s/%s"%(dataset, type))

    model = Seq2seq()
    model.p_dim = data.n_user
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0


    for i in range(1, iter):
        shuffle_idx = np.random.permutation(data.n_user)
        train_cost = 0
        data.create_train_iter()
        for j in range(int(data.n_user / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            X, y = data.create_batch(list_idx, data.X_iter, data.y_iter)

            feed = {model.X: X, model.y:y}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        if i % 10 == 0:
            model.train = False
            X_val, _ = data.create_batch(range(len(data.val)), data.val, data.val_infer)
            loss_val, y_val = sess.run([model.loss, model.predict],
                                       feed_dict={model.X: X_val})

            recall, _, _ = calc_recall(y_val, data.val, data.val_infer)
            print("Loss val: %f, recall %f" % (loss_val, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'multi-VAE-model'), i)

                X_test, _ = data.create_batch(range(len(data.test)), data.test, data.infer2)
                loss_test, y = sess.run([model.loss, model.predict],
                                        feed_dict={model.X: X_test})
                recall, hit, ndcg = calc_recall(y, data.test, data.infer2)
                print("Loss test: %f, recall: %f, hit: %f, ndcg: %f" % (loss_test, recall, hit, ndcg))
            model.train = True
        if i % 100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 2
            print("decrease lr to %f" % model.learning_rate)

    print(max_recall)




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default="Tool",
                    help='dataset name')
parser.add_argument('--type', type=str, default="implicit",
                    help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')


if __name__ == '__main__':
    main()