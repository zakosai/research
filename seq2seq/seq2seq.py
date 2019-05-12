import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
import argparse
from dataset import Dataset, calc_recall
import os
from scipy.sparse import load_npz



class Seq2seq(object):
    def __init__(self):
        self.w_size = 10
        self.p_dim = 100
        self.n_products = 3706
        self.n_hidden = 256
        self.learning_rate = 1e-4
        self.train = True
        self.cat_dim = 18
        # self.item_cat = item_cat.astype(np.float32)


    def encoder_BiLSTM(self, X, scope, n_hidden):
        with tf.variable_scope("cell_def_%s"%scope):
            f_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            f_cell = tf.contrib.rnn.DropoutWrapper(cell=f_cell, output_keep_prob=0.8)
            b_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            b_cell = tf.contrib.rnn.DropoutWrapper(cell=b_cell, output_keep_prob=0.8)
        with tf.variable_scope("cell_op_%s"%scope):
            outputs1, last_state = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, X, sequence_length=self.seq_len,
                                                          dtype=tf.float32)

        outputs = tf.concat(outputs1, 2)

        return outputs, last_state


    def encoder_LSTM(self, X, n_layers):
        stack_cell = []
        for i in range(n_layers):
            with tf.variable_scope("encoder_%d"%i):
                cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
                # cell = tf.contrib.rnn.AttentionCellWrapper(
                #     cell, attn_length=24, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)
                stack_cell.append(cell)

        stack = tf.contrib.rnn.MultiRNNCell(stack_cell, state_is_tuple=True)

        # The second output is the last state and we will not use that
        outputs, last_state = tf.nn.dynamic_rnn(stack, X, self.seq_len, dtype=tf.float32)

        return outputs, last_state



    def attention(self, inputs):
        # Trainable parameters
        hidden_size = inputs.shape[2].value
        u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

        with tf.name_scope('v'):
            v = tf.tanh(inputs)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        # Final output with tanh
        output = tf.tanh(output)

        return output, alphas


    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        return neg_ll


    def prediction(self, x, y, cat=None, y_cat=None, reuse=False):
        with tf.variable_scope("last_layer", reuse=reuse):
            out = layers.fully_connected(x, self.n_products, tf.nn.tanh)
            # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out, 100))

            if cat !=None:
                pred_cat = layers.fully_connected(cat, self.cat_dim, tf.nn.tanh)
                pred_cat = tf.reshape(pred_cat, [-1, self.cat_dim, 1])
                out_cat = tf.matmul(tf.broadcast_to(self.item_cat, [tf.shape(cat)[0], self.item_cat.shape[0],
                                                                    self.item_cat.shape[1]]),  pred_cat)
                pred = out_cat * out

                loss = 10*self.loss_reconstruct(y, pred) + self.loss_reconstruct(y_cat, out_cat)
            else:
                loss = self.loss_reconstruct(y, out)

        return loss, out


    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.w_size, self.p_dim])
        # self.X_cat = tf.placeholder(tf.float32, [None, self.w_size, self.cat_dim])
        self.y = tf.placeholder(tf.float32, [None, self.n_products])
        # self.y_cat = tf.placeholder(tf.float32, [None, self.cat_dim])

        self.seq_len = tf.fill([tf.shape(self.X)[0]], self.w_size)

        # assert tf.shape(self.X)[0] == tf.shape(self.X_cat)[0]


        outputs, _ = self.encoder_BiLSTM(self.X, "1", self.n_hidden)

        outputs, _ = self.encoder_BiLSTM(outputs, "2", self.n_hidden*2)
        # with tf.variable_scope('attention'):
        #     outputs, self.alphas = self.attention(outputs)
        #
        # # Dropout
        # with tf.variable_scope('dropout'):
        #     outputs = tf.nn.dropout(outputs, 0.8)

        last_state = tf.reshape(outputs[:, -1, :], (-1, self.n_hidden*4))
        # last_state = outputs

        # Categories
        # out_cat, _ = self.encoder_BiLSTM(self.X_cat, "cat", self.n_hidden)
        # print(out_cat.shape)
        # last_state_cat = tf.reshape(out_cat[:, -1, :], (-1, self.n_hidden*2))

        self.loss, self.predict = self.prediction(last_state, self.y)
        # self.loss *=10
        # for i in range(self.w_size-1):
        #     x = tf.reshape(outputs[:, i, :], (-1, self.n_hidden))
        #     y = tf.reshape(self.X[:, i+1, -self.n_products:], (-1, self.n_products))
        #     loss, _ = self.prediction(x, y, True)
        #     self.loss += loss

        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.y, self.predict, 100))

        # self.loss = self.loss_reconstruct(self.y, self.predict)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



def main():
    iter = 3000
    batch_size = 1000
    args = parser.parse_args()
    dataset = args.data
    type = args.type
    num_p = args.num_p
    checkpoint_dir = "experiment/%s/%s/" % (dataset, type)

    data = Dataset(num_p, "data/%s/%s"%(dataset, type))
    # data.create_item_cat("data/%s/%s"%(dataset, type))
    text = load_npz("data/%s/item.npz"%dataset)
    print(text.shape)
    data.item_emb = text

    model = Seq2seq()
    # model.p_dim = data.n_user
    model.w_size = data.w_size = args.w_size
    model.p_dim = text.shape[1]
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0


    for i in range(1, iter):
        shuffle_idx = np.random.permutation(data.n_user)
        train_cost = 0
        data.create_train_iter()
        print(data.item_emb.shape)

        for j in range(int(data.n_user / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            X, y = data.create_batch(list_idx, data.X_iter, data.y_iter)

            feed = {model.X: X, model.y:y}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        if i % 10 == 0:
            model.train = False
            X_val, y_val = data.create_batch(range(len(data.val)), data.val, data.val_infer)
            feed = {model.X: X_val, model.y: y_val}
            loss_val, y_val = sess.run([model.loss, model.predict],
                                       feed_dict=feed)

            recall, _, _ = calc_recall(y_val, data.val, data.val_infer)
            print("Loss val: %f, recall %f" % (loss_val, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'bilstm-model'), i)

                X_test, y_test = data.create_batch(range(len(data.test)), data.test, data.infer2)
                feed = {model.X: X_test, model.y: y_test}
                loss_test, y = sess.run([model.loss, model.predict],
                                        feed_dict=feed)
                recall, hit, ndcg = calc_recall(y, data.test, data.infer2)
                np.savez(os.path.join(checkpoint_dir, "pred"), p_val=y_val, p_test=y)
                print("Loss test: %f, recall: %f, hit: %f, ndcg: %f" % (loss_test, recall, hit, ndcg))
            model.train = True
        if i % 100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 10
            print("decrease lr to %f" % model.learning_rate)

    print(max_recall)




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default="Tool",
                    help='dataset name')
parser.add_argument('--type', type=str, default="implicit",
                    help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')
parser.add_argument('--w_size', type=int, default=10, help='window size')

if __name__ == '__main__':
    main()