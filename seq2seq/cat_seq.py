import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
import argparse
from dataset import Dataset, calc_recall
import os
import time
from scipy.sparse import load_npz



class Seq2seq(object):
    def __init__(self, n_layer=2, bilstm=True):
        self.w_size = 10
        self.p_dim = 100
        self.n_products = 3706
        self.n_hidden = 64
        self.learning_rate = 1e-3
        self.train = True
        self.cat_dim = 18
        self.n_layer = n_layer
        self.bilstm = bilstm
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
        self.y = tf.placeholder(tf.float32, [None, self.w_size, self.n_products])
        # self.y_cat = tf.placeholder(tf.float32, [None, self.cat_dim])

        self.seq_len = tf.fill([tf.shape(self.X)[0]], self.w_size)

        # assert tf.shape(self.X)[0] == tf.shape(self.X_cat)[0]

        if self.bilstm:
            for i in range(self.n_layer):
                outputs, _ = self.encoder_BiLSTM(self.X, str(i+1), self.n_hidden*2**i)
            last_state = tf.reshape(outputs[:, -1, :], (-1, self.n_hidden * 2**(i+1)))
        else:
            for i in range(self.n_layer):
                outputs, _ = self.encoder_LSTM(self.X, str(i+1), self.n_hidden)
            last_state = tf.reshape(outputs[:, -1, :], (-1, self.n_hidden))

        # with tf.variable_scope('attention'):
        #     outputs, self.alphas = self.attention(outputs)
        #
        # # Dropout
        # with tf.variable_scope('dropout'):
        #     outputs = tf.nn.dropout(outputs, 0.8)

        # last_state = outputs

        self.loss, self.predict = self.prediction(last_state, tf.reshape(self.y[:, -1, :], (-1, self.n_products)))
        # self.loss *=10
        # for i in range(self.w_size-1):
        #     x = tf.reshape(outputs[:, i, :], (-1, self.n_hidden *2**self.n_layer))
        #     y = tf.reshape(self.y[:, i+1, :], (-1, self.n_products))
        #     loss, _ = self.prediction(x, y, reuse=True)
        #     self.loss += loss

        # self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.y, self.predict, 100))

        # self.loss = self.loss_reconstruct(self.y, self.predict)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



def main():
    iter = 1000
    batch_size = 500
    args = parser.parse_args()
    dataset = args.data
    type = args.type
    num_p = len(list(open("data/%s/item_id.txt"%(dataset))))
    checkpoint_dir = "experiment/%s/%s/" % (dataset, type)

    data = Dataset(num_p, "data/%s"%(dataset), args.w_size, cat=args.cat, time=args.time)
    # data.create_item_cat("data/%s/%s"%(dataset, type))

    # data.item_emb = text.toarray()

    model = Seq2seq(args.n_layers, args.bilstm)
    # model.p_dim = data.n_user
    model.n_products = data.n_item
    model.w_size = args.w_size
    model.p_dim = 0
    if args.cat:
        print(data.item_cat.shape[1])
        model.p_dim += data.item_cat.shape[1]
    if args.time:
        model.p_dim += data.time_dim

    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    max_recall = 0

    f = open("experiment/result.txt", "a")
    f.write("-------------------------\n")
    f.write("Data: %s - num_p: %d - seq2seq\nbilstm: %s - n_layers: %d - w_size:%d\n"
            %(dataset, data.n_item, model.bilstm, model.n_layer, model.w_size))
    f.write("cat: %s - time: %s\n"%(args.cat, args.time))
    result = [0,0,0,0]
    print(len(data.val))
    for i in range(1, iter):
        shuffle_idx = np.random.permutation(data.n_user)
        train_cost = 0
        data.create_train_iter()

        for j in range(int(data.n_user / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            if args.time:
                X, y = data.create_batch_tmp(list_idx, data.X_iter, data.y_iter, data.time_emb)
            else:
                X, y = data.create_batch_tmp(list_idx, data.X_iter, data.y_iter)

            feed = {model.X: X, model.y:y}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        if i % 10 == 0:
            model.train = False
            if args.time:
                X_val, y_val = data.create_batch_tmp(range(len(data.val)), data.val, data.val_infer, data.time_emb_val)
            else:
                X_val, y_val = data.create_batch_tmp(range(len(data.val)), data.val, data.val_infer)
            for j in range(int(len(X_val) / batch_size)+1):
                if (j + 1) * batch_size > len(X_val):
                    X_b_val = X_val[j * batch_size:]
                    y_b = y_val[j*batch_size:]
                else:
                    X_b_val = X_val[j * batch_size:(j + 1) * batch_size]
                    y_b = y_val[j*batch_size : (j+1) * batch_size]
                feed = {model.X: X_b_val, model.y:y_b}
                loss_val, y_b_val = sess.run([model.loss, model.predict],
                                           feed_dict=feed)
                if j == 0:
                    p_val = y_b_val
                else:
                    p_val = np.concatenate((p_val, y_b_val), axis=0)

            recall, _, _ = calc_recall(p_val, data.val, data.val_infer)
            print("Loss val: %f, recall %f" % (loss_val, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'bilstm-model'))

                if args.time:
                    X_test, y_test = data.create_batch_tmp(range(len(data.test)), data.test, data.infer2, data.time_emb_test)
                else:
                    X_test, y_test = data.create_batch_tmp(range(len(data.test)), data.test, data.infer2)
                for j in range(int(len(X_test) / batch_size) + 1):
                    if (j + 1) * batch_size > len(X_test):
                        X_b_val = X_test[j * batch_size:]
                        y_b = y_test[j * batch_size:]
                    else:
                        X_b_val = X_test[j * batch_size:(j + 1) * batch_size]
                        y_b = y_test[j * batch_size:(j + 1) * batch_size]
                    feed = {model.X: X_b_val, model.y:y_b}
                    loss_val, y_b_val = sess.run([model.loss, model.predict],
                                                 feed_dict=feed)
                    if j == 0:
                        y = y_b_val
                    else:
                        y = np.concatenate((y, y_b_val), axis=0)
                recall, hit, ndcg = calc_recall(y, data.test, data.infer2)
                np.savez(os.path.join(checkpoint_dir, "pred_cat"), p_val=y_val, p_test=y)
                print("recall: %f, hit: %f, ndcg: %f" % (recall, hit, ndcg))
                if recall > result[1]:
                    result = [i, recall, hit, ndcg]
            model.train = True
        if i % 100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 2
            print("decrease lr to %f" % model.learning_rate)
    f.write("iter: %d - recall: %f - hit: %f - ndcg: %f\n"%(result[0], result[1], result[2], result[3]))

    print(max_recall)




parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default="Tool",
                    help='dataset name')
parser.add_argument('--type', type=str, default="implicit",
                    help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')
parser.add_argument('--w_size', type=int, default=10, help='window size')
parser.add_argument('--bilstm', type=bool, default=True, help='window size')
parser.add_argument('--n_layers', type=int, default=2, help='window size')
parser.add_argument('--cat', type=bool, default=False, help='window size')
parser.add_argument('--time', type=bool, default=False, help='window size')






if __name__ == '__main__':
    main()