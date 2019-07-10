import tensorflow as tf
from tensorflow.contrib import rnn, layers
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
import numpy as np
import argparse
from dataset import Dataset, calc_recall
import os
import math
from scipy.sparse import load_npz


class Seq2seq(object):
    def __init__(self, n_layers=2, model_type='bilstm'):
        self.w_size = 10
        self.p_dim = 100
        self.n_products = 3706
        self.n_hidden = 512
        self.learning_rate = 1e-3
        self.train = True
        self.cat_dim = 18
        self.layers = [1000, 100]
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.active_function = tf.nn.tanh
        self.n_layers = n_layers
        self.model_type = model_type


    def encoder_BiLSTM(self, X, scope, n_hidden):
        with tf.variable_scope("cell_def_%s"%scope):
            f_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            f_cell = tf.contrib.rnn.DropoutWrapper(cell=f_cell, output_keep_prob=0.8)
            b_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
            b_cell = tf.contrib.rnn.DropoutWrapper(cell=b_cell, output_keep_prob=0.8)
        with tf.variable_scope("cell_op_%s"%scope):
            outputs1, last_state = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, X, sequence_length=self.seq_len,dtype=tf.float32)

        outputs = tf.concat(outputs1, 2)

        return outputs, last_state

    def encoder_biGRU(self, X, scope, n_hidden):
        with tf.variable_scope("cell_def_%s" % scope):
            f_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=tf.nn.tanh)
            f_cell = tf.contrib.rnn.DropoutWrapper(cell=f_cell, output_keep_prob=0.8)
            b_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=tf.nn.tanh)
            b_cell = tf.contrib.rnn.DropoutWrapper(cell=b_cell, output_keep_prob=0.8)
        with tf.variable_scope("cell_op_%s" % scope):
            outputs1, last_state = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, X, sequence_length=self.seq_len,dtype=tf.float32)

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

    def encoder_gru(self, X, n_layers):
        stack_cell = []
        for i in range(n_layers):
            with tf.variable_scope("encoder_%d"%i):
                cell = tf.contrib.rnn.GRUCell(self.n_hidden, activation=tf.nn.tanh)
                # cell = tf.contrib.rnn.AttentionCellWrapper(
                #     cell, attn_length=24, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)
                stack_cell.append(cell)

        stack = tf.contrib.rnn.MultiRNNCell(stack_cell, state_is_tuple=True)

        # The second output is the last state and we will not use that
        outputs, last_state = tf.nn.dynamic_rnn(stack, X, self.seq_len, dtype=tf.float32)

        return outputs, last_state

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x

        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.3)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.3)
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

    def encode(self, x, dim):
        h = self.enc(x, "encode", dim)
        z, z_mu, z_sigma = self.gen_z(h, "VAE")
        return z, z_mu, z_sigma

    def decode(self, x, dim):
        y = self.dec(x, "decode", dim)
        return y

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        return neg_ll


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


    def mlp(self, x):
        x_ = x
        with tf.variable_scope("mlp"):
            for i in range(len(self.layers)):
                x_ = layers.fully_connected(x_, self.layers[i], self.active_function, scope="mlp_%d" % i,weights_regularizer=self.regularizer)
        return x_


    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = -tf.nn.log_softmax(x_recon)

        neg_ll = tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))

        return neg_ll


    def prediction(self, x, y, cat=None, y_cat=None, reuse=False):
        with tf.variable_scope("last_layer", reuse=reuse):
            x_ = x
            # x_ = layers.fully_connected(x,50, self.active_function, scope="mlp",weights_regularizer=self.regularizer)
            out = layers.fully_connected(x_, self.n_products, tf.nn.tanh)
            # out = tf.nn.leaky_relu(out, alpha=0.2)
            # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out, 100))

            if cat !=None:
                pred_cat = layers.fully_connected(cat, self.cat_dim, tf.nn.tanh)
                pred_cat = tf.reshape(pred_cat, [-1, self.cat_dim, 1])
                out_cat = tf.matmul(tf.broadcast_to(self.item_cat, [tf.shape(cat)[0], self.item_cat.shape[0],self.item_cat.shape[1]]),  pred_cat)
                pred = out_cat * out

                loss = 10*self.loss_reconstruct(y, pred) + self.loss_reconstruct(y_cat, out_cat)
            else:
                loss = self.loss_reconstruct(y, out)

        return loss, out


    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.w_size, self.p_dim])
        self.X_cat = tf.placeholder(tf.float32, [None, self.cat_dim])
        self.y = tf.placeholder(tf.float32, [None, self.w_size, self.n_products])

        self.seq_len = tf.fill([tf.shape(self.X)[0]], self.w_size)
        outputs = self.X
        if self.model_type == 'bilstm':
            for i in range(self.n_layers):
                outputs, _ = self.encoder_BiLSTM(outputs,  str(i+1), self.n_hidden/(4**i))

            last_state = tf.reshape(outputs[:, -1, :],
                                    (tf.shape(self.X)[0], self.n_hidden*2/(4*(self.n_layers-1))))
        if self.model_type == 'bigru':
            for i in range(self.n_layers):
                outputs, _ = self.encoder_biGRU(outputs, str(i + 1), self.n_hidden / (4 ** i))

            last_state = tf.reshape(outputs[:, -1, :],
                                    (tf.shape(self.X)[0], self.n_hidden * 2 / (4 * (self.n_layers - 1))))
        elif self.model_type == 'lstm':
            outputs, _ = self.encoder_LSTM(self.X, self.n_layers)
            last_state = tf.reshape(outputs[:, -1, :],
                                   (tf.shape(self.X)[0], self.n_hidden * 2 / (2 * (self.n_layers - 1))))
        elif self.model_type == 'gru':
            outputs, _ = self.encoder_gru(self.X, self.n_layers)
            last_state = tf.reshape(outputs[:, -1, :],
                                    (tf.shape(self.X)[0], self.n_hidden * 2 / (2 * (self.n_layers - 1))))

        last_state_cat = self.mlp(self.X_cat)
        last_state_cat = tf.reshape(last_state_cat, (tf.shape(self.X)[0], self.layers[-1]))
        last_state = tf.concat([last_state, last_state_cat], axis=1)

        self.loss, self.predict = self.prediction(last_state, tf.reshape(self.y[:, -1, :], (-1, self.n_products)))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)



def main():
    args = parser.parse_args()
    batch_size = args.batch_size
    iter = args.iter
    dataset = args.data
    type = args.type
    # text = load_npz("data/%s/item.npz" % dataset).toarray()

    num_p = len(list(open("data/%s/item_id.txt"%dataset)))
    checkpoint_dir = "experiment/%s/" % (dataset)
    data = Dataset(num_p, "data/%s"%(dataset), args.w_size, cat=args.cat, time=args.time)
    # variables = load_npz("data/%s/item.npz"%dataset)
    # variables = np.append(variables.toarray(), [[0]*variables.toarray().shape[1]], axis=0)
    # data.item_cat = np.concatenate((data.item_cat, variables), axis=-1)

    data.create_user_info("data/%s"%dataset)


    model = Seq2seq(n_layers=args.n_layers, model_type=args.model_type)
    # model.p_dim = data.n_user
    model.w_size = args.w_size
    model.p_dim = data.n_user
    if args.cat:
        model.p_dim += data.item_cat.shape[1]
    if args.time:
        model.p_dim += data.time_dim
    # model.p_dim = data.n_user
    # model.cat_dim = text.shape[1]
    model.cat_dim = data.n_item
    model.n_products = data.n_item
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    max_recall = 0

    f = open("experiment/%s/result.txt"%dataset, "a")
    f.write("-------------------------\n")
    f.write("Data: %s - num_p: %d - user info\nbilstm: True - n_layers: 2 - w_size:%d - model_type: %s\n"
            %(dataset, data.n_item, data.w_size, model.model_type))
    f.write("cat: %s - time: %s\n" % (args.cat, args.time))
    result = [0,0,0,0]

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(data.n_user)
        train_cost = 0
        data.create_train_iter()

        for j in range(0, int(data.n_user / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            if args.time:
                X, y, u = data.create_batch_u(list_idx, data.X_iter, data.y_iter, data.train, data.time_emb)
            else:
                X, y, u = data.create_batch_u(list_idx, data.X_iter, data.y_iter, data.train)
            # u = np.concatenate((data.user_info_train[list_idx], u), axis=-1)

            feed = {model.X: X, model.y:y, model.X_cat:u}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        if i % 10 == 0:
            model.train = False
            for j in range(0, int(math.ceil(float(len(data.val)) / batch_size))):
                idx = list(range(j * batch_size, min((j + 1) * batch_size, len(data.val))))
                if args.time:
                    X_b_val, y_b, u = data.create_batch_u(idx, data.val, data.val_infer,
                                                        data.tmp_val,data.time_emb_val)
                else:
                    X_b_val, y_b, u = data.create_batch_u(idx, data.val, data.val_infer, data.tmp_val)
                # u = np.concatenate((data.user_info_val[idx], u), axis=-1)

                feed = {model.X: X_b_val, model.X_cat:u, model.y:y_b}
                loss_val, y_b_val = sess.run([model.loss, model.predict],feed_dict=feed)
                if j == 0:
                    p_val = y_b_val
                else:
                    p_val = np.concatenate((p_val, y_b_val), axis=0)

            recall, _, _ = calc_recall(p_val, data.val, data.val_infer)
            print("Loss val: %f, recall %f" % (loss_val, recall))
            if recall >= max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'bilstm-model'))

                for j in range(int(math.ceil(float(len(data.test))/batch_size))):
                    idx = list(range(j*batch_size, min((j+1)*batch_size, len(data.test))))
                    if args.time:
                        X_b_test, y_b, u = data.create_batch_u(idx, data.test,data.infer2,
                                                          data.tmp_test,
                                                          data.time_emb_test)
                    else:
                        X_b_test, y_b, u = data.create_batch_u(idx, data.test, data.infer2, data.tmp_test)
                    # t_b = np.concatenate((data.user_info_test[idx], u), axis=-1)
                    feed = {model.X: X_b_test, model.X_cat: u, model.y: y_b}
                    loss_val, y_b_val = sess.run([model.loss, model.predict],feed_dict=feed)
                    if j == 0:
                        y = y_b_val
                        y_val = y_b
                    else:
                        y = np.concatenate((y, y_b_val), axis=0)
                        y_val = np.concatenate((y_val, y_b), axis=0)
                recall_test, hit, ndcg = calc_recall(y, data.tmp_test, data.infer2)
                np.savez(os.path.join(checkpoint_dir, "pred"), p_val=y_val, p_test=y)
                print("iter: %d recall: %f, hit: %f, ndcg: %f" % (i, recall_test, hit, ndcg))
                if recall_test > result[1]:
                    result = [i, recall_test, hit, ndcg]
            model.train = True
        if i % 100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 10
            print("decrease lr to %f" % model.learning_rate)
    f.write("iter: %d - recall: %f - hit: %f - ndcg: %f\n"
            % (result[0], result[1], result[2], result[3]))
    f.write("Last result- recall: %f - hit: %f - ndcg:%f\n"%(recall_test, hit, ndcg))
    print(max_recall)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default="Tool",
                    help='dataset name')
parser.add_argument('--type', type=str, default="implicit",
                    help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')
parser.add_argument('--w_size', type=int, default=10, help='window size')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--cat', type=bool, default=False)
parser.add_argument('--time', type=bool, default=False)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--iter', type=int, default=150)
parser.add_argument('--model_type', type=str, default='bilstm')


if __name__ == '__main__':
    main()