import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
import os
import math
import argparse
from dataset import calc_recall


class Data(object):
    def __init__(self, w_size, folder):
        self.w_size = w_size
        self.n_item = len(list(open("%s/item_id.txt" % folder)))

        self.train, self.infer1 = self.read_file("%s/train.txt" % folder)
        self.tmp_test, self.infer2 = self.read_file("%s/test.txt" % folder)
        self.train = self.train + self.tmp_test
        self.n_user = len(self.train)
        self.max_no = max([len(i) for i in self.train])

    def read_file(self, filename):
        train = []
        infer = []
        for line in open(filename):
            a = line.strip().split()
            if a == []:
                l = []
            else:
                l = [int(x) for x in a[1:-1]]
            # if len(l) < self.w_size:
            #     l = [self.n_item] * (self.w_size - len(l)) + l
            train.append(l)
            infer.append([int(a[-1])])
        return train, infer

    def get_batch_train(self, ids):
        train = np.ones((len(ids), self.w_size)) * self.n_item
        next_item = []

        for i in ids:
            train_len_i = len(train[i])
            r = np.random.randint(0, len(train_len_i) - self.w_size - 1) if train_len_i > self.w_size else 0
            train[i, max(-train_len_i, -self.w_size):] = train[i][r:-1]
            next_item.append(train[i][-1])

        return train.astype(np.int32), next_item

    def get_batch_test(self, ids, data, label):
        test = np.ones((len(ids), self.w_size)) * self.n_item
        next_item = [label[i] for i in ids]

        for i in ids:
            test[i, max(-len(data[i]), -self.w_size):] = data[i]

        return test.astype(np.int32), next_item


class Model(object):
    def __init__(self, user_no, user_embedding_size, item_no, item_embedding_size, w_size, n_layers):

        # Hyper-parameters for dataset
        self.user_no = user_no
        self.user_embedding_size = user_embedding_size
        self.item_no = item_no
        self.item_embedding_size = item_embedding_size
        self.w_size = w_size

        # Hyper-paramter for network
        self.n_hidden = 16
        self.layers = [100]
        self.n_layers = n_layers
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.active_function = tf.nn.tanh
        self.learning_rate = 1e-3

    def encoder_bigru(self, X, scope, n_hidden):
        with tf.variable_scope("cell_def_%s" % scope):
            f_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=tf.nn.tanh)
            f_cell = tf.contrib.rnn.DropoutWrapper(cell=f_cell, output_keep_prob=0.8)
            b_cell = tf.nn.rnn_cell.GRUCell(n_hidden, activation=tf.nn.tanh)
            b_cell = tf.contrib.rnn.DropoutWrapper(cell=b_cell, output_keep_prob=0.8)
        with tf.variable_scope("cell_op_%s" % scope):
            outputs1, last_state = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, X,
                                                                   sequence_length=self.seq_len,
                                                                   dtype=tf.float32)
        outputs = tf.concat(outputs1, 2)
        return outputs, last_state

    def mlp(self, x):
        x_ = x
        with tf.variable_scope("mlp"):
            for i in range(len(self.layers)):
                x_ = layers.fully_connected(x_, self.layers[i], self.active_function,
                                            scope="mlp_%d" % i,weights_regularizer=self.regularizer)
        return x_

    def prediction(self, x, y, reuse=False):
        with tf.variable_scope("last_layer", reuse=reuse):
            x_ = x
            out = layers.fully_connected(x_, self.item_no, tf.nn.tanh)
            # out = tf.nn.leaky_relu(out, alpha=0.2)
            # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y, out, 100))
            loss = self.loss_reconstruct(y, out)

        return loss, out

    def loss_reconstruct(self, x, x_recon):
        log_softmax_var = tf.nn.log_softmax(x_recon)
        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        return neg_ll

    def build_model(self):
        self.user_ids = tf.placeholder(tf.int32, [None])
        self.item_ids = tf.placeholder(tf.int32, [None, self.w_size])
        self.next_item = tf.placeholder(tf.int32, [None])
        self.seq_len = tf.fill([tf.shape(self.item_ids)[0]], self.w_size)

        # Get embedding
        user_embeddings = tf.get_variable("user_embeddings", [self.user_no, self.user_embedding_size])
        embedded_user_ids = tf.nn.embedding_lookup(user_embeddings, self.user_ids)
        item_embeddings = tf.get_variable("item_embeddings", [self.item_no, self.item_embedding_size])
        embedded_item_ids = tf.nn.embedding_lookup(item_embeddings, self.item_ids)

        # Bi-GRU for local
        n_hidden = self.n_hidden * 2
        outputs = embedded_item_ids
        for i in range(self.n_layers):
            n_hidden /= 2
            outputs, _ = self.encoder_bigru(outputs, str(i + 1), n_hidden)
        last_state = tf.reshape(outputs[:, -1, :], (tf.shape(self.item_ids)[0], n_hidden * 2))

        # MLP for global
        last_state_global = self.mlp(embedded_user_ids)
        last_state_global = tf.reshape(last_state_global, (tf.shape(embedded_user_ids)[0], self.layers[-1]))
        last_state = tf.concat([last_state, last_state_global], axis=1)

        self.loss, self.predict = self.prediction(last_state, tf.one_hot(self.next_item, self.item_no))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def main(args):
    batch_size = args.batch_size
    iter = args.iter
    dataset = args.data
    w_size = args.w_size
    data = Data(w_size, dataset)

    checkpoint_dir = "experiment/%s/" % dataset
    model = Model(user_no=data.n_user,
                  user_embedding_size=200,
                  item_no=data.n_item,
                  item_embedding_size=50,
                  w_size=w_size,
                  n_layers=args.n_layers)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    max_recall = 0

    f = open("experiment/%s/result.txt" % dataset, "a")
    f.write("-------------------------\n")
    f.write("Data: %s - num_p: %d - user info\n"
            "bilstm: True - n_layers: %d - w_size:%d - model_type: %s\n" % (
            dataset, data.n_item, model.n_layers, data.w_size, "embedding"))
    result = [0, 0, 0, 0]
    shuffle_idx = np.random.permutation(data.n_user)
    val_ids = shuffle_idx[:min(batch_size, data.n_user)]
    for i in range(1, iter):
        shuffle_idx = np.random.permutation(data.n_user)
        for j in range(0, int(data.n_user / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            item_ids, next_item = data.get_batch_train(list_idx)
            feed = {model.user_ids: list_idx,
                    model.item_ids: item_ids,
                    model.next_item: next_item}
            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        if i % 10 == 0:
            model.train = False
            item_ids, next_item = data.get_batch_test(val_ids, data.train, data.infer1)
            feed = {model.user_ids: val_ids,
                    model.item_ids: item_ids,
                    model.next_item: next_item}
            loss_val, y_val = sess.run([model.loss, model.predict], feed_dict=feed)

            recall, _, _ = calc_recall(y_val, data.train, next_item, ids=val_ids)
            print("Loss val: %f, recall %f" % (loss_val, recall))
            if recall >= max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'bilstm-model'))
                for j in range(int(math.ceil(float(len(data.tmp_test))/batch_size))):
                    idx = list(range(j*batch_size, min((j+1)*batch_size, len(data.tmp_test))))
                    item_ids, next_item = data.get_batch_test(idx, data.tmp_test, data.infer2)
                    feed = {model.user_ids: idx,
                            model.item_ids: item_ids,
                            model.next_item: next_item}
                    loss_val, y_b_val = sess.run([model.loss, model.predict], feed_dict=feed)
                    if j == 0:
                        y = y_b_val
                        y_val = next_item
                    else:
                        y = np.concatenate((y, y_b_val), axis=0)
                        y_val = np.concatenate((y_val, next_item), axis=0)
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
    f.write("Last result- recall: %f - hit: %f - ndcg:%f\n" % (recall_test, hit, ndcg))
    print(max_recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', type=str, default="Tool", help='dataset name')
    # parser.add_argument('--type', type=str, default="implicit", help='1p or 8p')
    # parser.add_argument('--num_p', type=int, default=7780, help='number of product')
    parser.add_argument('--w_size', type=int, default=10, help='window size')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--cat', type=bool, default=False)
    parser.add_argument('--time', type=bool, default=False)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--iter', type=int, default=150)
    parser.add_argument('--model_type', type=str, default='bilstm')
    args = parser.parse_args()
    print(args.cat, args.time)
    main(args)




