import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer
import numpy as np
import argparse
from scipy.sparse import load_npz
import os
class CCCFNET(object):
    def __init__(self, user_dim, item_A_dim, item_B_dim):
        self.user_dim = user_dim
        self.item_A_dim = item_A_dim
        self.item_B_dim = item_B_dim
        self.activation = tf.nn.tanh
        self.regular = l2_regularizer(scale=0.1)
        self.layer = [200, 50]
        self.learning_rate = 1e-4

    def encode(self, x, dim, scope):
        with tf.variable_scope(scope):
            for i in range(len(dim)):
                x = fully_connected(x, dim[i], self.activation, scope="%s_%d"%(scope, i), weights_regularizer=self.regular)
        return x

    def build_model(self):
        self.user = tf.placeholder(tf.float32, [None, self.user_dim], name='user_input')
        self.item_A = tf.placeholder(tf.float32, [None, self.item_A_dim], name='item_A_input')
        self.item_B = tf.placeholder(tf.float32, [None, self.item_B_dim], name='item_B_input')
        self.rating_A = tf.placeholder(tf.float32, [None], name="rating_A")
        self.rating_B = tf.placeholder(tf.float32, [None], name="rating_B")

        z_u = self.encode(self.user, self.layer, "user")
        z_A = self.encode(self.item_A, self.layer, "item_A")
        z_B = self.encode(self.item_B, self.layer, "item_B")

        predict_A = tf.reduce_sum(tf.multiply(z_u, z_A), axis=-1)
        predict_B = tf.reduce_sum(tf.multiply(z_u, z_B), axis=-1)

        self.loss = 0.5 * tf.reduce_sum((self.rating_A - predict_A)**2) + \
                    0.5 * tf.reduce_sum((self.rating_B - predict_B)**2) +  \
                    0.1 * tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.z_u = z_u
        self.z_A = z_A
        self.z_B = z_B


def create_dataset(A="Health", B="Clothing"):
    dense_A = read_data("data/%s_%s/%s_user_product.txt"%(A,B,A))
    num_A = 0
    for i in dense_A:
        if num_A < max(i):
            num_A = max(i)
    num_A += 1
    user_A = one_hot_vector(dense_A, num_A)

    dense_B = read_data("data/%s_%s/%s_user_product.txt"%(A, B, B))
    num_B = 0
    for i in dense_B:
        if num_B < max(i):
            num_B = max(i)
    num_B += 1
    user_B = one_hot_vector(dense_B, num_B)
    user = np.concatenate((user_A, user_B), axis=-1)
    item_A = user_A.T
    item_B = user_B.T
    variables = load_npz("data/%s_%s/mult_nor.npz"%(A,B))
    data = variables.toarray()
    item_A = np.concatenate((item_A, data[:num_A]), axis=-1)
    item_B = np.concatenate((item_B, data[num_A:]), axis=-1)

    for i in range(int(len(dense_A)*0.7)):
        p_A = dense_A[i]
        p_B = dense_B[i]
        if len(p_A) > len(p_B):
            max_len = len(p_A)
            while len(p_B) <len(p_A):
                p_B += p_B
        else:
            max_len = len(p_B)
            while len(p_A) < len(p_B):
                p_A += p_A

        u = [i]*max_len
        i_a = np.random.permutation(p_A)
        i_a = i_a[:max_len]
        i_b = np.random.permutation(p_B)
        i_b = i_b[:max_len]

        tr = np.column_stack((u, i_a, i_b))
        if i == 0:
            triple = np.array(tr)
        else:
            triple = np.concatenate((triple, tr))

    return user, dense_A, dense_B, num_A, num_B, triple, item_A, item_B

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j!= num_product:
                one_hot_A[i,j] = 1
    return one_hot_A

def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f

def calc_recall(pred, test, m=[100], type=None):
    for k in m:
        pred_ab = np.argsort(pred)[:,::-1][:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
            hits = set(test[i]) & set(p)

            #recall
            recall_val = float(len(hits)) / len(test[i])
            recall.append(recall_val)

            #ncdg
            score = []
            for j in range(k):
                if p[j] in hits:
                    score.append(1)
                else:
                    score.append(0)
            actual = dcg_score(score, pred[i, p], k)
            best = dcg_score(score, score, k)
            if best == 0:
                ndcg.append(0)
            else:
                ndcg.append(float(actual) / best)

        print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))


    return np.mean(np.array(recall))

def dcg_score(y_true, y_score, k=50):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def main():
    iter = 200
    batch_size= 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    if A == "Drama" or A=="Romance":
        k =[10, 20, 30, 40, 50]
        dim = 200
        share = 100
    else:
        k = [50, 100, 150, 200, 250, 300]
        dim = 600
        share = 200
    checkpoint_dir = "translation/%s_%s/"%(A,B)
    user, dense_A, dense_B, num_A, num_B, triple, item_A, item_B = create_dataset(A, B)
    encoding_dim_A = [dim]
    encoding_dim_B = [dim]
    share_dim = [share]
    decoding_dim_A = [dim, num_A]
    decoding_dim_B = [dim, num_B]
    z_dim = 50
    train_size = len(triple)
    val_position = int(len(user)*0.7)
    test_position = int(len(user)*0.75)
    test_size = len(user) - test_position
    val_size = len(user) - val_position -test_size



    model = CCCFNET(num_A+num_B, len(user)+8000, len(user)+8000)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0

    for i in range(0, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            u = user[triple[list_idx, 0]]
            i_A = item_A[triple[list_idx, 1]]
            i_B = item_B[triple[list_idx, 2]]
            r_A = [1]*batch_size
            r_B = [1]*batch_size

            feed = {model.user: u,
                    model.item_A: i_A,
                    model.item_B: i_B,
                    model.rating_A:r_A,
                    model.rating_B:r_B}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)
        print("Loss last batch: loss %f" % (loss))

        if i%10 == 0:
            # model.train = False
            # print("Loss last batch: loss %f" % (loss))

            u_A_val = np.concatenate((user[val_position:, :num_A], np.zeros((test_size+val_size, num_B))), axis=-1)
            u_B_val = np.concatenate((np.zeros((test_size+val_size, num_A)), user[val_position:, num_A:]), axis=-1)
            print(u_A_val.shape, u_B_val.shape)

            z_A, z_B = sess.run([model.z_A, model.z_B], feed_dict={model.item_A:item_A,
                                                                        model.item_B: item_B})
            z_u_A = sess.run([model.z_u], feed_dict={model.user:u_A_val})
            z_u_B = sess.run([model.z_u], feed_dict={model.user:u_B_val})
            # print(z_u_A.shape, z_u_B.shape, z_A.shape, z_B.shape)
            y_ab = np.dot(z_u_A, z_B.T)
            y_ba = np.dot(z_u_B, z_A.T)
            y_ab = y_ab.reshape((y_ab.shape[1], y_ab.shape[2]))
            y_ba = y_ba.reshape((y_ba.shape[1], y_ba.shape[2]))
            print(y_ab.shape, y_ba.shape)
            print(len(y_ba[:val_size]), len(dense_A[val_position:test_position]))

            recall = calc_recall(y_ba[:val_size], dense_A[val_position:test_position], [50]) + \
                    calc_recall(y_ab[:val_size], dense_B[val_position:test_position], [50])

            if recall > max_recall:
                max_recall = recall

                saver.save(sess, os.path.join(checkpoint_dir, 'CCFNET-model'), i)

                calc_recall(y_ba[val_size:], dense_A[test_position:], k, "A")
                calc_recall(y_ab[val_size:], dense_B[test_position:], k, "B")
                pred = np.argsort(-y_ba)[:, :10]
                f = open(os.path.join(checkpoint_dir, "predict_CCCFNET_%s.txt" % A), "w")
                for p in pred:
                    w = [str(i) for i in p]
                    f.write(','.join(w))
                    f.write("\n")
                f.close()
                pred = np.argsort(-y_ab)[:, :10]
                f = open(os.path.join(checkpoint_dir, "predict_CCCFNET_%s.txt" % B), "w")
                for p in pred:
                    w = [str(i) for i in p]
                    f.write(','.join(w))
                    f.write("\n")
                f.close()



            model.train = True
        if i%100 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k',  type=int, default=100,
                   help='top-K')

if __name__ == '__main__':
    main()






