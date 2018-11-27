import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse


class Translation:
    def __init__(self, batch_size, dim, encode_dim, decode_dim, z_dim, eps=1e-10,
                 lambda_0=10, lambda_1=0.1, lambda_2=100,
                 lambda_3=0.1,
                 lambda_4=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim = dim
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.z_dim = z_dim
        self.eps = eps
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.learning_rate = learning_rate
        self.active_function = tf.nn.tanh
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)


    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x

        if self.train:
            x_ = tf.nn.dropout(x_, 0.3)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
                x_ = batch_norm(x_, decay=0.995)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.3)
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

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.dim], name='input')

        x = self.x

        # VAE for domain A
        z, z_mu, z_sigma = self.encode(x, self.encode_dim)
        x_recon = self.decode(z, self.decode_dim)
        self.x_recon = x_recon

        # Loss VAE
        self.loss = self.lambda_1 * self.loss_kl(z_mu, z_sigma) + self.lambda_2 * self.loss_reconstruct(x,
                                                                                                        x_recon) + \
                    tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


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

    return user_A, user_B, dense_A, dense_B, num_A, num_B

def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f

def read_data2(filename):
    data = list(open(filename).readlines())
    data = data[1:]
    n_data = len(data)
    print(len(data))
    data = [d.strip() for d in data]
    data = [d.split(", ") for d in data]
    data = [d[:3] for d in data]
    data = np.array(data).reshape(n_data, 3).astype(np.int32)
    return data

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j!= num_product:
                one_hot_A[i,j] = 1
    return one_hot_A

def one_hot_vector2(A, num_product):
    one_hot = np.zeros((6557, num_product))
    for i in A:
        one_hot[i[0], i[1]] = i[2]
    return one_hot

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

def calc_rmse(pred, test):
    idx = np.where(test != 0)
    pred = pred[idx]
    test = test[idx]
    return np.sqrt(np.mean((test-pred)**2))

def main():
    iter = 3000
    batch_size= 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/" % (A, B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)

    print(num_A, num_B)
    if A == "Drama" or A=="Romance":
        k = [10, 20, 30, 40, 50]
        encoding_dim = [200, 100]
        decoding_dim = [100, 200, num_A + num_B]
    else:
        k = [50, 100, 150, 200, 250, 300]
        encoding_dim = [600, 200]
        decoding_dim = [200, 600, num_A + num_B]



    z_dim = 50


    perm = np.random.permutation(len(user_A))
    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    # user_A = user_A[perm]
    # user_B = user_B[perm]
    user_A = np.array(user_A)
    user_B = np.array(user_B)

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_val = user_A[train_size:train_size+val_size]
    user_B_val = user_B[train_size:train_size+val_size]
    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]

    user_train = np.concatenate((user_A_train, user_B_train), axis=1)
    user_val_A = np.concatenate((user_A_val, np.zeros(shape=user_B_val.shape)), axis=1)
    print(user_val_A.shape)
    user_val_B = np.concatenate((np.zeros(shape=user_A_val.shape), user_B_val), axis=1)
    user_test_A = np.concatenate((user_A_test, np.zeros(shape=user_B_test.shape)), axis=1)
    user_test_B = np.concatenate((np.zeros(shape=user_A_test.shape), user_B_test), axis=1)


    # dense_A_test = np.array(dense_A)[test_A]
    # dense_B_test = np.array(dense_B)[test_B]
    # test_A = [t - train_size - val_size for t in test_A]
    # test_B = [t - train_size - val_size for t in test_B]

    model = Translation(batch_size, num_A + num_B, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0
    dense_A_val = dense_A[train_size:train_size+val_size]
    dense_B_val = dense_B[train_size:train_size+val_size]

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x = user_train[list_idx]

            feed = {model.x: x}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)


        # print("Loss last batch: loss gen %f, loss dis %f, loss vae %f, loss gan %f, loss cc %f"%(loss_gen, loss_dis,
        #                                                                         loss_vae, loss_gan, loss_cc))

        # Validation Process
        if i%10 == 0:
            model.train = False
            loss_val_a, y_b = sess.run([model.loss, model.x_recon],
                                              feed_dict={model.x:user_val_A})
            loss_val_b, y_a = sess.run([model.loss, model.x_recon],
                                       feed_dict={model.x: user_val_B})
            print(len(y_a[0]), len(y_b[0]))
            recall = calc_recall(y_b[:, num_A:], dense_B_val, [50]) + calc_recall(y_a[:, :num_A], dense_A_val, [50])
            print("Loss val a: %f, Loss val b: %f, recall %f" % (loss_val_a, loss_val_b, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'multi-VAE-model'), i)
                loss_test_a, y_b= sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_A})
                loss_test_b, y_a = sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_B})
                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]
                calc_recall(y_a[:, :num_A], dense_A_test, k, "A")
                calc_recall(y_b[:, num_A:], dense_B_test, k, "B")
            model.train = True
        if i%100 == 0:
            model.learning_rate /= 2
            print("decrease lr to %f"%model.learning_rate)


    print(max_recall)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k', type=int, default=100, help='top-K')


if __name__ == '__main__':
    main()



