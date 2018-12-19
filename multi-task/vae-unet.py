import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse
import pandas as pd
import pickle

class Translation:
    def __init__(self, batch_size, dim, tag_dim, encode_dim, decode_dim, z_dim, eps=1e-10,
                 lambda_0=10, lambda_1=0.1, lambda_2=100,
                 lambda_3=0.1,
                 lambda_4=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim = dim
        self.tag_dim = tag_dim
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
        en_out = []

        if self.train:
            x_ = tf.nn.dropout(x_, 0.5)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer)
                x_ = batch_norm(x_, decay=0.995)
                en_out.append(x_)
        return x_, en_out

    def dec(self, x, scope, decode_dim,reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.5)
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
        h, en_out = self.enc(x, "encode", dim)
        z, z_mu, z_sigma = self.gen_z(h, "VAE")
        loss_kl = self.loss_kl(z_mu, z_sigma)
        y = self.dec(h, "decode", self.decode_dim)
        return y, loss_kl

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
        return neg_ll

        # return tf.reduce_mean(tf.abs(x - x_recon))

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.tag_dim*47], name='input')
        self.y = tf.placeholder(tf.float32, [None, self.dim], name='label')


        x = tf.concat([self.x, self.y], axis=1)
        # x = self.x
        # VAE for domain A
        x_recon, loss_kl = self.encode(x, self.encode_dim)
        self.x_recon = x_recon

        # Loss VAE
        self.loss = self.lambda_2 * self.loss_reconstruct(self.y,x_recon) + \
                    0.1 *tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def create_dataset_lastfm():
    user_tags = pd.read_table("hetrec2011-lastfm-2k/user_taggedartists.dat")
    user_artist = pd.read_table("hetrec2011-lastfm-2k/user_artists.dat")
    user_id = list(set(user_tags.userID))
    artist_id = list(set(user_tags.artistID))

    user_no = len(user_id)
    artist_no = len(artist_id)

    # Divide train test
    user_artist = user_artist.loc[(user_artist['userID'].isin(user_id)) & (user_artist['artistID'].isin(artist_id))]
    test = user_artist.sample(frac=0.3)
    train = user_artist.loc[~user_artist.index.isin(test.index)]

    # initial one hot
    user_onehot = np.zeros(shape=(user_no, artist_no), dtype=np.float32)
    train_matrix = []
    test_matrix = []
    print("finish initial")

    # create train one hot
    for _, ua in train.iterrows():
        if ua.userID in user_id and ua.artistID in artist_id:
            uid = user_id.index(ua.userID)
            aid = artist_id.index(ua.artistID)
            user_onehot[uid, aid] = 1

        else:
            print(ua)

    print("finish create train")

    # create test
    user_artist_test = {}
    for index, ua in test.iterrows():
        if ua.userID in user_id and ua.artistID in artist_id:
            uid = user_id.index(ua.userID)
            aid = artist_id.index(ua.artistID)
            if uid not in user_artist_test:
                user_artist_test[uid] = [aid]
            else:
                user_artist_test[uid].append(aid)

        else:
            print(ua)

    print("finish create test")

    train = np.array(train_matrix)

    print("finish matrix")

    dataset = {'user_no': user_no,
               'item_no': artist_no,
               'user_onehot': user_onehot,
               'train': train,
               'user_item_test': user_artist_test}
    print("finish dataset")

    return dataset

def calc_recall(pred, test, m=[100], type=None):
    result = {}
    for k in m:
        pred_ab = np.argsort(-pred)[:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
            if len(test[i]) != 0:
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
        result['recall@%d'%k] = np.mean(recall)
        result['ndcg@%d'%k] = np.mean(ndcg)


    return np.mean(np.array(recall)), result

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
    f = open("hetrec2011-lastfm-2k/dataset.pkl", 'rb')
    dataset = pickle.load(f)
    content = np.load("hetrec2011-lastfm-2k/exp1/item.npz")
    content = content['rec']

    num_p = dataset['item_no']
    num_u = dataset['user_no']
    num_t = dataset['tag_no']
    encoding_dim = [600, 200]
    decoding_dim = [200, 600, num_p]

    z_dim = 50

    user_item = np.zeros((num_u, num_u*47))
    for i in range(num_u):
        idx = np.where(dataset['user_onehot'][i] == 1)
        u_c = content[idx]
        u_c = u_c.flatten()
        user_item[i, :len(u_c)] = u_c




    model = Translation(batch_size, num_p,num_u, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(num_u)
        train_cost = 0
        for j in range(int(num_u/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            y = dataset['user_onehot'][list_idx]
            x = user_item[list_idx]

            feed = {model.x: x, model.y:y}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)


        # print("Loss last batch: loss gen %f, loss dis %f, loss vae %f, loss gan %f, loss cc %f"%(loss_gen, loss_dis,
        #                                                                         loss_vae, loss_gan, loss_cc))

        # Validation Process
        if i%10 == 0:
            model.train = True
            x = user_item[dataset['user_item_test'].keys()]
            y = dataset['user_onehot'][dataset['user_item_test'].keys()]
            item_pred = sess.run(model.x_recon,
                                              feed_dict={model.x:x, model.y:y})

            recall_item, _ = calc_recall(item_pred, dataset['user_item_test'].values(), [50], "item")
            model.train = True

            if recall_item > max_recall:
                max_recall = recall_item
                if i > 1000:
                    if max_recall < 0.1:
                        _, result = calc_recall(item_pred, dataset['user_item_test'].values(), [10, 20, 30, 40, 50],
                                                "item")
                    else:
                        _, result = calc_recall(item_pred, dataset['user_item_test'].values(),
                                                [50, 100, 150, 200, 250, 300], "item")


        if i%100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)


    print(max_recall)
    np.save(args.data.split(".")[0] + "_result.npy", result)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data',  type=str, default="Tool",
                   help='dataset name')
parser.add_argument('--type',  type=int, default=1,
                   help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')


if __name__ == '__main__':
    main()



