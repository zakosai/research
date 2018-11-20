import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer
import numpy as np
import argparse
from scipy.sparse import load_npz
from keras.backend import binary_crossentropy

import os
class CCCFNET(object):
    def __init__(self, user_dim, item_dim):
        self.user_dim = user_dim
        self.item_A_dim = item_dim
        self.activation = tf.nn.relu
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
        self.item_A = tf.placeholder(tf.float32, [None, self.item_dim], name='item_input')
        self.rating_A = tf.placeholder(tf.float32, [None], name="rating")

        z_u = self.encode(self.user, self.layer, "user")
        z_A = self.encode(self.item_A, self.layer, "item")
        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(z_u), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(z_A), axis=1))


        predict = tf.reduce_sum(tf.multiply(z_u, z_A), axis=-1)/ (norm_item_output* norm_user_output)
        self.predict = tf.maximum(1e-6, predict)

        self.loss = tf.reduce_sum(binary_crossentropy(self.rating_A, self.predict), axis=1) + \
                    0.1 * tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.z_u = z_u
        self.z_A = z_A


def create_dataset(dataset="Health", type=1):
    dense_user = read_data("data2/%s/cf-train-%dp-users.dat" % (dataset, type))
    dense_item = read_data("data2/%s/cf-train-%dp-items.dat" % (dataset, type))

    num_p = len(dense_item)
    num_u = len(dense_user)
    train_user = one_hot_vector(dense_user, num_p)
    train_item = one_hot_vector(dense_item, num_u)

    dense_test = read_data("data2/%s/cf-test-%dp-users.dat" % (dataset, type))

    rating = []
    for user_id, user in enumerate(dense_user):
        neg = set(range(num_p)) - set(user)
        neg = np.random.permutation(list(neg))
        for j, item_id in enumerate(user):
            rating.append([user_id, item_id, 1])
            rating.append([user_id, neg[j], 0])

    return dense_user, dense_test, train_user, train_item, num_u, num_p, rating

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j!= num_product:
                one_hot_A[i,j] = 1
    return one_hot_A

def read_data(filename):
    arr = []
    for line in open(filename):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr

def calc_recall(pred, train, test, k=10, type=None):
    pred_ab = np.argsort(-pred)
    recall = []
    ndcg = []
    hit = 0
    for i in range(len(pred_ab)):
        p = pred_ab[i, :k+len(train[i])]
        p = p.tolist()
        for u in train[i]:
            if u in p:
                p.remove(u)
        p = p[:k]
        hits = set(test[i]) & set(p)

        #recall
        recall_val = float(len(hits)) / len(test[i])
        recall.append(recall_val)

        #hit
        hits_num = len(hits)
        if hits_num > 0:
            hit += 1

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

    # print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))


    return np.mean(np.array(recall)), float(hit)/len(pred_ab), np.mean(ndcg)

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
    dataset = args.dataset
    type = args.type

    dim = 600
    share = 200
    checkpoint_dir = "%s/%d/"%(dataset,type)
    dense_user, dense_test, train_user, train_item, num_u, num_p, rating = create_dataset(dataset, type)

    z_dim = 50




    model = CCCFNET(num_u, num_p)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0

    for i in range(0, iter):
        shuffle_idx = np.random.permutation(len(rating))
        train_cost = 0
        for j in range(int(len(rating) / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            u = train_user[rating[list_idx, 0]]
            i = train_item[rating[list_idx, 1]]
            r = rating[list_idx, 2]

            feed = {model.user: u,
                    model.item_A: i,
                    model.rating_A:r}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)
        print("Loss last batch: loss %f" % (loss))

        if i%10 == 0:
            # model.train = False
            # print("Loss last batch: loss %f" % (loss))

            z = sess.run([model.z_A], feed_dict={model.item_A:train_item})
            z_u = sess.run([model.z_u], feed_dict={model.user:train_user[:100]})
            # print(z_u_A.shape, z_u_B.shape, z_A.shape, z_B.shape)
            y_ = np.dot(z_u, z.T)
            y_ = y_.reshape((y_.shape[1], y_.shape[2]))
            recall, _, _ = calc_recall(y_, dense_user[:100], dense_test[:100])
            if recall > max_recall:

                saver.save(sess, os.path.join(checkpoint_dir, 'CCFNET-model'), i)
                z_u = sess.run([model.z_u], feed_dict={model.user: train_user})
                y_ = np.dot(z_u, z.T)

                recall, hit, ndcg = calc_recall(y_, dense_user, dense_test)
                print("Test: recall: %f, hit: %f, ndcg: %f"%(y_, dense_user, dense_test))



            model.train = True
        if i%100 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset',  type=str, default="Tool",
                   help='domain A')
parser.add_argument('--type',  type=int, default=1,
                   help='domain B')

if __name__ == '__main__':
    main()






