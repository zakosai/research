import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
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

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.dim], name='input')

        x = self.x

        # VAE for domain A
        z, z_mu, z_sigma = self.encode(x, self.encode_dim)
        x_recon = self.decode(z, self.decode_dim)
        self.x_recon = x_recon

        # Loss VAE
        self.loss = self.lambda_1 * self.loss_kl(z_mu, z_sigma) + self.lambda_2 * self.loss_reconstruct(x,x_recon) + \
                    0.1* tf.losses.get_regularization_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


def create_dataset(dataset="ml-1m", num_p=7780):

    dense_train, dense_infer1 = read_data("data/%s/train.txt"%(dataset))
    train = one_hot_vector(dense_train, num_p)
    train_no = int(len(train)*0.9)
    val = train[train_no:]
    dense_val = dense_train[train_no:]
    dense_infer_val = dense_infer1[train_no:]

    train = train[:train_no]
    dense_train = dense_train[:train_no]
    dense_infer1 = dense_infer1[:train_no]

    dense_test, dense_infer2 = read_data("data/%s/test.txt"%(dataset))
    test = one_hot_vector(dense_test, num_p)

    data = {'train': train,
            'dense_train': dense_train,
            'dense_infer1': dense_infer1,
            'val': val,
            'dense_val': dense_val,
            'dense_infer_val': dense_infer_val,
            'test': test,
            'dense_test': dense_test,
            'dense_infer2': dense_infer2}

    print("Summary: \n# num train: %d\n# num val: %d\n# num test: %d"%(len(train), len(val), len(test)))

    return data

def read_data(filename):
    arr = []
    out = []
    for line in open(filename):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:-1]]
        arr.append(l)
        out.append([int(a[-1])])
    return arr, out

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

def calc_recall(pred, train, test, k=20, type=None):
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
        # print(test[i], p, hits)

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

def calc_rmse(pred, test):
    idx = np.where(test != 0)
    pred = pred[idx]
    test = test[idx]
    return np.sqrt(np.mean((test-pred)**2))

def main():
    args = parser.parse_args()
    iter = args.iter
    batch_size= args.batch_size
    dataset = args.data
    type = args.type
    num_p = len(list(open("data/%s/item_id.txt"%dataset)))
    print(num_p)
    checkpoint_dir = "experiment/%s" % (dataset)
    data = create_dataset(dataset, num_p)
    num_u = len(data['dense_train'])
    train = data['train']



    encoding_dim = [600]
    decoding_dim = [600, num_p]


    z_dim = 200

    model = Translation(batch_size, num_p, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    max_recall = 0

    f = open("experiment/%s/result.txt" % dataset, "a")
    f.write("-------------------------\n")
    f.write("Data: %s - Multi-VAE\n"%dataset)

    result =[0, 0, 0, 0]
    for i in range(1, iter):
        shuffle_idx = np.random.permutation(num_u)
        train_cost = 0
        for j in range(int(num_u/batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            if j == int(num_u/batch_size) -1:
                list_idx = shuffle_idx[j * batch_size:]
            x = train[list_idx]

            feed = {model.x: x}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)


        # print("Loss last batch: loss gen %f, loss dis %f, loss vae %f, loss gan %f, loss cc %f"%(loss_gen, loss_dis,
        #                                                                         loss_vae, loss_gan, loss_cc))

        # Validation Process
        if i%10 == 0:
            model.train = False
            for j in range(int(len(data['val'])/batch_size)+1):
                last_idx = min((j+1)*batch_size, len(data['val']))
                x_val = data['val'][j*batch_size:last_idx]
                loss_b_val, y_b_val = sess.run([model.loss, model.x_recon], feed_dict={model.x: x_val})
                if j == 0:
                    y_val = y_b_val
                else:
                    y_val = np.concatenate((y_val, y_b_val), axis=0)

            recall, _, _ = calc_recall(y_val, data['dense_val'], data['dense_infer_val'])
            print("Loss val: %f, recall %f" % (loss_b_val, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'multi-VAE-model'))


                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]
                for j in range(int(len(data['test']) / batch_size) + 1):
                    last_idx = min((j + 1) * batch_size, len(data['test']))
                    x_test = data['test'][j * batch_size:last_idx]
                    loss_b_test, y_b_test = sess.run([model.loss, model.x_recon], feed_dict={model.x: x_test})
                    if j == 0:
                        y_test = y_b_test
                    else:
                        y_test = np.concatenate((y_test, y_b_test), axis=0)

                recall, hit, ndcg = calc_recall(y_test, data['dense_test'], data['dense_infer2'])
                print("Loss test: %f, recall: %f, hit: %f, ndcg: %f" % (loss_b_test, recall, hit, ndcg))
                if recall > result[1]:
                    result = [i, recall, hit, ndcg]
            model.train = True
        if i%100 == 0 and model.learning_rate > 1e-6:
            model.learning_rate /= 2
            print("decrease lr to %f"%model.learning_rate)

    f.write("iter: %d - recall: %f - hit: %f - ndcg: %f\n" % (result[0], result[1], result[2], result[3]))
    f.write("Last result- recall: %d - hit: %f - ndcg:%f\n" % (recall, hit, ndcg))
    print(max_recall)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data',  type=str, default="Tool",
                   help='dataset name')
parser.add_argument('--type',  type=str, default="implicit",
                   help='1p or 8p')
parser.add_argument('--batch_size', type=int, default=500, help='number of batch_size')
parser.add_argument('--iter', type=int, default=1500, help='number of iter')


if __name__ == '__main__':
    main()



