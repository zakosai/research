import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm, maxout
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse
import pandas as pd
import pickle

class MultiTask:
    def __init__(self, dim_user, dim_item, dim_tag, encode_user, encode_item, encode_tag, decode_user, decode_item,
                 decode_tag, tag_pred_layer, rating_pred_layer, z_dim, share_dim, learning_rate=1e-4, eps=1e-10,
                 lambda_1=1, lambda_2=0.1, lambda_3=10, lambda_4=1):
        self.dim_user = dim_user
        self.dim_item = dim_item
        self.dim_tag = dim_tag
        self.encode_user = encode_user
        self.encode_item = encode_item
        self.decode_user = decode_user
        self.decode_item = decode_item
        self.encode_tag = encode_tag
        self.decode_tag = decode_tag
        self.tag_pred_layer = tag_pred_layer
        self.rating_pred_layer = rating_pred_layer
        self.learning_rate = learning_rate
        self.active_function = tf.nn.tanh
        self.z_dim = z_dim
        self.eps = eps
        self.share_dim = share_dim
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.train = True
        self.freeze = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)


    def enc(self, x, scope, layer, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(layer)):
                x_ = fully_connected(x_, layer[i], scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                # y = maxout(x_, encode_dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = self.active_function(x_)
        return x_

    def dec(self, x, scope, layer, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(layer)):
                x_ = fully_connected(x_, layer[i], scope="dec_%d" % i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = self.active_function(x_)
            # x_ = fully_connected(x_, layer[-1], scope="last_dec",
            #                  weights_regularizer=self.regularizer, trainable=self.freeze)
        return x_

    def adversal(self, x, scope, layer, reuse=False):
        x_ = x

        with tf.variable_scope(scope, reuse=reuse):
            if self.train:
                x_ = tf.nn.dropout(x_, 0.7)
            for i in range(len(layer)):
                x_ = fully_connected(x_, layer[i], scope="adv_%d" % i)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = self.active_function(x_)
            # x_ = fully_connected(x_, layer[-1], scope="adv_last")
        return x_

    def share_layer(self, x, scope, layer, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(layer)):
                x_ = fully_connected(x_, layer[i],  scope="share_%d"%i,
                                     weights_regularizer=self.regularizer)
                # y = maxout(x_, dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = self.active_function(x_)

        return x_

    def mlp(self, x, scope, layer, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(layer)):
                x_ = fully_connected(x_, layer[i], scope="%s_%d"%(scope, i), weights_regularizer=self.regularizer)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = self.active_function(x_)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.active_function, scope="z_mu", weights_regularizer=self.regularizer)
            z_sigma = fully_connected(h, self.z_dim,  self.active_function, scope="z_sigma",
                                      weights_regularizer=self.regularizer)
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def encode(self, x, scope1, scope2, layers, reuse_enc, reuse_share, reuse_z=False):
        h = self.enc(x, "encode_%s_%s"%(scope1, scope2), layers, reuse_enc)
        h = self.share_layer(h, "encode_%s"%scope1, self.share_dim, reuse_share)
        z, z_mu, z_sigma = self.gen_z(h, "encode_%s"%scope1, reuse=reuse_z)

        loss_kl = self.loss_kl(z_mu, z_sigma)
        return z, loss_kl

    def decode(self, x, scope1, scope2, layers, reuse_dec, reuse_share):
        x = self.share_layer(x, "decode_%s"%scope1, self.share_dim[::-1], reuse_share)
        x = self.dec(x, "decode_%s_%s"%(scope1, scope2), layers, reuse_dec)
        return x

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):
        # return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(x, x_recon), axis=1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        # return tf.reduce_mean(tf.reduce_sum((x-x_recon)**2, axis=1))
        # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x))

        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        return neg_ll


    def loss_recsys(self, pred, label):
        return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(label, pred), axis=1))

    def loss_discriminator(self, x, x_fake):
        loss_real = tf.reduce_mean(tf.squared_difference(x, 1))
        loss_fake = tf.reduce_mean(tf.squared_difference(x_fake, 0))
        return loss_real + loss_fake
    def loss_generator(self, x):
        return tf.reduce_mean(tf.squared_difference(x, 1))

    def build_model(self):
        self.user = tf.placeholder(tf.float32, shape=[None, self.dim_item], name='user_input')
        self.user_tag = tf.placeholder(tf.float32, shape=[None, self.dim_tag], name='user_tag_input')
        self.itempos = tf.placeholder(tf.float32, shape=[None, self.dim_user], name='item_pos_input')
        self.itempos_tag = tf.placeholder(tf.float32, shape=[None, self.dim_tag], name='item_pos_tag_input')
        self.itemneg = tf.placeholder(tf.float32, shape=[None, self.dim_user], name='item_neg_input')
        # self.itemneg_tag = tf.placeholder(tf.float32, shape=[None, self.dim_tag], name='item_neg_tag_input')
        self.tag = tf.placeholder(tf.float32, shape=[None, self.dim_tag], name='tag_input')

        z_user, loss_kl_user = self.encode(self.user, "user", "onehot",self.encode_user, False, False, False)
        user_rec = self.decode(z_user, "user", "onehot", self.decode_user, False, False)
        z_user_tag, loss_kl_user_tag = self.encode(self.user_tag, "user", "tag", self.encode_tag, False, True, True)
        user_tag_rec = self.decode(z_user_tag, "user", "tag", self.decode_tag, False, True)
        user_fake = self.decode(z_user_tag, "user", "onehot", self.decode_user, True, True)
        user_tag_fake = self.decode(z_user, "user", "tag", self.decode_tag, True, True)

        z_itempos, loss_kl_itempos = self.encode(self.itempos, "item", "onehot", self.encode_item, False, False, False)
        itempos_rec = self.decode(z_itempos, "item", "onehot", self.decode_item, False, False)
        z_itempos_tag, loss_kl_itempos_tag = self.encode(self.itempos_tag, "item", "tag", self.encode_tag, False,
                                                         True, True)
        itempos_tag_rec = self.decode(z_itempos_tag, "item", "tag", self.decode_tag, False, True)
        itempos_fake = self.decode(z_itempos_tag, "item", "onehot", self.decode_item, True, True)
        item_tag_fake = self.decode(z_itempos, "item", "tag", self.decode_tag, True, True)

        z_itemneg, loss_kl_itemneg = self.encode(self.itemneg, "item", "onehot", self.encode_item, True, True, True)
        itemneg_rec = self.decode(z_itemneg, "item", "onehot", self.decode_item, True, True)


        tag_concat = tf.concat([user_tag_rec, itempos_tag_rec], axis=1)
        tag_pred = self.mlp(tag_concat, "tag", self.tag_pred_layer)

        ratingpos_concat = tf.concat([z_user, z_itempos], axis=1)
        ratingpos_pred = self.adversal(ratingpos_concat, "rating", self.rating_pred_layer)

        ratingneg_concat = tf.concat([z_user, z_itemneg], axis=1)
        ratingneg_pred = self.adversal(ratingneg_concat, "rating", self.rating_pred_layer, True)


        #Loss Function #
        #Loss VAE
        loss_vae_user = self.lambda_1 * self.loss_reconstruct(self.user, user_rec) + self.lambda_2 * loss_kl_user
        loss_vae_itempos = self.lambda_1 * self.loss_reconstruct(self.itempos, itempos_rec) + self.lambda_2 * loss_kl_itempos
        loss_vae_itemneg = self.lambda_1 * self.loss_reconstruct(self.itemneg, itemneg_rec) + self.lambda_2 * loss_kl_itemneg
        loss_vae_user_tag = self.lambda_1 * self.loss_reconstruct(self.user_tag, user_tag_rec) + \
                            self.lambda_2 * loss_kl_user_tag
        loss_vae_itempos_tag = self.lambda_1 * self.loss_reconstruct(self.itempos_tag, itempos_tag_rec) + \
                               self.lambda_2 * loss_kl_itempos_tag

        #Loss tag pred
        loss_tag = self.lambda_3 * self.loss_reconstruct(self.tag, tag_pred)

        #Loss GAN
        loss_rating_dis = self.lambda_4 * self.loss_discriminator(ratingpos_pred, ratingneg_pred)

        self.loss_pretrained = loss_vae_user + loss_vae_user_tag + loss_vae_itempos + loss_vae_itempos_tag + \
                               loss_vae_itemneg +  0.01 * tf.losses.get_regularization_loss() + self.lambda_1 * \
                               (self.loss_reconstruct(self.user, user_fake) +
                                self.loss_reconstruct(self.user_tag, user_tag_fake) +
                                self.loss_reconstruct(self.itempos, itempos_fake) +
                                self.loss_reconstruct(self.itempos_tag, item_tag_fake)) + loss_tag

        self.loss_gen = loss_tag + self.lambda_4 * self.loss_generator(ratingpos_pred) + \
                        0.01 * tf.losses.get_regularization_loss()
        self.loss_dis = loss_rating_dis

        self.train_op_pretrained = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_pretrained)
        self.train_op_tag = tf.train.AdamOptimizer(1e-5).minimize(loss_tag)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_gen)
        self.train_op_dis = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_dis)

        self.user_rec = ratingpos_pred
        self.tag_pred = tag_pred


def create_dataset_lastfm():
    user_tags = pd.read_table("hetrec2011-lastfm-2k/user_taggedartists.dat")
    user_artist = pd.read_table("hetrec2011-lastfm-2k/user_artists.dat")
    user_neg = np.genfromtxt("hetrec2011-lastfm-2k/user_neg_100.txt", dtype=np.int32, delimiter=',')
    user_id = list(set(user_tags.userID))
    artist_id = list(set(user_tags.artistID))
    tag_id = list(set(user_tags.tagID))

    user_no = len(user_id)
    artist_no = len(artist_id)
    tag_no = len(tag_id)

    # Divide train test
    user_artist = user_artist.loc[(user_artist['userID'].isin(user_id)) & (user_artist['artistID'].isin(artist_id))]
    test = user_artist.sample(frac=0.2)
    train = user_artist.loc[~user_artist.index.isin(test.index)]

    # initial one hot
    user_onehot = np.zeros(shape=(user_no, artist_no), dtype=np.int8)
    artist_onehot = np.zeros(shape=(artist_no, user_no), dtype=np.int8)
    tag_user_onehot = np.zeros(shape=(user_no, tag_no), dtype=np.int8)
    tag_artist_onehot = np.zeros(shape=(artist_no, tag_no), dtype=np.int8)
    tag_label_train = np.zeros(shape=(train.shape[0], tag_no), dtype=np.int8)
    train_matrix = []
    test_matrix = []
    print("finish initial")

    # create train one hot
    index = 0
    for _, ua in train.iterrows():
        if ua.userID in user_id and ua.artistID in artist_id:
            uid = user_id.index(ua.userID)
            aid = artist_id.index(ua.artistID)
            user_onehot[uid, aid] = 1
            artist_onehot[aid, uid] = 1
            tag_list = list(user_tags.loc[(user_tags.userID == ua.userID) & (user_tags.artistID == ua.artistID)][
                                'tagID'])
            if len(tag_list) != 0:
                train_matrix.append([uid, aid])
                for i in tag_list:
                    tid = tag_id.index(i)
                    tag_user_onehot[uid, tid] = 1
                    tag_artist_onehot[aid, tid] = 1
                    tag_label_train[index, tid] = 1
        else:
            print(ua)
        index += 1

    print("finish create train")

    # create test
    user_artist_test = {}
    tag_test = []
    for index, ua in test.iterrows():
        if ua.userID in user_id and ua.artistID in artist_id:
            uid = user_id.index(ua.userID)
            aid = artist_id.index(ua.artistID)
            if uid not in user_artist_test:
                user_artist_test[uid] = [aid]
            else:
                user_artist_test[uid].append(aid)

            tag_list = list(user_tags.loc[(user_tags.userID == ua.userID) & (user_tags.artistID == ua.artistID)][
                                'tagID'])
            if len(tag_list) != 0:
                tag_list = [tag_id.index(t) for t in tag_list]
                tag_test.append(tag_list)
                test_matrix.append([uid, aid])
        else:
            print(ua)

    print("finish create test")

    train = np.array(train_matrix)
    test = np.array(test_matrix)

    print("finish matrix")

    dataset = {'user_no': user_no,
               'item_no': artist_no,
               'tag_no': tag_no,
               'user_onehot': user_onehot,
               'item_onehot': artist_onehot,
               'tag_user_onehot': tag_user_onehot,
               'tag_item_onehot': tag_artist_onehot,
               'tag_label_train': tag_label_train,
               'train': train,
               'test': test,
               'user_item_test': user_artist_test,
               'tag_test': tag_test,
               'user_neg': user_neg}
    print("finish dataset")
    f = open("hetrec2011-lastfm-2k/dataset.pkl", "wb")
    pickle.dump(dataset, f)


    return dataset

def calc_recall(pred, test, m=[100], type=None):

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
    # dataset = create_dataset_lastfm()
    f = open("hetrec2011-lastfm-2k/dataset.pkl", 'rb')
    dataset = pickle.load(f)
    print("finish create dataset")
    print(len(dataset['tag_label_train']),len(dataset['train']))


    encode_user = encode_item = encode_tag = [200]
    share_dim = [100]
    decode_user = [200, dataset['item_no']]
    decode_item = [200, dataset['user_no']]
    decode_tag = [200, dataset['tag_no']]

    batch_size = 500
    epoches = 3000


    model = MultiTask(dataset['user_no'], dataset['item_no'], dataset['tag_no'], encode_user, encode_item, encode_tag,
                      decode_user, decode_item, decode_tag, [dataset['tag_no']], [10, 1], 50, share_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0

    train = dataset['train']
    train_size = len(train)
    for i in range(1, epoches):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size / batch_size)):
            list_idx = shuffle_idx[j * batch_size:(j + 1) * batch_size]
            user = dataset['user_onehot'][train[list_idx, 0]]
            itempos = dataset['item_onehot'][train[list_idx, 1]]
            tag_user = dataset['tag_user_onehot'][train[list_idx, 0]]
            tag_itempos = dataset['tag_item_onehot'][train[list_idx, 1]]
            tag_label = dataset['tag_label_train'][list_idx]
            neg_idx = np.random.randint(0, 100, size=batch_size)
            itemneg = dataset['item_onehot'][dataset['user_neg'][train[list_idx,0], neg_idx]]

            feed = {model.user: user,
                    model.itempos: itempos,
                    model.itemneg: itemneg,
                    model.user_tag: tag_user,
                    model.itempos_tag: tag_itempos,
                    model.tag: tag_label}

            if i < 20:
                _, loss_pretrained = sess.run([model.train_op_pretrained, model.loss_pretrained], feed_dict=feed)
                loss_gen = loss_dis = 0
            else:
                _, loss_gen = sess.run([model.train_op_gen, model.loss_gen], feed_dict=feed)
                _, loss_dis = sess.run([model.train_op_dis, model.loss_dis], feed_dict=feed)

        if i % 10 == 0 and i > 20:
            model.train = False
            print("Loss lass batch: Loss gen %f, loss dis %f"%(loss_gen, loss_dis))

            # test
            user_id = dataset['user_item_test'].keys()
            print(len(dataset['user_item_test'].keys()))
            item_pred = []
            for u in dataset['user_item_test'].keys():
                user_id = [u] * dataset['item_no']
                pred = sess.run(model.user_rec, feed_dict={model.user: dataset['user_onehot'][user_id],
                                                                model.itempos:dataset['item_onehot']})
                pred = [item for sublist in pred for item in sublist]

                item_pred.append(pred)
            recall_item = calc_recall(item_pred, dataset['user_item_test'].values(), [50], "item")

            user = dataset['user_onehot'][dataset['test'][:,0]]
            itempos = dataset['item_onehot'][dataset['test'][:,1]]
            tag_user = dataset['tag_user_onehot'][dataset['test'][:,0]]
            tag_itempos = dataset['tag_item_onehot'][dataset['test'][:,1]]
            feed = {model.user: user,
                    model.itempos: itempos,
                    model.user_tag: tag_user,
                    model.itempos_tag: tag_itempos}

            tag_pred = sess.run(model.tag_pred, feed_dict=feed)
            recall_tag = calc_recall(tag_pred, dataset['tag_test'], [10], "tag")
            model.train = True

        if i % 100 ==0 and model.learning_rate > 1e-6:
            model.learning_rate /= 10
            print(model.learning_rate)




if __name__ == '__main__':
    main()




