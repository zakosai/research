import tensorflow as tf
import os
from tensorbayes.layers import dense, placeholder, conv2d, conv2d_transpose, max_pool
from tensorbayes.utils import progbar
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits
from keras.backend import binary_crossentropy
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras import metrics
from keras import backend as K
import numpy as np
import time
from vae import vanilla_vae
import scipy
import scipy.io as sio
from operator import add
from resnet_model import conv2d_fixed_padding, building_block, block_layer
import ml_metrics
import math
import tensorflow.contrib.layers as slim

class params:
    def __init__(self):
        self.C_a = 1.0
        self.C_b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 1.0
        self.lambda_r = 1.0
        self.max_iter_m = 1

        # for updating W and b in vae
        self.learning_rate = 1e-3
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30
        self.weight_decay = 2e-4

class neuVAE:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims, loss_type="cross_entropy", useTranse = False, eps = 1e-10, model=0, ckpt_folder='pre_model',
                 initial=True, model_mat=None, user_dim=9975):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.params = params

        self.U = 0.1 * np.random.randn(self.num_users, self.num_factors)
        self.V = 0.1 * np.random.randn(self.num_items, self.num_factors)
        self.exp_z = 0.1 * np.random.rand(self.num_items, self.num_factors)
        self.exp_z_im = 0.1 * np.random.rand(self.num_items, self.num_factors)

        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoding_dims = encoding_dims
        self.decoding_dims = decoding_dims
        self.loss_type = loss_type
        self.useTranse = useTranse
        self.eps = eps
        self.initial = initial

        self.input_width = 32
        self.input_height = 32
        self.channel = 3
        self.num_conv = 4
        self.intermediate_dim = 256
        self.filter = 64
        self.model = model
        self.ckpt_model = ckpt_folder
        self.user_dim = user_dim
        print(self.params.EM_iter)

    def e_step(self, x_data, u_data, data=None, train=True):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.x_u_ = placeholder((None, self.user_dim))  # we need these global nodes
        if train:
            self.rating_ = placeholder((None), dtype=tf.uint8)


        # inference process
        with tf.variable_scope("text"):
            x = self.x_
            # if train:
            #     x = tf.layers.dropout(x, rate=0.7)
            depth_inf = len(self.encoding_dims)
            for i in range(depth_inf):
                x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
            h_encode = x
            z_mu = slim.fully_connected(h_encode, self.z_dim, scope="mu_layer")
            z_log_sigma_sq = slim.fully_connected(h_encode, self.z_dim, scope="sigma_layer")
            e = tf.random_normal(tf.shape(z_mu))
            self.z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

            # generative process
            depth_gen = len(self.decoding_dims)
            y = self.z
            for i in range(depth_gen):
                y = dense(y, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.relu)
            x_recons = y

        with tf.variable_scope("user"):
            encoding_dims = [400]
            decoding_dims = [400,self.user_dim]
            x_u = self.x_u_
            # if train:
            #     x_u = tf.layers.dropout(x_u, rate=0.7)
            depth_inf = len(encoding_dims)
            for i in range(depth_inf):
                x_u = dense(x_u, encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.relu)
            h_u_encode = x_u
            z_u_mu = slim.fully_connected(h_u_encode, self.z_dim, scope="mu_layer")
            z_u_log_sigma_sq = slim.fully_connected(h_u_encode, self.z_dim, scope="sigma_layer")
            e_u = tf.random_normal(tf.shape(z_u_mu))
            self.z_u = z_u_mu + tf.sqrt(tf.maximum(tf.exp(z_u_log_sigma_sq), self.eps)) * e_u

            # generative process
            depth_gen = len(decoding_dims)
            y_u = self.z_u
            for i in range(depth_gen):
                y_u = dense(y_u, decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.relu)
            x_u_recons = y_u

        with tf.variable_scope("neuCF"):
            em = tf.concat([self.z, self.z_u], 1)
            layers = [100, 50]
            # if train:
            #     em = tf.layers.dropout(em, rate=0.7)

            for i in range(len(layers)):
                em = dense(em, layers[i], scope="neuCF_layer%s"%i, activation=tf.nn.relu)

            rating_ = dense(em, 2, scope="neuCF_lastlayer", activation=tf.nn.softmax)
            label = tf.one_hot(self.rating_, 2)
            print(label.shape)

            print(rating_.shape)


        if train:
            loss_u_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_u_, x_u_recons), axis=1))
            loss_u_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_u_mu) + tf.exp(z_u_log_sigma_sq)
                                                           - z_u_log_sigma_sq - 1, 1))
            loss_i_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_, x_recons), axis=1))
            loss_i_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq)
                                                           - z_log_sigma_sq - 1, 1))

            loss_rating = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(label, rating_), axis=1))
            self.loss = loss_i_recons + loss_u_recons + loss_i_kl + loss_u_kl
            #
            text_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="text")
            user_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="user")
            neuCF_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="neuCF")

            #
            train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss, var_list=text_varlist+user_varlist)
            train_op_rating = tf.train.AdamOptimizer(self.params.learning_rate).minimize(loss_rating, var_list=neuCF_varlist)
            # self.loss = loss_rating
            # train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "neuvae_%d.ckpt"%self.model)
        if self.initial:

            ckpt_file = os.path.join(self.ckpt_model, "vae_text.ckpt")
            text_saver = tf.train.Saver(var_list=text_varlist)
            # if init == True:
            text_saver.restore(self.sess, ckpt_file)

            ckpt_file = os.path.join(self.ckpt_model, "vae_user.ckpt")
            user_saver = tf.train.Saver(var_list=user_varlist)
            # if init == True:
            user_saver.restore(self.sess, ckpt_file)

            self.initial = False
            self.saver = tf.train.Saver()

        else:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, ckpt)

        if train:
            start = time.time()
            for i in range(self.params.num_iter):
                idx_list = np.random.permutation(len(data))
                for j in range(0, len(data) / self.params.batch_size + 1):
                    id = min((j+1)*self.params.batch_size, len(data))
                    idx = idx_list[(j*self.params.batch_size) : id]
                    x_batch = x_data[data[idx,1], :]
                    u_batch = u_data[data[idx, 0], :]
                    rating = np.array(data[idx,2]).astype(np.int8)

                    _, l = self.sess.run((train_op, self.loss),
                                         feed_dict={self.x_:x_batch, self.x_u_:u_batch, self.rating_: rating})

                    _, lr = self.sess.run((train_op_rating, loss_rating),
                                          feed_dict={self.x_: x_batch, self.x_u_: u_batch, self.rating_: rating})
                print("epoches: %d\t loss: %f\t time: %d s"%(i,l, time.time()-start))
                # if i%10 == 9:
                #     self.params.learning_rate /= 2

            self.saver.save(self.sess, ckpt)
        else:
            pred_all = []
            for id in range(self.num_users):
                users = (np.ones(self.num_items) * id).astype(np.int32)
                items = np.array(range(self.num_items))
                rat = []

                for i in range(0, int(self.num_items) / self.params.batch_size + 1):
                    idx = min(self.num_items, (i + 1) * self.params.batch_size)
                    u_batch = u_data[users[i * self.params.batch_size:idx]]
                    x_batch = x_data[items[i * self.params.batch_size:idx]]

                    r = self.sess.run(rating_, feed_dict={self.x_: x_batch, self.x_u_: u_batch})

                    rat += r[:,1].tolist()

                pred_all.append(rat)
            pred_all = np.array(pred_all).reshape(self.num_users, self.num_items)
            return pred_all
        return None

    def fit(self, data, x_data, u_data):
        start = time.time()
        self.e_step(x_data, u_data, data)
        print("time: %d"%(time.time()-start))
        return None

    def predict(self, train_users, test_users, x_data, u_data, M=10):

        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]
        pred_all = self.e_step(x_data, u_data, train=False)
        print(self.initial)
        print(pred_all.shape)

        for m in range(10, 11, 1):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            ndcg = []
            hit = 0
            apk_vals = []
            for i in range(self.num_users):
                train = train_users[i]
                top_M = list(np.argsort(-pred_all[i])[0:(m + len(train))])
                for u in train:
                    if u in top_M:
                        top_M.remove(u)
                top_M = top_M[:m]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                hits = set(top_M) & set(user_all[i])  # item idex from 0
                hits_num = len(hits)
                if hits_num > 0:
                    hit += 1
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                pred = np.array(pred_all[i])
                score = []
                for k in range(m):
                    if top_M[k] in hits:
                        score.append(1)
                    else:
                        score.append(0)
                actual = self.dcg_score(score, pred[top_M], m)
                best = self.dcg_score(score, score, m)
                if best == 0:
                    ndcg.append(0)
                else:
                    ndcg.append(float(actual) / best)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            #                        [u for u in user_all if len(u)!=0], m)

            print("recall %f, hit: %f, NDCG: %f" % (recall_avg, float(hit) / len(user_all), np.mean(ndcg)))
            if file != None:
                file.write("m = %d, recall = %f\t" % (m, recall_avg))

    def predict_val(self, pred_all, train_users, test_users, file=None):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in range(10, 100, 10):
            ndcg = []
            hit = 0
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            for i in range(len(user_all)):
                train = train_users[i]
                top_M = list(np.argsort(-pred_all[i])[0:(m +len(train))])
                for u in train:
                    if u in top_M:
                        top_M.remove(u)
                top_M = top_M[:m]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                hits = set(top_M) & set(user_all[i])  # item idex from 0
                hits_num = len(hits)
                if hits_num > 0:
                    hit += 1
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                pred = np.array(pred_all[i])
                score = []
                for k in range(m):
                    if top_M[k] in hits:
                        score.append(1)
                    else:
                        score.append(0)
                actual = self.dcg_score(score, pred[top_M], m)
                best = self.dcg_score(score, score, m)
                if best == 0:
                    ndcg.append(0)
                else:
                    ndcg.append(float(actual) / best)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            #                        [u for u in user_all if len(u)!=0], m)

            print("recall %f, hit: %f, NDCG: %f" % (recall_avg, float(hit) / len(user_all), np.mean(ndcg)))
            if file != None:
                file.write("m = %d, recall = %f\t"%(m, recall_avg))
            # precision_avgs.append(precision_avg)

    def dcg_score(self, y_true, y_score, k=5):
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