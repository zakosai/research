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
from tensorflow.contrib import slim


class params:
    def __init__(self):
        self.C_a = 1.0
        self.C_b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 1.0
        self.lambda_r = 1.0
        self.max_iter_m = 1

        # for updating W and b in vae
        self.learning_rate = 0.001
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30

class cf_vae_extend:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims,
                 decoding_dims_str, loss_type="cross_entropy", useTranse = False, eps = 1e-10, model=0, ckpt_folder='pre_model', initial=True, model_mat=None):
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
        self.decoding_dims_str = decoding_dims_str
        self.loss_type = loss_type
        self.useTranse = useTranse
        self.eps = eps
        self.initial = initial

        self.input_width = 64
        self.input_height = 64
        self.channel = 3
        self.num_conv = 4
        self.intermediate_dim = 256
        self.filter = 64
        self.model = model
        self.ckpt_model = ckpt_folder
        print(self.params.EM_iter)
        if self.initial == False:
            self.load_model(model_mat)


    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data, im_data, str_data):
        print "e_step finetuning"
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.v_ = placeholder((None, self.num_factors))
        scope = "text"
        tf.reset_default_graph()
        def encoder_func(x, eps):
            net = tf.concat([x, eps], axis=-1)
            for i in range(len(self.encoding_dims)):
                net = slim.fully_connected(net, self.encoding_dims[i], activation_fn=tf.nn.elu)

            z = slim.fully_connected(net, self.z_dim, activation_fn=None)
            return z


        def decoder_func(z):
            net = z
            for i in range(len(self.decoding_dims) -1):
                net = slim.fully_connected(net, self.decoding_dims[i], activation_fn=tf.nn.elu)

            xlogits = slim.fully_connected(net, self.decoding_dims[-1], activation_fn=None)
            return xlogits

        def discriminator_func(x, z):
            net = tf.concat([x, z], axis=1)
            net =  slim.fully_connected(net, 256, activation_fn=tf.nn.elu)
            for i in range(5):
                dnet = slim.fully_connected(net, 256, scope='fc_%d_r0' % (i+1))
                net += slim.fully_connected(dnet, 256, activation_fn=None, scope='fc_%d_r1' % (i+1),
                                            weights_initializer=tf.constant_initializer(0.))
                net = tf.nn.elu(net)

        #     net =  slim.fully_connected(net, 512, activation_fn=tf.nn.elu)
            net =  slim.fully_connected(net, 1, activation_fn=None)
            net = tf.squeeze(net, axis=1)
            net += tf.reduce_sum(tf.square(z), axis=1)

            return net

        # def create_scatter(x_test_labels, eps_test, savepath=None):
        #     plt.figure(figsize=(5,5), facecolor='w')
        #
        #     for i in range(4):
        #         z_out = sess.run(z_inferred, feed_dict={x_real_labels: x_test_labels[i], eps: eps_test})
        #         plt.scatter(z_out[:, 0], z_out[:, 1],  edgecolor='none', alpha=0.5)
        #
        #     plt.xlim(-3, 3); plt.ylim(-3.5, 3.5)
        #
        #     plt.axis('off')
        #     if savepath:
        #         plt.savefig(savepath, dpi=512)

        encoder = tf.make_template('encoder_%s'%scope, encoder_func)
        decoder = tf.make_template('decoder_%s'%scope, decoder_func)
        discriminator = tf.make_template('discriminator_%s'%scope, discriminator_func)

        # with tf.variable_scope(scope):

        eps = tf.random_normal([self.params.batch_size, self.input_dim])
        z_sampled = tf.random_normal([self.params.batch_size, self.z_dim])
        x_real = self.x_
        z_inferred = encoder(x_real, eps)
        x_reconstr_logits = decoder(z_inferred)

        Tjoint = discriminator(x_real, z_inferred)
        Tseperate = discriminator(x_real, z_sampled)

        reconstr_err = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x_real, logits=x_reconstr_logits),
            axis=1
        )

        loss_primal = tf.reduce_mean(reconstr_err + Tjoint)
        loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z_inferred), 1))
        self.loss_e_step = loss_primal + loss_v
        loss_dual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=Tjoint, labels=tf.ones_like(Tjoint))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=Tseperate, labels=tf.zeros_like(Tseperate))
        )

        optimizer_primal = tf.train.AdamOptimizer(2e-5)
        optimizer_dual = tf.train.AdamOptimizer(1e-4)

        qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder_%s"%scope)
        pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder_%s"%scope)
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_%s"%scope)

        train_op_primal = optimizer_primal.minimize(self.loss_e_step, var_list=pvars+qvars)
        train_op_dual = optimizer_dual.minimize(loss_dual, var_list=dvars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "cave_%d.ckpt"%self.model)
        saver = tf.train.Saver(var_list=qvars+pvars+dvars)
        if self.initial:

            text_ckpt = os.path.join(self.ckpt_model, "vae_%s.ckpt"%scope)
            saver.restore(self.sess, text_ckpt)
        else:
            saver.restore(self.sess, ckpt)

            # num_turn = x_input.shape[0] / self.batch_size
        start = time.time()
        for i in range(self.params.num_iter):
            idx = np.random.choice(self.num_items, self.params.batch_size, replace=False)
            x_batch = x_data[idx]
            v_batch = self.V[idx]

            self.sess.run([loss_primal, train_op_primal], feed_dict={self.x_:x_batch, self.v_:v_batch})
            g_loss, _ = self.sess.run([loss_primal, train_op_primal], feed_dict={self.x_:x_batch, self.v_:v_batch})
            g_loss, _ = self.sess.run([loss_primal, train_op_primal], feed_dict={self.x_:x_batch, self.v_:v_batch})
            d_loss, _ = self.sess.run([loss_dual, train_op_dual], feed_dict={self.x_:x_batch, self.v_:v_batch})
            if i % 50 == 0:
                print("epoches: %d\t g_loss: %f\t d_loss: %f\t time: %d s"%(i, g_loss, d_loss, time.time()-start))

        saver.save(self.sess, ckpt)
        self.z_mu = z_inferred
        self.x_recons = x_reconstr_logits

    def pmf_estimate(self, users, items, params):
        """
        users: list of list
        """
        min_iter = 1
        a_minus_b = params.C_a - params.C_b
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)
        it = 0
        while ((it < params.max_iter_m and converge > 1e-6) or it < min_iter):
            likelihood_old = likelihood
            likelihood = 0
            # update U
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0
            v = self.V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * params.C_b + np.eye(self.z_dim) * params.lambda_u

            for i in xrange(len(users)):
                item_ids = users[i]
                n = len(item_ids)
                if n > 0:
                    A = np.copy(XX)
                    A += np.dot(self.V[item_ids, :].T, self.V[item_ids,:])*a_minus_b
                    x = params.C_a * np.sum(self.V[item_ids, :], axis=0)
                    self.U[i, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * params.lambda_u * np.sum(self.U[i]*self.U[i])

            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.U[ids]
            XX = np.dot(u.T, u) * params.C_b
            for j in xrange(len(items)):
                user_ids = items[j]
                m = len(user_ids)
                if m>0 :
                    A = np.copy(XX)
                    A += np.dot(self.U[user_ids,:].T, self.U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.z_dim) * params.lambda_v
                    if self.model == 1:
                        x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * (self.exp_z[j,:] + self.exp_z_im[j,:])
                    elif self.model != 6:
                         x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * self.exp_z[j,:]
                    else:
                        x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * self.exp_z_im[j,:]
                    self.V[j, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * m * params.C_a
                    likelihood += params.C_a * np.sum(np.dot(self.U[user_ids, :], self.V[j,:][:, np.newaxis]),axis=0)
                    if self.model == 1:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:])[:,np.newaxis])
                        ep = self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:]
                    elif self.model != 6:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:])[:,np.newaxis])
                        ep = self.V[j,:] - self.exp_z[j,:]
                    else:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z_im[j,:])[:,np.newaxis])
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.z_dim) * params.lambda_v
                    if self.model == 1:
                        x = params.lambda_v * (self.exp_z[j,:] + self.exp_z_im[j,:])
                    elif self.model != 6:
                        x = params.lambda_v * self.exp_z[j,:]
                    else:
                        x = params.lambda_v * self.exp_z_im[j,:]
                    self.V[j, :] = scipy.linalg.solve(A, x)

                    if self.model == 1:
                        ep = self.V[j,:] - self.exp_z[j,:]- self.exp_z_im[j,:]
                    elif self.model != 6:
                        ep = self.V[j,:] - self.exp_z[j,:]
                    else:
                        ep = self.V[j,:] - self.exp_z_im[j,:]

                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
            # computing negative log likelihood
            #likelihood += -0.5 * params.lambda_u * np.sum(self.m_U * self.m_U)
            #likelihood += -0.5 * params.lambda_v * np.sum(self.m_V * self.m_V)
            # split R_ij into 0 and 1
            # -sum(0.5*C_ij*(R_ij - u_i^T * v_j)^2) = -sum_ij 1(R_ij=1) 0.5*C_ij +
            #  sum_ij 1(R_ij=1) C_ij*u_i^T * v_j - 0.5 * sum_j v_j^T * U C_i U^T * v_j

            it += 1
            converge = abs(1.0*(likelihood - likelihood_old)/likelihood_old)

            # if self.verbose:
            #     if likelihood < likelihood_old:
            #         print("likelihood is decreasing!")

            print("[iter=%04d], likelihood=%.5f, converge=%.10f" % (it, likelihood, converge))

        return likelihood

    def m_step(self, users, items, params):
        num_users = len(users)
        num_items = len(items)
        print("M-step")
        start =time.time()
        for i in range(params.max_iter_m):
            likelihood = 0

            for u in range(num_users):

                idx_a = np.ones(num_items) < 0
                idx_a[users[u]] = True   # pick those rated ids
                Lambda_inv = params.C_a * np.dot(self.V[idx_a].T, self.V[idx_a]) + \
                             params.C_b * np.dot(self.V[~idx_a].T, self.V[~idx_a]) + \
                             np.eye(self.num_factors) * params.lambda_u

                rx = params.C_a * np.sum(self.V[users[u], :], axis=0)
                self.U[u, :] = scipy.linalg.solve(Lambda_inv, rx, check_finite=True)

                likelihood += -0.5 * params.lambda_u * np.sum(self.U[u] * self.U[u])

            for v in range(num_items):
                idx_a = np.ones(num_users) < 0
                idx_a[items[v]] = True
                Lambda_inv = params.C_a * np.dot(self.U[idx_a].T, self.U[idx_a]) + \
                             params.C_b * np.dot(self.U[~idx_a].T, self.U[~idx_a]) + \
                             np.eye(self.num_factors) * params.lambda_v
                if self.model == 1:
                    rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * (self.exp_z[v, :] + self.exp_z_im[v, :])
                elif self.model != 6:
                    rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * self.exp_z[v, :]
                else:
                    rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * self.exp_z_im[v, :]
                self.V[v, :] = scipy.linalg.solve(Lambda_inv, rx, check_finite=True)

            print("iter: %d\t time:%d" %(i, time.time()-start))
        return None

    def get_exp_hidden(self, x_data, im_data, str_data):
        if self.model != 6:
            self.exp_z = self.sess.run(self.z_mu, feed_dict={self.x_: x_data})
        else:
            self.exp_z = 0
        if self.model == 1 or self.model == 2 or self.model == 6:
            for i in range(len(im_data), self.params.batch_size):
                im_batch = im_data[i:i+self.params.batch_size]
                exp_z_im = self.sess.run(self.z_im_mu, feed_dict={self.x_im_: im_batch})
                self.exp_z_im = np.concatenate((self.exp_z_im, exp_z_im), axis=0)
        else:
        # print(self.exp_z_im.shape)
             self.exp_z_im = 0

        if self.model == 2 or self.model == 3:
            self.exp_z_s = self.sess.run(self.z_s_mu, feed_dict={self.x_s_: str_data})
        else:
            self.exp_z_s = 0
        return self.exp_z, self.exp_z_im, self.exp_z_s

    def fit(self, users, items, x_data, im_data, str_data, params, test_users):
        start = time.time()
        file = open(os.path.join(self.ckpt_model, "result_%d.txt"%self.model), "w")
        self.e_step(x_data, im_data, str_data)
        self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)
        for i in range(params.EM_iter):
            print("iter: %d"%i)

            self.pmf_estimate(users, items, params)
            self.e_step(x_data, im_data, str_data)
            self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)

            if i%5 == 4:
                file.write("---------iter %d--------\n"%i)
                pred_all = self.predict_all(self.U)
                self.predict_val(pred_all, users, test_users, file)
                self.save_model(save_path_pmf=os.path.join(self.ckpt_model, "cf_vae_%d_%d.mat"%(self.model, i)))
                print(time.time() - start)
        file.close()
        print("time: %d"%(time.time()-start))
        return None

    def save_model(self, save_path_pmf):
        # self.saver.save(self.sess, save_path_weights)
        sio.savemat(save_path_pmf, {"U":self.U, "V":self.V, "Z":self.exp_z, "Z_im":self.exp_z_im})
        print "all parameters saved"

    def load_model(self, load_path_pmf):
        # self.saver.restore(self.sess, load_path_weights)
        data = sio.loadmat(load_path_pmf)
        self.U = data["U"]
        self.V = data["V"]
        self.exp_z = data["Z"]
        self.exp_z_im = data["Z_im"]
        print "model loaded"

    def predict(self, pred_all, train_users, test_users, M):
        # user_all = map(add, train_users, test_users)
        # user_all = np.array(user_all)    # item idex from 1
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in range(1, M, 1):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            apk_vals = []
            for i in range(len(user_all)):
                top_M = list(np.argsort(-pred_all[i])[0:(m +1)])
                if train_users[i] in top_M:
                    top_M.remove(train_users[i])
                else:
                    top_M = top_M[:-1]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                if len(train_users[i]) != 1:
                    print(i)
                hits = set(top_M) & set(user_all[i])   # item idex from 0
                hits_num = len(hits)
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)
                apk_vals.append( ml_metrics.apk(top_M, user_all[i], m))

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            # #                        [u for u in user_all if len(u)!=0], m)
            mapk = np.mean(np.array(apk_vals))
            print recall_avg
            recall_avgs.append(recall_avg)
            # precision_avgs.append(precision_avg)
            mapk_avgs.append(mapk)

        return recall_avgs, mapk_avgs

    def predict_val(self, pred_all, train_users, test_users, file):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in [5, 35]:
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            for i in range(len(user_all)):
                top_M = list(np.argsort(-pred_all[i])[0:(m +1)])
                if train_users[i] in top_M:
                    top_M.remove(train_users[i])
                else:
                    top_M = top_M[:-1]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                if len(train_users[i]) != 1:
                    print(i)
                hits = set(top_M) & set(user_all[i])   # item idex from 0
                hits_num = len(hits)
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            # #                        [u for u in user_all if len(u)!=0], m)
            print recall_avg
            file.write("m = %d, recall = %f"%(m, recall_avg))
            # precision_avgs.append(precision_avg)


    def predict_all(self, U):
        return np.dot(U, (self.V.T))
