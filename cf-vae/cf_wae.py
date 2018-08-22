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
        self.learning_rate = 0.0001
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30
        self.weight_decay = 2e-4

class cf_vae_extend:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims, encoding_dims_str,
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
        self.encoding_dims_str = encoding_dims_str
        self.decoding_dims_str = decoding_dims_str
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
        print(self.params.EM_iter)
        if self.initial == False:
            self.load_model(model_mat)


    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.v_ = placeholder((None, self.num_factors))
        z_fake = placeholder((None, self.z_dim))

        # inference process
        with tf.variable_scope("text"):
            x = self.x_
            depth_inf = len(self.encoding_dims)

            # noisy_level = 1
            # x = x + noisy_level*tf.random_normal(tf.shape(x))

            with tf.variable_scope("encode"):
                for i in range(depth_inf):
                    x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)

                h_encode = x
                z_mu = dense(h_encode, self.z_dim, scope="mu_layer")
                z_log_sigma_sq = dense(h_encode, self.z_dim, scope = "sigma_layer")
                e = tf.random_normal(tf.shape(z_mu))
                z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

            # generative process
            y_true = self.decode(z)
            self.reconstructed = y_true

            y_fake = self.decode(z_fake, reuse=True)

            self.wae_lambda = 0.5
            self.loss_gan, self.penalty = self.gan_penalty(z_fake, z)
            self.loss_reconstruct = 0.2*tf.reduce_mean(tf.nn.l2_loss(self.x_- self.reconstructed))
            loss = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean(tf.reduce_sum(tf.square(self.v_ - z),1))
            self.wae_objective = self.loss_reconstruct + \
                                 self.wae_lambda * self.penalty + loss

        z_adv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/z_adversary')
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/encode')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/decode')
        ae_vars = encoder_vars + decoder_vars

        ae_opt = tf.train.AdadeltaOptimizer().minimize(loss=self.wae_objective,
                                   var_list=encoder_vars + decoder_vars)
        z_adv_opt = tf.train.AdadeltaOptimizer().minimize(
            loss=self.loss_gan[0], var_list=z_adv_vars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "cwae_%d.ckpt"%self.model)
        if self.initial:
            ckpt_file = os.path.join(self.ckpt_model, "wae_text.ckpt")
            text_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="text")
            text_saver = tf.train.Saver(var_list=text_varlist)
            # if init == True:
            text_saver.restore(self.sess, ckpt_file)


            self.initial = False
            self.saver = tf.train.Saver()

        else:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, ckpt)

        start = time.time()
        for i in range(self.params.num_iter):
            idx = np.random.choice(self.num_items, self.params.batch_size, replace=False)
            x_batch = x_data[idx]
            v_batch = self.V[idx]
            sample_noise = self.sample_pz('normal')
            _, l, lv = self.sess.run((ae_opt, self.wae_objective, loss),
                                 feed_dict={self.x_:x_batch, self.v_:v_batch, z_fake:sample_noise})
            _, lg = self.sess.run((z_adv_opt, self.loss_gan[0]),
                                 feed_dict={self.x_:x_batch, self.v_:v_batch, z_fake:sample_noise})

            if i % 50 == 0:
               print("epoches: %d\t loss: %f\t loss v:%f\t loss adv: %f\t time: %d s"%(i, l,lv, lg, time.time()-start))

        self.z_mu = z_mu
        self.x_recons = self.reconstructed
        self.saver.save(self.sess, ckpt)

        return None
    def sample_pz(self, distr):
        noise = None
        if distr == 'uniform':
            noise = np.random.uniform(
                -1, 1, [self.params.batch_size, self.z_dim]).astype(np.float32)
        elif distr in ('normal', 'sphere'):
            mean = np.zeros(self.z_dim)
            cov = np.identity(self.z_dim)
            noise = np.random.multivariate_normal(
                mean, cov, self.params.batch_size).astype(np.float32)
            if distr == 'sphere':
                noise = noise / np.sqrt(
                    np.sum(noise * noise, axis=1))[:, np.newaxis]
        return 0.5*noise

    def decode(self, z, reuse=False):
        with tf.variable_scope("decode", reuse=reuse):
            depth_gen = len(self.decoding_dims)
            y = z
            for i in range(depth_gen):
                y = dense(y, self.decoding_dims[i], scope="dec_layer" + "%s" % i, activation=tf.nn.sigmoid)
        return y

    def reconstruction_loss(self, real, reconstr):
        loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
        loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        return loss

    def gan_penalty(self, sample_qz, sample_pz):
        # Pz = Qz test based on GAN in the Z space
        logits_Pz = self.z_adversary(sample_pz)
        logits_Qz = self.z_adversary(sample_qz, reuse=True)
        loss_Pz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
        loss_Qz = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
        loss_Qz_trick = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
        loss_adversary = self.wae_lambda * (loss_Pz + loss_Qz)
        # Non-saturating loss trick
        loss_match = loss_Qz_trick
        return (loss_adversary, logits_Pz, logits_Qz), loss_match

    def z_adversary(self, inputs, reuse=False):
        num_units = 100
        num_layers = 1
        nowozin_trick = 0
        # No convolutions as GAN happens in the latent space
        with tf.variable_scope('z_adversary', reuse=reuse):
            hi = inputs
            for i in xrange(num_layers):
                hi = dense(hi, num_units, scope='hi_%d' % i)
                hi = tf.nn.sigmoid(hi)
            hi = dense(hi, 1, scope='hfinal_lin')
            # if nowozin_trick:
            #     # We are doing GAN between our model Qz and the true Pz.
            #     # Imagine we know analytical form of the true Pz.
            #     # The optimal discriminator for D_JS(Pz, Qz) is given by:
            #     # Dopt(x) = log dPz(x) - log dQz(x)
            #     # And we know exactly dPz(x). So add log dPz(x) explicitly
            #     # to the discriminator and let it learn only the remaining
            #     # dQz(x) term. This appeared in the AVB paper.
            #     assert opts['pz'] == 'normal', \
            #         'The GAN Pz trick is currently available only for Gaussian Pz'
            #     sigma2_p = float(opts['pz_scale']) ** 2
            #     normsq = tf.reduce_sum(tf.square(inputs), 1)
            #     hi = hi - normsq / 2. / sigma2_p \
            #          - 0.5 * tf.log(2. * np.pi) \
            #          - 0.5 * opts['zdim'] * np.log(sigma2_p)
        return hi

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

            for i in xrange(self.num_users):
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
            for j in xrange(self.num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m>0 :
                    A = np.copy(XX)
                    A += np.dot(self.U[user_ids,:].T, self.U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.z_dim) * params.lambda_v
                    x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * self.exp_z[j,:]

                    self.V[j, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * m * params.C_a
                    likelihood += params.C_a * np.sum(np.dot(self.U[user_ids, :], self.V[j,:][:, np.newaxis]),axis=0)
                    likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:])[:,np.newaxis])
                    ep = self.V[j,:] - self.exp_z[j,:]
                    likelihood += -0.5 * params.lambda_v * np.sum(ep*ep)
                else:
                    # m=0, this article has never been rated
                    A = np.copy(XX)
                    A += np.eye(self.z_dim) * params.lambda_v
                    x = params.lambda_v * self.exp_z[j, :]
                    ep = self.V[j, :] - self.exp_z[j, :]

                    self.V[j, :] = scipy.linalg.solve(A, x)

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

    def get_exp_hidden(self, x_data):
        self.exp_z = self.sess.run(self.z_mu, feed_dict={self.x_: x_data})
        return self.exp_z

    def fit(self, users, items, x_data, params, test_users, im_data=None, str_data=None, ):
        start = time.time()
        self.e_step(x_data)
        self.exp_z = self.get_exp_hidden(x_data)
        for i in range(params.EM_iter):
            print("iter: %d"%i)

            self.pmf_estimate(users, items, params)
            self.e_step(x_data)
            self.exp_z= self.get_exp_hidden(x_data)

            if i%100 == 90:
                file = open(os.path.join(self.ckpt_model, "result_type_0_%d.txt"%self.model), 'a')
                file.write("---------iter %d--------\n"%i)
                pred_all = self.predict_all()
                self.predict_val(pred_all, users, test_users, file)
                self.save_model(save_path_pmf=os.path.join(self.ckpt_model, "cf_vae_%d_%d.mat"%(self.model, i)))
                print(time.time() - start)
                file.close()
        print("time: %d"%(time.time()-start))
        return None

    def save_model(self, save_path_pmf):
        # self.saver.save(self.sess, save_path_weights)
        sio.savemat(save_path_pmf, {"U":self.U, "V":self.V, "Z":self.exp_z})
        print "all parameters saved"

    def load_model(self, load_path_pmf):
        # self.saver.restore(self.sess, load_path_weights)
        data = sio.loadmat(load_path_pmf)
        try:
            self.U = data["U"]
            self.V = data["V"]
            self.exp_z = data["Z"]
            print "model loaded"
        except:
            self.U = data["m_U"]
            self.V = data["m_V"]
            self.exp_z = data["m_theta"]
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
        for m in range(10, 10, 10):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            apk_vals = []
            for i in range(len(user_all)):
                train = train_users[i]
                top_M = list(np.argsort(-pred_all[i])[0:(m +len(train))])
                for u in train:
                    if u in top_M:
                        top_M.remove(u)
                top_M = top_M[:m]
                if len(top_M) != m:
                    print(top_M, train_users[i])
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
            mapk = np.mean(np.array(apk_vals))
            print recall_avg
            recall_avgs.append(recall_avg)
            # precision_avgs.append(precision_avg)
            mapk_avgs.append(mapk)

        return recall_avgs, mapk_avgs

    def predict_val(self, pred_all, train_users, test_users, file=None):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in [10]:
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            ndcg = []
            hit = 0
            for i in range(len(user_all)):
                train = train_users[i]
                top_M = list(np.argsort(-pred_all[i])[0:(m +len(train))])
                for u in train:
                    if u in top_M:
                        top_M.remove(u)
                top_M = top_M[:m]
                if len(top_M) != m:
                    print(top_M, train_users[i])
                hits = set(top_M) & set(user_all[i])   # item idex from 0
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
                if best ==0:
                    ndcg.append(0)
                else:
                    ndcg.append(float(actual)/best)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            #                        [u for u in user_all if len(u)!=0], m)

            print("recall %f, hit: %f, NDCG: %f"%(recall_avg, float(hit)/len(user_all), np.mean(ndcg)))
            #print recall_avg
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

    def predict_all(self):
        return np.dot(self.U, (self.V.T))
