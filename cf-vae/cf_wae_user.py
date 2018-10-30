import tensorflow as tf
import os
from tensorbayes.layers import dense, placeholder, conv2d, conv2d_transpose, max_pool
from tensorbayes.utils import progbar
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits
from keras.backend import binary_crossentropy
from keras.layers import merge, multiply
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm, maxout, l2_regularizer
import numpy as np
import time
from vae import vanilla_vae
import scipy
import scipy.io as sio

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
        self.learning_rate = 0.001
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30
        self.weight_decay = 2e-4

class cf_vae_extend:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims, encoding_dims_str,
                 decoding_dims_str, loss_type="cross_entropy", useTranse = False, eps = 1e-10, model=0, ckpt_folder='pre_model',
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
        self.encoding_dims_str = encoding_dims_str
        self.decoding_dims_str = decoding_dims_str
        self.loss_type = 'gan'
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
        if self.initial == False:
            self.load_model(model_mat)


        self.regularizer = l2_regularizer(scale=0.1)


    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data, im_data, str_data, u_data):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.v_ = placeholder((None, self.num_factors))
        self.x_u_ = placeholder((None, self.user_dim))  # we need these global nodes
        self.u_ = placeholder((None, self.num_factors))
        z_fake = placeholder((None, self.z_dim))
        z_u_fake = placeholder((None, self.z_dim))


        # inference process
        if self.model != 6 and self.model != 7:
            with tf.variable_scope("text"):
                x = self.x_
                depth_inf = len(self.encoding_dims)
                #x = tf.layers.dropout(x, rate=0.3)
                # noisy_level = 1
                # x = x + noisy_level*tf.random_normal(tf.shape(x))
                reg_loss = 0
                # attention = dense(x, self.input_dim, scope='att_layer', activation=tf.nn.sigmoid)
                # x = multiply([x, attention], name='attention_mul')
                def encode(x, reuse=False):
                    with tf.variable_scope("encode", reuse=reuse):
                        for i in range(depth_inf):
                            x = fully_connected(x, self.encoding_dims[i], tf.nn.sigmoid, scope="enc_layer" + "%s" % i,
                                                weights_regularizer=self.regularizer)

                        h_encode = x
                        z_mu = fully_connected(h_encode, self.z_dim, scope="mu_layer",
                                               weights_regularizer=self.regularizer)
                        z_log_sigma_sq = fully_connected(h_encode, self.z_dim, scope="sigma_layer",
                                                         weights_regularizer=self.regularizer)
                        e = tf.random_normal(tf.shape(z_mu))
                        z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e
                    return z, z_mu

                # generative process
                z, z_mu = encode(x)
                x_recons = self.decode(z, self.decoding_dims)
                self.wae_lambda = 0.5
                if self.loss_type == 'gan':
                    loss_gan_x, penalty_x = self.gan_penalty(z_fake, z)


        with tf.variable_scope("user"):
            encoding_dims = [100]
            decoding_dims = [100,self.user_dim]

            x_u = self.x_u_
            depth_inf = len(encoding_dims)

            def encode(x, reuse=False):
                with tf.variable_scope("encode", reuse=reuse):
                    for i in range(depth_inf):
                        x = fully_connected(x, encoding_dims[i], tf.nn.sigmoid, scope="enc_layer" + "%s" % i,
                                            weights_regularizer=self.regularizer)

                    h_encode = x
                    z_mu = fully_connected(h_encode, self.z_dim, scope="mu_layer", weights_regularizer=self.regularizer)
                    z_log_sigma_sq = fully_connected(h_encode, self.z_dim, scope="sigma_layer",
                                                     weights_regularizer=self.regularizer)
                    e = tf.random_normal(tf.shape(z_mu))
                    z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e
                return z, z_mu

            z_u, z_u_mu = encode(x_u)
            x_u_recons = self.decode(z_u, decoding_dims)


            if self.loss_type == 'gan':
                loss_gan_u, penalty_u = self.gan_penalty(z_u_fake, z_u)
                self.loss_gan = loss_gan_x + loss_gan_u
                self.penalty = penalty_x + penalty_u

            elif self.loss_type == 'mmd':
                self.penalty = self.mmd_penalty(z_fake, z)

        self.loss_reconstruct = self.reconstruction_loss(self.x_, x_recons) + self.reconstruction_loss(self.x_u_, x_u_recons)
        loss_x = 1.0 * self.params.lambda_v / self.params.lambda_r * tf.reduce_mean(tf.reduce_sum(tf.square(self.v_ -
                                                                                                          z), 1))
        loss_u = self.params.lambda_u / self.params.lambda_r * tf.reduce_mean(tf.reduce_sum(tf.square(self.u_ - z_u),
                                                                                            1))
        loss = loss_x + loss_u
        self.wae_objective = self.loss_reconstruct + \
                             self.wae_lambda * self.penalty + loss + 0.1 + tf.losses.get_regularization_loss()

        # encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/encode')
        # decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/decode')
        # ae_vars = encoder_vars + decoder_vars
        if self.loss_type == 'gan':
            z_adv_vars_text = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='text/z_adversary')
            z_adv_vars_user = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS, scope='user/z_adversary')
            z_adv_opt = tf.train.AdamOptimizer(self.params.learning_rate).minimize(
                loss=self.loss_gan[0], var_list=z_adv_vars_text + z_adv_vars_user)

        ae_opt = tf.train.AdamOptimizer(self.params.learning_rate).minimize(loss=self.wae_objective)

        # with tf.variable_scope("loss"):
        #     train_op_u = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step_u)
        #     train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "cwae_user_%d.ckpt"%self.model)
        if self.initial:
            if self.model != 6:
                ckpt_file = os.path.join(self.ckpt_model, "wae_text.ckpt")
                text_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="text")
                text_saver = tf.train.Saver(var_list=text_varlist)
                # if init == True:
                text_saver.restore(self.sess, ckpt_file)

            # LOAD IMAGE##
            if self.model == 1 or self.model == 2 or self.model == 6:
                ckpt_file_img = os.path.join(self.ckpt_model, "vae_image.ckpt")
                img_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="image")
                img_saver = tf.train.Saver(var_list=img_varlist)
                img_saver.restore(self.sess, ckpt_file_img)

            # Load Structure
            if self.model == 2 or self.model == 3:
                ckpt_file = os.path.join(self.ckpt_model, "vae_structure.ckpt")
                structure_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="structure")
                structure_saver = tf.train.Saver(var_list=structure_varlist)
                structure_saver.restore(self.sess, ckpt_file)

            ckpt_file = os.path.join(self.ckpt_model, "wae_user.ckpt")
            user_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="user")
            user_saver = tf.train.Saver(var_list=user_varlist)
            # if init == True:
            user_saver.restore(self.sess, ckpt_file)

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

            idu = np.random.choice(self.num_users, self.params.batch_size, replace=False)
            x_u_batch = u_data[idu]
            u_batch = self.U[idu]
            sample_noise = self.sample_pz('normal')
            sample_noise_u = self.sample_pz('normal')
            if self.model != 0:
                img_batch = im_data[idx]
                str_batch = str_data[idx]
            _, l, lv = self.sess.run((ae_opt, self.wae_objective, loss),
                                     feed_dict={self.x_: x_batch, self.v_: v_batch, z_fake: sample_noise,
                                                self.x_u_:x_u_batch, self.u_:u_batch, z_u_fake:sample_noise_u})

            if self.loss_type == 'gan':
                _, lg = self.sess.run((z_adv_opt, self.loss_gan[0]),
                                      feed_dict={self.x_: x_batch, self.v_: v_batch, z_fake: sample_noise,
                                                 self.x_u_: x_u_batch, self.u_: u_batch, z_u_fake: sample_noise_u})

            if i % 50 == 0:
                print("epoches: %d\t loss: %f\t loss v:%f\t time: %d s" % (i, l, lv, time.time() - start))
            #     _, lu = self.sess.run((train_op_u, self.loss_e_step_u),
            #                          feed_dict={self.x_:x_batch, self.v_:v_batch, self.x_s_:str_batch, self.x_im_:img_batch,
            #                                     self.x_u_: x_u_batch, self.u_:u_batch, })
            # else:
            #     _, _,l, lu = self.sess.run((train_op, train_op_u, self.loss_e_step, self.loss_e_step_u),
            #                          feed_dict={self.x_:x_batch, self.v_:v_batch,
            #                                     self.x_u_: x_u_batch, self.u_:u_batch})

        # for i in range(self.params.num_iter):
        #     idx = np.random.choice(self.num_users, self.params.batch_size, replace=False)
        #     x_u_batch = u_data[idx]
        #     u_batch = self.U[idx]
        #     _, l = self.sess.run((train_op_u, self.loss_e_step_u), feed_dict={self.x_u_: x_u_batch, self.u_:u_batch})
        #
        #     if i % 50 == 0:
        #        print("epoches: %d\t loss: %f\t time: %d s"%(i, l, time.time()-start))

        self.z_mu = z_mu
        self.x_recons = x_recons
        self.z_u_mu = z_u_mu
        self.x_u_recons = x_u_recons
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

    def decode(self, z, dim, reuse=False):
        with tf.variable_scope("decode", reuse=reuse):
            depth_gen = len(dim)
            y = z
            for i in range(depth_gen):
                y = fully_connected(y, dim[i], tf.nn.sigmoid, scope="dec_layer" + "%s" % i,
                                    weights_regularizer=self.regularizer)
        return y

    def reconstruction_loss(self, real, reconstr):
        # loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
        # loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        # return loss
        log_softmax_var = tf.nn.log_softmax(reconstr)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * real,
            axis=-1))
        # return tf.reduce_mean(tf.abs(x - x_recon))
        return neg_ll

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
                hi = fully_connected(hi, num_units, scope='hi_%d' % i, weights_regularizer=self.regularizer)
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
                    x = params.C_a * np.sum(self.V[item_ids, :], axis=0) + params.lambda_u * self.exp_z_u[i, :]
                    self.U[i, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * params.lambda_u * np.sum(self.U[i]*self.U[i])

            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.U[ids]
            XX = np.dot(u.T, u) * params.C_b
            for j in xrange(self.num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m > 0:
                    A = np.copy(XX)
                    A += np.dot(self.U[user_ids, :].T, self.U[user_ids, :]) * a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.z_dim) * params.lambda_v
                    if self.model == 1:
                        x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * (
                                    self.exp_z[j, :] + self.exp_z_im[j, :])
                    elif self.model != 6:
                        x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * self.exp_z[j, :]
                    else:
                        x = params.C_a * np.sum(self.U[user_ids, :], axis=0) + params.lambda_v * self.exp_z_im[j, :]

                    self.V[j, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * m * params.C_a
                    likelihood += params.C_a * np.sum(np.dot(self.U[user_ids, :], self.V[j,:][:, np.newaxis]),axis=0)
                    if self.model == 1:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:])[:,np.newaxis])
                        ep = self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:]
                    elif self.model == 2:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:] - self.exp_z_s[j,:])[:,np.newaxis])
                        ep = self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:] - self.exp_z_s
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
                    elif self.model == 2:
                         x = params.lambda_v * (self.exp_z[j,:] + self.exp_z_im[j,:] + self.exp_z_s[j, :])
                    elif self.model != 6:
                        x = params.lambda_v * self.exp_z[j,:]
                    else:
                        x = params.lambda_v * self.exp_z_im[j,:]
                    self.V[j, :] = scipy.linalg.solve(A, x)

                    if self.model == 1:
                        ep = self.V[j,:] - self.exp_z[j,:]- self.exp_z_im[j,:]
                    elif self.model == 2:
                         ep = self.V[j,:] - self.exp_z[j,:]- self.exp_z_im[j,:] - self.exp_z_s[j, :]
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


    def get_exp_hidden(self, x_data, im_data, str_data, u_data):
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

        self.exp_z_u = self.sess.run(self.z_u_mu, feed_dict={self.x_u_:u_data})
        return self.exp_z, self.exp_z_im, self.exp_z_s, self.exp_z_u

    def fit(self, users, items, x_data, params, test_users, u_data, im_data=None, str_data=None):
        start = time.time()
        self.e_step(x_data, im_data, str_data, u_data)
        self.exp_z, self.exp_z_im, self.exp_z_s, self.exp_z_u = self.get_exp_hidden(x_data, im_data, str_data, u_data)
        for i in range(params.EM_iter):
            print("iter: %d"%i)

            self.pmf_estimate(users, items, params)
            self.e_step(x_data, im_data, str_data, u_data)
            self.exp_z, self.exp_z_im, self.exp_z_s, self.exp_z_u = self.get_exp_hidden(x_data, im_data, str_data, u_data)

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
        sio.savemat(save_path_pmf, {"U":self.U, "V":self.V, "Z":self.exp_z, "Z_im":self.exp_z_im})
        print "all parameters saved"

    def load_model(self, load_path_pmf):
        # self.saver.restore(self.sess, load_path_weights)
        data = sio.loadmat(load_path_pmf)
        try:
            self.U = data["U"]
            self.V = data["V"]
            self.exp_z = data["Z"]
            self.exp_z_im = data["Z_im"]
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

    def predict_val(self, pred_all, train_users, test_users, file=None):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in range(10, 10, 10):
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
            if file != None:
                file.write("m = %d, recall = %f\t"%(m, recall_avg))
            # precision_avgs.append(precision_avg)
        return recall_avg


    def predict_all(self):
        return np.dot(self.U, (self.V.T))
