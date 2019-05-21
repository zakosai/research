__author__ = 'linh'
import numpy as np
import tensorflow as tf
import scipy.io
from scipy.sparse import load_npz
import  argparse
import os
import pickle
from tensorbayes.layers import dense, placeholder, conv2d, conv2d_transpose, max_pool
from keras.backend import binary_crossentropy
import time
import scipy

import math
import tensorflow.contrib.layers as slim

np.random.seed(0)
tf.set_random_seed(0)


class params_class:
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
        if self.initial == False:
            self.load_model(model_mat)


    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data, im_data, str_data, u_data):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.v_ = placeholder((None, self.num_factors))
        self.x_u_ = placeholder((None, self.user_dim))  # we need these global nodes
        self.u_ = placeholder((None, self.num_factors))


        # inference process
        if self.model != 6 and self.model != 7:
            with tf.variable_scope("text"):
                x = self.x_
                depth_inf = len(self.encoding_dims)
                #x = tf.layers.dropout(x, rate=0.3)
                # noisy_level = 1
                # x = x + noisy_level*tf.random_normal(tf.shape(x))
                reg_loss = 0
                attention = dense(x, self.input_dim, scope='att_layer', activation=tf.nn.sigmoid)
                x = multiply([x, attention], name='attention_mul')
                for i in range(depth_inf):
                    x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)

                h_encode = x
                print(x.shape)
                z_mu = slim.fully_connected(h_encode, self.z_dim, scope="mu_layer")
                z_log_sigma_sq = slim.fully_connected(h_encode, self.z_dim, scope="sigma_layer")
                e = tf.random_normal(tf.shape(z_mu))
                z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

                # generative process
                print(self.decoding_dims)
                depth_gen = len(self.decoding_dims)
                y = z
                for i in range(depth_gen):
                    y = dense(y, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)



                x_recons = y

        if self.model == 2 or self.model == 3:
            self.x_im_ = placeholder((None, self.input_width, self.input_height, self.channel))
            self.x_s_ = placeholder((None, 4526))

            with tf.variable_scope("structure"):
                x_s = self.x_s_
                depth_inf = len(self.encoding_dims_str)
                for i in range(depth_inf):
                    x_s = dense(x_s, self.encoding_dims_str[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # print("enc_layer0/weights:0".graph)
                h_s_encode = x_s
                z_s_mu = dense(h_s_encode, self.z_dim, scope="mu_layer")
                z_s_log_sigma_sq = dense(h_s_encode, self.z_dim, scope = "sigma_layer")
                e_s = tf.random_normal(tf.shape(z_s_mu))
                z_s = z_s_mu + tf.sqrt(tf.maximum(tf.exp(z_s_log_sigma_sq), self.eps)) * e_s

                # generative process
                depth_gen = len(self.decoding_dims_str)
                y_s = z_s
                for i in range(depth_gen):
                    y_s = dense(y_s, self.decoding_dims_str[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # if last_layer_nonelinear: depth_gen -1

                x_s_recons = y_s

        if self.model == 1 or self.model == 2 or self.model==6:

            with tf.variable_scope("image"):
                x_im_ = self.x_im_
                x_im = x_im_
                keep_prob = 0.8
                #x_im = tf.layers.dropout(x_im, rate=0.3)
                # for i in range(self.num_conv):
                #     x_im = conv2d(x_im, self.filter * np.power(2, i),kernel_size=(2,2), strides=(2,2), scope="enc_layer"+"%s" %i, activation=tf.nn.relu)

                x_im = conv2d(x_im, 32,kernel_size=(3,3), strides=(2,2), scope="enc_layer0", activation=tf.nn.relu)
                x_im = tf.nn.dropout(x_im, keep_prob)
                x_im = conv2d(x_im, 64,kernel_size=(3,3), strides=(2,2), scope="enc_layer1", activation=tf.nn.relu)
                x_im = tf.nn.dropout(x_im, keep_prob)
                x_im = conv2d(x_im, 128,kernel_size=(3,3), strides=(2,2), scope="enc_layer2", activation=tf.nn.relu)
                x_im = tf.nn.dropout(x_im, keep_prob)
                x_im = conv2d(x_im, 256,kernel_size=(3,3), strides=(2,2), scope="enc_layer3", activation=tf.nn.relu)
                x_im = tf.nn.dropout(x_im, keep_prob)
                x_im = conv2d(x_im, 256,kernel_size=(3,3), strides=(2,2), scope="enc_layer4", activation=tf.nn.relu)
                # x_im = conv2d(x_im, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer5", activation=tf.nn.relu)
                # x_im = max_pool(x_im, kernel_size=(3,3), strides=(2,2))

                h_im_encode = tf.reshape(x_im, [-1, 256])
                z_im_mu = dense(h_im_encode, self.z_dim, scope="mu_layer")
                z_im_log_sigma_sq = dense(h_im_encode, self.z_dim, scope = "sigma_layer")
                e_im = tf.random_normal(tf.shape(z_im_mu))
                z_im = z_im_mu + tf.sqrt(tf.maximum(tf.exp(z_im_log_sigma_sq), self.eps)) * e_im

                # generative process
                # h_decode = dense(z_im, self.intermediate_dim, activation=tf.nn.relu)
                h_upsample = dense(z_im, 256, activation=tf.nn.relu)
                y_im = tf.reshape(h_upsample, [-1, 1, 1, 256])

                # y_im = conv2d_transpose(y_im, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer0", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 256, kernel_size=(3,3), strides=(2,2), scope="dec_layer1", activation=tf.nn.relu)
                y_im = tf.nn.dropout(y_im, keep_prob)
                y_im = conv2d_transpose(y_im, 128, kernel_size=(3,3), strides=(2,2), scope="dec_layer2", activation=tf.nn.relu)
                y_im = tf.nn.dropout(y_im, keep_prob)
                y_im = conv2d_transpose(y_im, 64, kernel_size=(3,3), strides=(2,2), scope="dec_layer3", activation=tf.nn.relu)
                y_im = tf.nn.dropout(y_im, keep_prob)
                y_im= conv2d_transpose(y_im, 32, kernel_size=(3,3), strides=(2,2), scope="dec_layer4", activation=tf.nn.relu)
                y_im = tf.nn.dropout(y_im, keep_prob)
                y_im = conv2d_transpose(y_im, 3, kernel_size=(3,3), strides=(2,2), scope="dec_layer5", activation=tf.nn.relu)

                x_im_recons = y_im
                m = tf.reshape(x_im_, [-1, self.input_width*self.input_height, self.channel])
                n = tf.reshape(x_im_recons, [-1, self.input_width*self.input_height, self.channel])


        with tf.variable_scope("user"):
            encoding_dims = [200]
            decoding_dims = [200,self.user_dim]

            x_u = self.x_u_
            depth_inf = len(encoding_dims)

            attention = dense(x_u, self.user_dim, scope='att_layer', activation=tf.nn.sigmoid)
            x_u = multiply([x_u, attention],name='attention_mul')
            for i in range(depth_inf):
                x_u = dense(x_u, encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)

            h_u_encode = x_u
            z_u_mu = slim.fully_connected(h_u_encode, self.z_dim, scope="mu_layer")
            z_u_log_sigma_sq = slim.fully_connected(h_u_encode, self.z_dim, scope="sigma_layer")
            e_u = tf.random_normal(tf.shape(z_u_mu))
            z_u = z_u_mu + tf.sqrt(tf.maximum(tf.exp(z_u_log_sigma_sq), self.eps)) * e_u
            # generative process
            depth_gen = len( decoding_dims)
            y_u = z_u
            for i in range(depth_gen):
                y_u = dense(y_u, decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
            x_u_recons = y_u

        loss_u_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_u_, x_u_recons), axis=1))
        loss_u_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_u_mu) + tf.exp(z_u_log_sigma_sq)
            - z_u_log_sigma_sq - 1, 1))
        loss_u = 1.0*self.params.lambda_u/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.u_ - z_u), 1))
        self.loss_e_step_u = loss_u_recons+ loss_u_kl +loss_u

        if self.loss_type == "cross_entropy":
            if self.model != 6:
                loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_, x_recons), axis=1))
                loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq)
            - z_log_sigma_sq - 1, 1))
            else:
                loss_im_recons = -tf.reduce_mean(tf.reduce_sum(m * tf.log(tf.maximum(n, 1e-10)) + (1-m) * tf.log(tf.maximum(1 - n, 1e-10)),1))
                loss_im_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_im_mu) + tf.exp(z_im_log_sigma_sq) - z_im_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_  - z_im), 1))
                self.loss_e_step = loss_v + loss_im_kl + loss_im_recons

            if self.model == 0:
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_v

            elif self.model == 1:
                # loss_im_recons = self.input_width * self.input_height * metrics.binary_crossentropy(K.flatten(x_im_), K.flatten(x_im_recons))
                # loss_im_kl = 0.5 * tf.reduce_sum(tf.square(z_im_mu) + tf.exp(z_im_log_sigma_sq) - z_im_log_sigma_sq - 1, 1)
                # loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_im), 1))
                # self.loss_e_step = loss_recons + loss_kl + loss_v + K.mean(loss_im_recons + loss_im_kl)
                loss_im_recons = -tf.reduce_mean(tf.reduce_sum(m * tf.log(tf.maximum(n, 1e-10)) + (1-m) * tf.log(tf.maximum(1 - n, 1e-10)),1))
                loss_im_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_im_mu) + tf.exp(z_im_log_sigma_sq) - z_im_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z - z_im), 1))
                self.loss_e_step = loss_v + loss_im_kl + loss_im_recons + loss_kl + loss_recons

            elif self.model == 3:
                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_s_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_s_mu) + tf.exp(z_s_log_sigma_sq) - z_s_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_s), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_s_recons + loss_s_kl + loss_v

            elif self.model == 2:
                print("abc")
                loss_im_recons = -tf.reduce_mean(tf.reduce_sum(m * tf.log(tf.maximum(n, 1e-10)) + (1-m) * tf.log(tf.maximum(1 - n, 1e-10)),1))
                loss_im_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_im_mu) + tf.exp(z_im_log_sigma_sq) - z_im_log_sigma_sq - 1, 1))

                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_s_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_s_mu) + tf.exp(z_s_log_sigma_sq) - z_s_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_s - z_im), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_s_recons + loss_s_kl + loss_v + loss_im_recons + loss_im_kl

        with tf.variable_scope("loss"):
            train_op_u = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step_u)
            train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "cvae_%d.ckpt"%self.model)
        if self.initial:
            if self.model != 6:
                ckpt_file = os.path.join(self.ckpt_model, "vae_text.ckpt")
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

            ckpt_file = os.path.join(self.ckpt_model, "vae_user.ckpt")
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
            if self.model != 0:
                img_batch = im_data[idx]
                str_batch = str_data[idx]
                _, lu = self.sess.run((train_op_u, self.loss_e_step_u),
                                     feed_dict={self.x_:x_batch, self.v_:v_batch, self.x_s_:str_batch, self.x_im_:img_batch,
                                                self.x_u_: x_u_batch, self.u_:u_batch})
            else:
                _, _,l, lu = self.sess.run((train_op, train_op_u, self.loss_e_step, self.loss_e_step_u),
                                     feed_dict={self.x_:x_batch, self.v_:v_batch,
                                                self.x_u_: x_u_batch, self.u_:u_batch})
            if i % 50 == 0:
               print("epoches: %d\t loss u: %f\t time: %d s"%(i,lu, time.time()-start))
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

        if self.model == 1 or self.model == 2:
            self.z_im_mu = z_im_mu
            self.x_im_recons = x_im_recons

        if self.model == 2 or self.model == 3:
            self.z_s_mu = z_s_mu
            self.x_s_recons = x_s_recons

        self.z_u_mu = z_u_mu
        self.x_u_recons = x_u_recons
        self.saver.save(self.sess, ckpt)
        return None

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

    def save_model(self, save_path_pmf):
        # self.saver.save(self.sess, save_path_weights)
        scipy.io.savemat(save_path_pmf, {"U":self.U, "V":self.V, "Z":self.exp_z, "Z_im":self.exp_z_im})
        print "all parameters saved"

    def load_model(self, load_path_pmf):
        # self.saver.restore(self.sess, load_path_weights)
        data = scipy.io.loadmat(load_path_pmf)
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
        pred_all = np.argsort(-pred_all)
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]

        recall_avgs = []
        precision_avgs = []
        mapk_avgs = []
        for m in [10]:
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            ndcg = []
            hit = 0
            for i in range(len(user_all)):
                p = pred_all[i, :(m + len(train_users[i]))]
                p = p.tolist()
                for u in train_users[i]:
                    if u in p:
                        p.remove(u)
                top_M = p[:m]
                hits = set(test_users[i]) & set(top_M)
                hits_num = len(hits)
                if hits_num > 0:
                    hit += 1
                try:
                    recall_val = float(hits_num) / float(ground_tr_num[i])
                except:
                    recall_val = 1
                recall_vals.append(recall_val)
                # pred = np.array(pred_all[i])
                # score = []
                # for k in range(m):
                #     if top_M[k] in hits:
                #         score.append(1)
                #     else:
                #         score.append(0)
                # actual = self.dcg_score(score, pred[top_M], m)
                # best = self.dcg_score(score, score, m)
                # if best ==0:
                #     ndcg.append(0)
                # else:
                #     ndcg.append(float(actual)/best)
                # precision = float(hits_num) / float(m)
                # precision_vals.append(precision)

            recall_avg = np.mean(np.array(recall_vals))
            # precision_avg = np.mean(np.array(precision_vals))
            # mapk = ml_metrics.mapk([list(np.argsort(-pred_all[k])) for k in range(len(pred_all)) if len(user_all[k])!= 0],
            #                        [u for u in user_all if len(u)!=0], m)

            print("recall %f, hit: %f, NDCG:"%(recall_avg, float(hit)/len(user_all),))
            #print recall_avg
            if file != None:
                file.write("m = %d, recall = %f\t"%(m, recall_avg))
            # precision_avgs.append(precision_avg)
        return recall_avg
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



def load_cvae_data(data_dir, item_no):
  variables = load_npz(os.path.join(data_dir,"item.npz"))
  dataset = {}
  dataset["content"] = variables.toarray()


  dataset["train_users"], dataset["train_items"], dataset["test_users"], dataset["train_no"] = read_file(data_dir, item_no)
  user_info = list(open(os.path.join(data_dir, "user_info_train.txt"))) + \
              list(open(os.path.join(data_dir, "user_info_test.txt")))
  user_info = [u.strip() for u in user_info]
  user_info = [u.split(",") for u in user_info]
  user_info = [u[1:] for u in user_info]
  dataset['user_info'] = np.array(user_info).astype(np.float32)

  return dataset

def read_file(dir, item_no):
    train = []
    infer = []
    item = [0] * item_no
    idx = 0
    for line in open(os.path.join(dir, "implicit/train.txt")):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:-1]]
            for i in l:
                if item[i] == 0:
                    item[i] = [idx]
                else:
                    item[i].append(idx)
        train.append(l)
        infer.append([int(a[-1])])
        idx += 1
    train_no = idx
    for line in open(os.path.join(dir, "implicit/test.txt")):
        a = line.strip().split()
        if a == []:
            l = []
        else:
            l = [int(x) for x in a[1:-1]]
            for i in l:
                if item[i] == 0:
                    item[i] = [idx]
                else:
                    item[i].append(idx)
        train.append(l)
        infer.append([int(a[-1])])
        idx += 1

    for i in range(len(item)):
        if item[i] == 0:
            item[i] = []

    print(train_no)

    return train, item, infer, train_no

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=int, default=0,
                        help='type of model: 0-only text, 1-text+image, 2-text+image+structure, 3-text+structure')
    parser.add_argument('--ckpt_folder', type=str, default='pre_model/exp1/',
                        help='where model is stored')
    parser.add_argument('--initial', type=bool, default=True,
                        help='where model is stored')
    parser.add_argument('--iter', type=int, default=50,
                        help='where model is stored')
    parser.add_argument('--data_dir', type=str, default='data/amazon',
                        help='where model is stored')
    parser.add_argument('--zdim', type=int, default=50,
                        help='where model is stored')
    parser.add_argument('--gridsearch', type=int, default=1,
                        help='gridsearch or not')
    parser.add_argument('--mat_file', type=int, default=0,
                        help='gridsearch or not')

    args = parser.parse_args()
    model_type = args.model
    ckpt = args.ckpt_folder
    initial = args.initial
    iter = args.iter
    data_dir = args.data_dir
    zdim = args.zdim
    gs = args.gridsearch
    print(model_type)
    params = params_class()
    params.lambda_u = 10
    params.lambda_v = 1
    params.lambda_r = 0.1
    params.C_a = 1
    params.C_b = 0.01
    params.max_iter_m = 1
    params.EM_iter = args.iter
    params.num_iter = 150

    C = [0.1, 1, 10]

    # # for updating W and b in vae
    # self.learning_rate = 0.001
    # self.batch_size = 500
    # self.num_iter = 3000
    # self.EM_iter = 100
    n_item = len(list(open(os.path.join(data_dir, "item_id.txt"))))

    data = load_cvae_data(data_dir, n_item)
    np.random.seed(0)
    tf.set_random_seed(0)

    # images = np.fromfile(os.path.join(data_dir,"images.bin"), dtype=np.uint8)
    # img = images.reshape((13791, 32, 32, 3))
    # img = img.astype(np.float32)/255
    num_factors = zdim
    best_recall = 0
    best_hyper = []

    dim = data['content'].shape[1]


    if gs == 1:
        i = 0
        recalls = []
        for u in [0.1, 1, 10]:
            params.lambda_u = u
            for v in [1, 10, 100]:
                params.lambda_v = v
                for r in [0.1, 1, 10]:
                    params.lambda_r = r
                    if i > -1:
                        model = cf_vae_extend(num_users=len(data['train_users']), num_items=len(data["train_items"]),
                                              num_factors=num_factors,
                                              params=params,
                                              input_dim=dim, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200,
                                                                                                                 400,dim],
                                              decoding_dims_str=[200, 4526], loss_type='cross_entropy',
                                              model = model_type, ckpt_folder=ckpt)
                        model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"], data['user_info'])
                        model.save_model(os.path.join(ckpt,"cf_vae_%d_%d.mat"%(model_type, i)))
                        # model.load_model("cf_vae.mat")
                        f = open(os.path.join(ckpt, "result_cvae_%d.txt"%model_type), 'a')
                        f.write("%d-----------%f----------%f----------%f\n"%(i,u,v,r))
                        pred_all = model.predict_all()
                        f.write("val: ")
                        recall = model.predict_val(pred_all[:data['train_no']], data["train_users"][:data['train_no']],data["test_users"][:data['train_no']], f)
                        f.write("\n")
                        f.write("test: ")
                        recall = model.predict_val(pred_all[data['train_no']:], data["train_users"][data['train_no']:],data["test_users"][data['train_no']:], f)
                        f.write("\n")
                        f.close()
                        if recall > best_recall:
                            best_recall = recall
                            best_hyper = [u,v, r, i]

                        print(u, v, r)
                    i += 1

        f = open(os.path.join(ckpt, "result_sum.txt"), "a")
        f.write("Best recall CVAE: %f at %d (%f, %f, %f)\n" % (best_recall, best_hyper[3], best_hyper[0], best_hyper[1],
                                                                                       best_hyper[2]))
        f.close()
    else:
        u = [0.1, 1, 10]
        v = [1,10, 100]
        r = [0.1, 1, 10]
        params.lambda_u = u[int(args.mat_file/9)]
        params.lambda_v = v[int((args.mat_file%9)/3)]
        params.lambda_r = r[int(args.mat_file%3)]

        model = cf_vae_extend(num_users=data['user_no'], num_items=data['item_no'], num_factors=num_factors, params=params,
                              input_dim=dim, encoding_dims=[400, 200], z_dim=zdim, decoding_dims=[200, 400,dim],
                                decoding_dims_str=[200, 500, 4526], loss_type='cross_entropy',
                              model = model_type, ckpt_folder=ckpt)
        model.fit(data["train_users"], data["train_items"], data["content"], params, data["test_users"])
        model.save_model(os.path.join(ckpt,"cf_vae_%d.mat"%(model_type)))
        #model.load_model(os.path.join(ckpt, "cf_dae_0.mat"))
        pred_all = model.predict_all()
        pred_all = pred_all[data['test_users'].keys()]
        recall = model.predict_val(pred_all, train_user, data["test_users"].values())

if __name__ == '__main__':
    main()

