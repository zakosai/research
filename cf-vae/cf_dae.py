import tensorflow as tf
from tensorbayes.layers import dense, placeholder, conv2d, conv2d_transpose
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
import os
import math

class params:
    def __init__(self):
        self.C_a = 1.0
        self.C_b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 10.0
        self.lambda_r = 1.0
        self.max_iter_m = 30

        # for updating W and b in vae
        self.learning_rate = 0.001
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30

class cf_vae_extend:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims,
                 decoding_dims_str, loss_type="cross_entropy", useTranse = False, eps = 1e-10, model=0, ckpt_folder="pre3/dae"):
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
        self.initial = True

        self.input_width = 32
        self.input_height = 32
        self.channel = 3
        self.num_conv = 4
        self.intermediate_dim = 256
        self.filter = 64
        self.model = model #0 -> only text, 1-> text + image, 2 -> text + image + structure, 3-> text + structure
        self.ckpt =ckpt_folder

    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data, im_data, str_data):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        # self.x_s_ = placeholder((None, 1863))

        self.v_ = placeholder((None, self.num_factors))

        # inference process

        with tf.variable_scope("text"):
            x = self.x_
            x = tf.layers.dropout(x, rate=0.3)
            depth_inf = len(self.encoding_dims)
            for i in range(depth_inf):
                x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
                # print("enc_layer0/weights:0".graph)
            h_encode = x
            z = dense(h_encode, self.z_dim, scope="mu_layer", activation=tf.nn.sigmoid)


            # generative process
            depth_gen = len(self.decoding_dims)
            y = z
            for i in range(depth_gen):
                y = dense(y, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                # if last_layer_nonelinear: depth_gen -1

            x_recons = y
        if self.model == 2 or self.model == 3:
            self.x_s_ = placeholder((None, 4526))

            with tf.variable_scope("structure"):
                x_s = self.x_s_
                depth_inf = len(self.encoding_dims)
                for i in range(depth_inf):
                    x_s = dense(x_s, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # print("enc_layer0/weights:0".graph)
                h_s_encode = x_s
                z_s = dense(h_s_encode, self.z_dim, scope="mu_layer")

                # generative process
                depth_gen = len(self.decoding_dims_str)
                y_s = z_s
                for i in range(depth_gen):
                    y_s = dense(y_s, self.decoding_dims_str[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # if last_layer_nonelinear: depth_gen -1

                x_s_recons = y_s

        if self.model == 1 or self.model ==2:
            self.x_im_ = placeholder((None, self.input_width, self.input_height, self.channel))

            with tf.variable_scope("image"):
                x_im_ = self.x_im_
                x_im = x_im_
                # for i in range(self.num_conv):
                #     x_im = conv2d(x_im, self.filter * np.power(2, i),kernel_size=(2,2), strides=(2,2), scope="enc_layer"+"%s" %i, activation=tf.nn.relu)

                x_im = conv2d(x_im, 64,kernel_size=(3,3), strides=(2,2), scope="enc_layer0", activation=tf.nn.relu)
                x_im = conv2d(x_im, 128,kernel_size=(3,3), strides=(2,2), scope="enc_layer1", activation=tf.nn.relu)
                x_im = conv2d(x_im, 256,kernel_size=(3,3), strides=(2,2), scope="enc_layer2", activation=tf.nn.relu)
                x_im = conv2d(x_im, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer3", activation=tf.nn.relu)
                x_im = conv2d(x_im, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer4", activation=tf.nn.relu)
                # x_im = conv2d(x_im, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer5", activation=tf.nn.relu)

                # num_blocks = 3
                # is_training = True
                # data_format = 'channels_last'
                # x_im = conv2d_fixed_padding( inputs=x_im, filters=64, kernel_size=3, strides=1,
                #                                data_format=data_format)
                # x_im = tf.identity(x_im, 'initial_conv')
                #
                # x_im = block_layer(inputs=x_im, filters=64, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer1', data_format=data_format)
                #
                # x_im = block_layer(inputs=x_im, filters=128, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer2', data_format=data_format)
                #
                # x_im = block_layer(inputs=x_im, filters=256, block_fn=building_block, blocks=num_blocks,
                #                     strides=2, is_training=is_training, name='block_layer3',data_format=data_format)
                #
                # x_im = block_layer(inputs=x_im, filters=512, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer4', data_format=data_format)
                # x_im = block_layer(inputs=x_im, filters=512, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer4', data_format=data_format)
                # x_im = block_layer(inputs=x_im, filters=512, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer4', data_format=data_format)
                flat = Flatten()(x_im)
                h_im_encode = Dense(self.intermediate_dim, activation='relu')(flat)
                z_im = dense(h_im_encode, self.z_dim, scope="mu_layer")


                # generative process
                h_decode = dense(z_im, self.intermediate_dim, activation=tf.nn.relu)
                h_upsample = dense(h_decode, 512, activation=tf.nn.relu)
                y_im = Reshape((1,1,512))(h_upsample)

                # for i in range(self.num_conv-1):
                #     y_im = conv2d_transpose(y_im, self.filter*np.power(2,self.num_conv-2-i), kernel_size=(2,2),
                #                          strides=(2,2), scope="dec_layer"+"%s" %i, activation=tf.nn.relu)
                #
                # y_im = conv2d_transpose(y_im, self.channel, scope="dec_layer"+"%s" %(self.num_conv-1) , kernel_size=(2,2),
                #                          strides=(2,2), activation=tf.nn.relu)
                        # if last_layer_nonelinear: depth_gen -1
                # y_im = conv2d_transpose(y_im, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer0", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer1", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 256, kernel_size=(3,3), strides=(2,2), scope="dec_layer2", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 128, kernel_size=(3,3), strides=(2,2), scope="dec_layer3", activation=tf.nn.relu)
                y_im= conv2d_transpose(y_im, 64, kernel_size=(3,3), strides=(2,2), scope="dec_layer4", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 3, kernel_size=(3,3), strides=(2,2), scope="dec_layer5", activation=tf.nn.relu)

                x_im_recons = y_im
        sparsity_weight = 0.2
        sparsity_target = 0.1
        def kl_divergence(p,q):
            return  p*tf.log(p/q) + (1-p)*tf.log((1-p)/(1-q))

        if self.loss_type == "cross_entropy":
            loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_, x_recons), axis=1))
            sparsity_loss = tf.reduce_sum(kl_divergence(tf.reduce_mean(z), sparsity_target))
            # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
            if self.model == 1:
                # loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_im_recons = self.input_width * self.input_height * metrics.binary_crossentropy(K.flatten(x_im_), K.flatten(x_im_recons))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z - z_im), 1))
                self.loss_e_step = loss_recons + loss_v + K.mean(loss_im_recons)
            elif self.model == 2:
                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_im_recons = self.input_width * self.input_height * metrics.binary_crossentropy(K.flatten(x_im_), K.flatten(x_im_recons))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z - z_im - z_s), 1))
                self.loss_e_step = loss_recons + loss_v + K.mean(loss_im_recons) +loss_s_recons

            elif self.model ==3:
                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z - z_s), 1))
                self.loss_e_step = loss_recons + loss_v + loss_s_recons

            else:
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z), 1))
                self.loss_e_step = loss_recons + loss_v + sparsity_loss*sparsity_weight

        train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt, "cf_dae_%d.ckpt"%self.model)
        if self.initial:
            ckpt_file = os.path.join(self.ckpt, "dae_text.ckpt")
            text_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="text")
            text_saver = tf.train.Saver(var_list=text_varlist)
            # if init == True:
            text_saver.restore(self.sess, ckpt_file)

            # LOAD IMAGE##
            if self.model == 1 or self.model == 2:
                ckpt_file_img = os.path.join(self.ckpt, "dae_image.ckpt")
                img_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="image")
                img_saver = tf.train.Saver(var_list=img_varlist)
                img_saver.restore(self.sess, ckpt_file_img)

            # Load Structure
            if self.model == 2 or self.model == 3:
                ckpt_file = os.path.join(self.ckpt, "dae_structure.ckpt")
                structure_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="structure")
                structure_saver = tf.train.Saver(var_list=structure_varlist)
                structure_saver.restore(self.sess, ckpt_file)

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
            if self.model != 0:
                img_batch = im_data[idx]
                str_batch = str_data[idx]
                _, l = self.sess.run((train_op, self.loss_e_step),
                                     feed_dict={self.x_:x_batch, self.v_:v_batch, self.x_s_:str_batch, self.x_im_:img_batch})
            else:
                _, l = self.sess.run((train_op, self.loss_e_step),
                                     feed_dict={self.x_:x_batch, self.v_:v_batch})
            if i % 50 == 0:
               print("epoches: %d\t loss: %f\t time: %d s"%(i, l, time.time()-start))

        self.z_mu = z
        self.x_recons = x_recons

        if self.model == 1 or self.model == 2:
            self.z_im_mu = z_im
            self.x_im_recons = x_im_recons

        if self.model == 2 or self.model ==3:
            self.z_s_mu = z_s
            self.x_s_recons = x_s_recons
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
                    elif self.model == 2:
                        likelihood += -0.5 * self.V[j,:].dot(B).dot((self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:] - self.exp_z_s[j,:])[:,np.newaxis])
                        ep = self.V[j,:] - self.exp_z[j,:] - self.exp_z_im[j,:] - self.exp_z_s[j, :]
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
                         x = params.lambda_v * (self.exp_z[j,:] + self.exp_z_im[j,:] + self.exp_z_s[j,:])
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
                if self.model ==1:
                    rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * (self.exp_z[v, :] + self.exp_z_im[v, :])
                else:
                    rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * self.exp_z[v, :]
                self.V[v, :] = scipy.linalg.solve(Lambda_inv, rx, check_finite=True)

            print("iter: %d\t time:%d" %(i, time.time()-start))
        return None

    def get_exp_hidden(self, x_data, im_data, str_data):
        self.exp_z = self.sess.run(self.z_mu, feed_dict={self.x_: x_data})
        if self.model == 1 or self.model == 2:
            for i in range(len(im_data), self.params.batch_size):
                im_batch = im_data[i:i+self.params.batch_size]
                exp_z_im = self.sess.run(self.z_im_mu, feed_dict={self.x_im_: im_batch})
                self.exp_z_im = np.concatenate((self.exp_z_im, exp_z_im), axis=0)
        else:

        # print(self.exp_z_im.shape)
            self.exp_z_im = np.zeros((len(self.exp_z), self.z_dim))


        if self.model == 2 or self.model == 3:
            self.exp_z_s = self.sess.run(self.z_s_mu, feed_dict={self.x_s_: str_data})
        else:
            self.exp_z_s = 0
        return self.exp_z, self.exp_z_im, self.exp_z_s

    def fit(self, users, items, x_data, params, test_users, im_data=None, str_data=None):
        start = time.time()
        self.e_step(x_data, im_data, str_data)
        self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)
        for i in range(params.EM_iter):
            print("iter: %d"%i)

            self.pmf_estimate(users, items, params)
            self.e_step(x_data, im_data, str_data)
            self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)
            if i%500 == 400:
                file = open(os.path.join(self.ckpt, "result_%d.txt"%self.model), 'a')
                file.write("---------iter %d--------\n"%i)
                pred_all = self.predict_all()
                self.predict_val(pred_all, users, test_users, file)
                self.save_model(save_path_pmf=os.path.join(self.ckpt, "cf_dae_%d_%d.mat"%(self.model, i)))
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
        user_all = map(add, train_users, test_users)
        # user_all = np.array(user_all)    # item idex from 1
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        recall_avgs = []
        for m in range(5, M, 5):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            for i in range(len(user_all)):
                top_M = np.argsort(-pred_all[i])[0:m]
                hits = set(top_M) & set(user_all[i])   # item idex from 0
                hits_num = len(hits)
                recall_val = float(hits_num) / float(ground_tr_num[i])
                recall_vals.append(recall_val)
            recall_avg = np.mean(np.array(recall_vals))
            print recall_avg
            recall_avgs.append(recall_avg)
        return recall_avgs

    def predict_val(self, pred_all, train_users, test_users, file=None):
        user_all = test_users
        ground_tr_num = [len(user) for user in user_all]


        pred_all = list(pred_all)

        for m in [10, 100]:
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
            if file!= None:
                file.write("m = %d, recall = %f"%(m, recall_avg))

    def predict_all(self):
        return np.dot(self.U, (self.V.T))
