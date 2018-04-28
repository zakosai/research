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

class params:
    def __init__(self):
        self.C_a = 1.0
        self.C_b = 0.01
        self.lambda_u = 0.1
        self.lambda_v = 1.0
        self.lambda_r = 1.0
        self.max_iter_m = 30

        # for updating W and b in vae
        self.learning_rate = 0.001
        self.batch_size = 500
        self.num_iter = 300   # used in the e_step
        self.EM_iter = 30

class cf_vae_extend:
    def __init__(self, num_users, num_items, num_factors, params, input_dim, encoding_dims, z_dim, decoding_dims,
                 decoding_dims_str, loss_type="cross_entropy", useTranse = False, eps = 1e-10, model=0, ckpt_folder='pre_model', initial=True):
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


    # def e_step(self, x_data, reuse = None):
    def e_step(self, x_data, im_data, str_data):
        print "e_step finetuning"
        tf.reset_default_graph()
        self.x_ = placeholder((None, self.input_dim))  # we need these global nodes
        self.x_im_ = placeholder((None, self.input_width, self.input_height, self.channel))
        self.x_s_ = placeholder((None, 8000))
        self.v_ = placeholder((None, self.num_factors))

        # inference process

        with tf.variable_scope("text"):
            x = self.x_
            depth_inf = len(self.encoding_dims)
            # noisy_level = 1
            # x = x + noisy_level*tf.random_normal(tf.shape(x))
            for i in range(depth_inf):
                x = dense(x, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
                # print("enc_layer0/weights:0".graph)
            h_encode = x
            z_mu = dense(h_encode, self.z_dim, scope="mu_layer")
            z_log_sigma_sq = dense(h_encode, self.z_dim, scope = "sigma_layer")
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_log_sigma_sq), self.eps)) * e

            # generative process
            depth_gen = len(self.decoding_dims)
            for i in range(depth_gen):
                y = dense(z, self.decoding_dims[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                # if last_layer_nonelinear: depth_gen -1

            x_recons = y

        if self.model == 2 or self.model == 3:

            with tf.variable_scope("structure"):
                x_s = self.x_s_
                depth_inf = len(self.encoding_dims)
                for i in range(depth_inf):
                    x_s = dense(x_s, self.encoding_dims[i], scope="enc_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # print("enc_layer0/weights:0".graph)
                h_s_encode = x_s
                z_s_mu = dense(h_s_encode, self.z_dim, scope="mu_layer")
                z_s_log_sigma_sq = dense(h_s_encode, self.z_dim, scope = "sigma_layer")
                e_s = tf.random_normal(tf.shape(z_s_mu))
                z_s = z_s_mu + tf.sqrt(tf.maximum(tf.exp(z_s_log_sigma_sq), self.eps)) * e_s

                # generative process
                depth_gen = len(self.decoding_dims_str)
                for i in range(depth_gen):
                    y_s = dense(z_s, self.decoding_dims_str[i], scope="dec_layer"+"%s" %i, activation=tf.nn.sigmoid)
                    # if last_layer_nonelinear: depth_gen -1

                x_s_recons = y_s

        if self.model == 1 or self.model == 2:
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
                x_im = conv2d(x_im, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer5", activation=tf.nn.relu)
                # x_im = max_pool(x_im, kernel_size=(3,3), strides=(2,2))


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
                #                      strides=2, is_training=is_training, name='block_layer5', data_format=data_format)
                # x_im = block_layer(inputs=x_im, filters=512, block_fn=building_block, blocks=num_blocks,
                #                      strides=2, is_training=is_training, name='block_layer6', data_format=data_format)
                flat = Flatten()(x_im)
                h_im_encode = Dense(self.intermediate_dim, activation='relu')(flat)
                z_im_mu = dense(h_im_encode, self.z_dim, scope="mu_layer")
                z_im_log_sigma_sq = dense(h_im_encode, self.z_dim, scope = "sigma_layer")
                e_im = tf.random_normal(tf.shape(z_im_mu))
                z_im = z_im_mu + tf.sqrt(tf.maximum(tf.exp(z_im_log_sigma_sq), self.eps)) * e_im

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
                y_im = conv2d_transpose(y_im, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer0", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer1", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 256, kernel_size=(3,3), strides=(2,2), scope="dec_layer2", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 128, kernel_size=(3,3), strides=(2,2), scope="dec_layer3", activation=tf.nn.relu)
                y_im= conv2d_transpose(y_im, 64, kernel_size=(3,3), strides=(2,2), scope="dec_layer4", activation=tf.nn.relu)
                y_im = conv2d_transpose(y_im, 3, kernel_size=(3,3), strides=(2,2), scope="dec_layer5", activation=tf.nn.relu)

                x_im_recons = y_im

        if self.loss_type == "cross_entropy":
            loss_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_, x_recons), axis=1))
            loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))

            if self.model == 0:
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_v

            elif self.model == 1:
                loss_im_recons = self.input_width * self.input_height * metrics.binary_crossentropy(K.flatten(x_im_), K.flatten(x_im_recons))
                loss_im_kl = 0.5 * tf.reduce_sum(tf.square(z_im_mu) + tf.exp(z_im_log_sigma_sq) - z_im_log_sigma_sq - 1, 1)
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_im), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_v + K.mean(loss_im_recons + loss_im_kl)

            elif self.model == 3:
                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_s_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_s_mu) + tf.exp(z_s_log_sigma_sq) - z_s_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_s), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_s_recons + loss_s_kl + loss_v

            else:
                loss_im_recons = self.input_width * self.input_height * metrics.binary_crossentropy(K.flatten(x_im_), K.flatten(x_im_recons))
                loss_im_kl = 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1)
                loss_s_recons = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(self.x_s_, x_s_recons), axis=1))
                loss_s_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_s_mu) + tf.exp(z_s_log_sigma_sq) - z_s_log_sigma_sq - 1, 1))
                loss_v = 1.0*self.params.lambda_v/self.params.lambda_r * tf.reduce_mean( tf.reduce_sum(tf.square(self.v_ - z  - z_s - z_im), 1))
                self.loss_e_step = loss_recons + loss_kl + loss_s_recons + loss_s_kl + loss_v + K.mean(loss_im_recons +loss_im_kl)

        train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss_e_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # LOAD TEXT#
        ckpt = os.path.join(self.ckpt_model, "cvae_%d.ckpt"%self.model)
        if self.initial:
            ckpt_file = os.path.join(self.ckpt_model, "vae_text.ckpt")
            text_varlist = tf.get_collection(tf.GraphKeys.VARIABLES, scope="text")
            text_saver = tf.train.Saver(var_list=text_varlist)
            # if init == True:
            text_saver.restore(self.sess, ckpt_file)

            # LOAD IMAGE##
            if self.model == 1 or self.model == 2:
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
            img_batch = im_data[idx]
            str_batch = str_data[idx]
            _, l = self.sess.run((train_op, self.loss_e_step),
                                 feed_dict={self.x_:x_batch, self.x_im_:img_batch, self.v_:v_batch, self.x_s_:str_batch})
            if i % 50 == 0:
               print("epoches: %d\t loss: %f\t time: %d s"%(i, l, time.time()-start))

        self.z_mu = z_mu
        self.x_recons = x_recons

        if self.model == 1 or self.model == 2:
            self.z_im_mu = z_im_mu
            self.x_im_recons = x_im_recons

        if self.model == 2 or self.model == 3:
            self.z_s_mu = z_s_mu
            self.x_s_recons = x_s_recons
        self.saver.save(self.sess, ckpt)
        return None


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

                rx = params.C_a * np.sum(self.U[items[v], :], axis=0) + params.lambda_v * (self.exp_z[v, :] + self.exp_z_im[v, :])
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
             self.exp_z_im = 0

        if self.model == 2 or self.model == 3:
            self.exp_z_s = self.sess.run(self.z_s_mu, feed_dict={self.x_s_: str_data})
        else:
            self.exp_z_s = 0
        return self.exp_z, self.exp_z_im, self.exp_z_s

    def fit(self, users, items, x_data, im_data, str_data, params):

        self.e_step(x_data, im_data, str_data)
        self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)
        for i in range(params.EM_iter):
            print("iter: %d"%i)

            self.m_step(users, items, params)
            self.e_step(x_data, im_data, str_data)
            self.exp_z, self.exp_z_im, self.exp_z_s = self.get_exp_hidden(x_data, im_data, str_data)

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
        for m in range(5, M, 5):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            precision_vals = []
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
            # mapk = ml_metrics.mapk([list(np.argsort(-k)) for k in pred_all], user_all, m)
            print recall_avg
            recall_avgs.append(recall_avg)
            # precision_avgs.append(precision_avg)
            # mapk_avgs.append(mapk)

        return recall_avgs

    def predict_all(self):
        return np.dot(self.U, (self.V.T))
