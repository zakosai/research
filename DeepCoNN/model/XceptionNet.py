'''
DeepCoNN
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''

import tensorflow as tf
from tensorflow.contrib import slim



class DeepCoNN(object):
    def __init__(
            self, user_length, item_length, num_classes, user_vocab_size, item_vocab_size, fm_k, n_latent, user_num,
            item_num,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, l2_reg_V=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        self.h_pool_u = self.Xception("user_conv", self.embedded_users)
        _, width, height, channel = self.h_pool_u.get_shape().as_list()
        filter_u = width * height * channel
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, filter_u])

        self.h_pool_i = self.Xception("item_conv", self.embedded_items)
        _, width, height, channel = self.h_pool_i.get_shape().as_list()
        filter_i = width * height * channel
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, width * height*channel])


        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):
            Wu = tf.get_variable(
                "Wu",
                shape=[filter_u, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_fea = tf.matmul(self.h_drop_u, Wu) + bu
            # self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            Wi = tf.get_variable(
                "Wi",
                shape=[filter_i, n_latent],
                initializer=tf.contrib.layers.xavier_initializer())
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_fea = tf.matmul(self.h_drop_i, Wi) + bi
            # self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        with tf.name_scope('fm'):
            self.z = tf.nn.relu(tf.concat(1, [self.u_fea, self.i_fea]))

            # self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1 = tf.Variable(
                tf.random_uniform([n_latent * 2, 1], -0.1, 0.1), name='fm1')
            Wf2 = tf.Variable(
                tf.random_uniform([n_latent * 2, fm_k], -0.1, 0.1), name='fm2')
            one = tf.matmul(self.z, WF1)

            inte1 = tf.matmul(self.z, Wf2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(Wf2))

            inter = (tf.square(inte1) - inte2) * 0.5

            inter = tf.nn.dropout(inter, self.dropout_keep_prob)

            inter = tf.reduce_sum(inter, 1, keep_dims=True)
            print inter
            b = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = one + inter + b

            print self.predictions
        with tf.name_scope("loss"):
            # losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

    def Xception(self, scope, x):
        with tf.variable_scope(scope):
            #===========ENTRY FLOW==============
            #Block 1 /2
            net = slim.conv2d(x, 32, [3,3], stride=2, padding='VALID', scope='block1_conv1')
            net = slim.batch_norm(net, scope='block1_bn1')
            net = tf.nn.relu(net, name='block1_relu1')
            net = slim.conv2d(net, 64, [3,3], padding='VALID', scope='block1_conv2')
            net = slim.batch_norm(net, scope='block1_bn2')
            net = tf.nn.relu(net, name='block1_relu2')
            residual = slim.conv2d(net, 128, [1,1], stride=2, scope='block1_res_conv') # -->115x24x128
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            #Block 2 /2
            net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv1', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block2_bn1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.separable_conv2d(net, 128, [3,3], scope='block2_dws_conv2', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block2_bn2')
            net = slim.max_pool2d(net, [3,3], stride=2, padding='SAME', scope='block2_max_pool')#-->115x24x128
            net = tf.add(net, residual, name='block2_add') #--> 115x24x256
            residual = slim.conv2d(net, 256, [1,1], stride=2, scope='block2_res_conv')
            residual = slim.batch_norm(residual, scope='block2_res_bn')
            #
            #Block 3 /2
            net = tf.nn.relu(net, name='block3_relu1')
            net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv1', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block3_bn1')
            net = tf.nn.relu(net, name='block3_relu2')
            net = slim.separable_conv2d(net, 256, [3,3], scope='block3_dws_conv2', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block3_bn2')
            net = slim.max_pool2d(net, [3,3], stride=2, padding='SAME', scope='block3_max_pool')
            net = tf.add(net, residual, name='block3_add')
            residual = slim.conv2d(net, 728, [1,1], stride=2, scope='block3_res_conv')
            residual = slim.batch_norm(residual, scope='block3_res_bn')

            #Block 4 /2
            net = tf.nn.relu(net, name='block4_relu1')
            net = slim.separable_conv2d(net, 728, [3,3], scope='block4_dws_conv1', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block4_bn1')
            net = tf.nn.relu(net, name='block4_relu2')
            net = slim.separable_conv2d(net, 728, [3,3], scope='block4_dws_conv2', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block4_bn2')
            net = slim.max_pool2d(net, [3,3], stride=2, padding='SAME', scope='block4_max_pool')
            net = tf.add(net, residual, name='block4_add')

            #===========MIDDLE FLOW===============
            for i in range(5):
                block_prefix = 'block%s_' % (str(i + 5))

                residual = net
                net = tf.nn.relu(net, name=block_prefix+'relu1')
                net = slim.separable_conv2d(net, 728, [3,3], scope=block_prefix+'dws_conv1', depth_multiplier=1)
                net = slim.batch_norm(net, scope=block_prefix+'bn1')
                net = tf.nn.relu(net, name=block_prefix+'relu2')
                net = slim.separable_conv2d(net, 728, [3,3], scope=block_prefix+'dws_conv2', depth_multiplier=1)
                net = slim.batch_norm(net, scope=block_prefix+'bn2')
                net = tf.nn.relu(net, name=block_prefix+'relu3')
                net = slim.separable_conv2d(net, 728, [3,3], scope=block_prefix+'dws_conv3', depth_multiplier=1)
                net = slim.batch_norm(net, scope=block_prefix+'bn3')
                net = tf.add(net, residual, name=block_prefix+'add')


            #========EXIT FLOW============
            #/2
            residual = slim.conv2d(net, 1024, [1,1], stride=2, scope='block12_res_conv')
            residual = slim.batch_norm(residual, scope='block12_res_bn')
            net = tf.nn.relu(net, name='block13_relu1')
            net = slim.separable_conv2d(net, 728, [3,3], scope='block13_dws_conv1', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block13_bn1')
            net = tf.nn.relu(net, name='block13_relu2')
            net = slim.separable_conv2d(net, 1024, [3,3], scope='block13_dws_conv2', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block13_bn2')
            net = slim.max_pool2d(net, [3,3], stride=2, padding='SAME', scope='block13_max_pool')
            net = tf.add(net, residual, name='block13_add')

            net = slim.separable_conv2d(net, 1536, [3,3], scope='block14_dws_conv1', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block14_bn1')
            net = tf.nn.relu(net, name='block14_relu1')
            net = slim.separable_conv2d(net, 2048, [3,3], scope='block14_dws_conv2', depth_multiplier=1)
            net = slim.batch_norm(net, scope='block14_bn2')
            net = tf.nn.relu(net, name='block14_relu2')

        return net