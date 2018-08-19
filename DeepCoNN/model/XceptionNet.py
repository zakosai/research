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
from keras.layers import Input, Embedding, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dense, Flatten
from k_maxpooling import *

# from tensor2tensor.layers import common_attention
# from tensor2tensor.layers import common_hparams
# from tensor2tensor.layers import common_layers
# from tensor2tensor.utils import registry
# from tensor2tensor.utils import t2t_model
# from tensor2tensor.models import slicenet



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


        # self.h_pool_u = self.Xception("user_conv", self.embedded_users)
        # _, width, height, channel = self.h_pool_u.get_shape().as_list()
        # filter_u = width * height * channel
        # self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, filter_u])
        #
        # self.h_pool_i = self.Xception("item_conv", self.embedded_items)
        # _, width, height, channel = self.h_pool_i.get_shape().as_list()
        # filter_i = width * height * channel
        # self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, width * height*channel])
        #
        #
        # with tf.name_scope("dropout"):
        #     self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
        #     self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        # with tf.name_scope("get_fea"):
        #     Wu = tf.get_variable(
        #         "Wu",
        #         shape=[filter_u, n_latent],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
        #     self.u_fea = tf.matmul(self.h_drop_u, Wu) + bu
        #     # self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
        #     Wi = tf.get_variable(
        #         "Wi",
        #         shape=[filter_i, n_latent],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
        #     self.i_fea = tf.matmul(self.h_drop_i, Wi) + bi
        #     # self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        self.u_fea = self.VDCNN(self.embedded_user, n_latent, 9)
        self.i_fea = self.VDCNN(self.embedded_item, n_latent, 9)

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

    def identity_block(self, inputs, filters, kernel_size=3, use_bias=False, shortcut=False):
        conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        relu = Activation('relu')(bn1)
        conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(relu)
        out = BatchNormalization()(conv2)
        if shortcut:
            out = Add()([out, inputs])
        return Activation('relu')(out)

    def conv_block(self,inputs, filters, kernel_size=3, use_bias=False, shortcut=False,
                   pool_type='max', sorted=True, stage=1):
        conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(bn1)

        conv2 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(relu1)
        out = BatchNormalization()(conv2)

        if shortcut:
            residual = Conv1D(filters=filters, kernel_size=1, strides=2, name='shortcut_conv1d_%d' % stage)(inputs)
            residual = BatchNormalization(name='shortcut_batch_normalization_%d' % stage)(residual)
            out = self.downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
            out = Add()([out, residual])
            out = Activation('relu')(out)
        else:
            out = Activation('relu')(out)
            out = self.downsample(out, pool_type=pool_type, sorted=sorted, stage=stage)
        if pool_type is not None:
            out = Conv1D(filters=2 * filters, kernel_size=1, strides=1, padding='same', name='1_1_conv_%d' % stage)(out)
            out = BatchNormalization(name='1_1_batch_normalization_%d' % stage)(out)
        return out

    def downsample(self, inputs, pool_type='max', sorted=True, stage=1):
        if pool_type == 'max':
            out = MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_%d' % stage)(inputs)
        elif pool_type == 'k_max':
            k = int(inputs._keras_shape[1] / 2)
            out = KMaxPooling(k=k, sorted=sorted, name='pool_%d' % stage)(inputs)
        elif pool_type == 'conv':
            out = Conv1D(filters=inputs._keras_shape[-1], kernel_size=3, strides=2, padding='same',
                         name='pool_%d' % stage)(inputs)
            out = BatchNormalization()(out)
        elif pool_type is None:
            out = inputs
        else:
            raise ValueError('unsupported pooling type!')
        return out

    def VDCNN(self, embedded_chars, n_latent, depth=9, sequence_length=1024, embedding_dim=16,
              shortcut=False, pool_type='max', sorted=True, use_bias=False, input_tensor=None):
        if depth == 9:
            num_conv_blocks = (1, 1, 1, 1)
        elif depth == 17:
            num_conv_blocks = (2, 2, 2, 2)
        elif depth == 29:
            num_conv_blocks = (5, 5, 2, 2)
        elif depth == 49:
            num_conv_blocks = (8, 8, 5, 3)
        else:
            raise ValueError('unsupported depth for VDCNN.')

        out = Conv1D(filters=64, kernel_size=3, strides=1, padding='same', name='temp_conv')(embedded_chars)

        # Convolutional Block 64
        for _ in range(num_conv_blocks[0] - 1):
            out = self.identity_block(out, filters=32, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
        out = self.conv_block(out, filters=32, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                         pool_type=pool_type, sorted=sorted, stage=1)

        # Convolutional Block 128
        for _ in range(num_conv_blocks[1] - 1):
            out = self.identity_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
        out = self.conv_block(out, filters=64, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                         pool_type=pool_type, sorted=sorted, stage=2)

        # Convolutional Block 256
        for _ in range(num_conv_blocks[2] - 1):
            out = self.identity_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
        out = self.conv_block(out, filters=128, kernel_size=3, use_bias=use_bias, shortcut=shortcut,
                         pool_type=pool_type, sorted=sorted, stage=3)

        # # Convolutional Block 512
        # for _ in range(num_conv_blocks[3] - 1):
        #     out = self.identity_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=shortcut)
        # out = self.conv_block(out, filters=256, kernel_size=3, use_bias=use_bias, shortcut=False,
        #                  pool_type=None, stage=4)

        # k-max pooling with k = 8
        out = KMaxPooling(k=8, sorted=True)(out)
        out = Flatten()(out)

        # Dense Layers
        out = Dense(512, activation='relu')(out)
        out = Dense(n_latent)(out)
        return out