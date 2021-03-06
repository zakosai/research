import tensorflow as tf
from tensorbayes.layers import dense, placeholder, conv2d, conv2d_transpose
from tensorbayes.utils import progbar
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits
from keras.metrics import binary_crossentropy
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
import numpy as np
import time
import os
from resnet_model import conv2d_fixed_padding, block_layer, building_block
class vanilla_vae:
    """
    build a vanilla vae
    you can customize the activation functions pf each layer yourself.
    """

    def __init__(self, width, height, channel=3, filter=64, intermediate_dim=256, num_conv=4, num_layers=4,
                 z_dim=50, loss="cross_entropy", useTranse = False, eps = 1e-10, ckpt_folder="pre3/dae"):
        # useTranse: if we use trasposed weigths of inference nets
        # eps for numerical stability
        # structural info
        self.input_width = width
        self.input_height = height
        self.channel = channel
        self.filter = filter
        self.intermediate_dim = intermediate_dim
        self.num_conv = num_conv
        self.z_dim = z_dim
        self.num_layers = num_layers
        self.loss = loss
        self.useTranse = useTranse
        self.eps = eps
        self.weights = []    # better in np form. first run, then append in
        self.bias = []
        self.ckpt = ckpt_folder



    def fit(self, x_input, epochs = 1000, learning_rate = 0.001, batch_size = 100, print_size = 50, train=True, scope="text"):
        # training setting
        self.DO_SHARE = False
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_size = print_size

        self.g = tf.Graph()
        # inference process
        ########TEXT###################
        with tf.variable_scope(scope):
            dropout_rate = 0.3
            x_ = placeholder((None, self.input_width, self.input_height, self.channel))
            x = x_
            x = tf.layers.dropout(x, dropout_rate, training=train )


            # for i in range(self.num_conv):
            #     x = conv2d(x, self.filter * np.power(2, i),kernel_size=(2,2), strides=(2,2), scope="enc_layer"+"%s" %i, activation=tf.nn.relu)
            x = conv2d(x, 64,kernel_size=(3,3), strides=(2,2), scope="enc_layer0", activation=tf.nn.relu)
            x = conv2d(x, 128,kernel_size=(3,3), strides=(2,2), scope="enc_layer1", activation=tf.nn.relu)
            x = conv2d(x, 256,kernel_size=(3,3), strides=(2,2), scope="enc_layer2", activation=tf.nn.relu)
            x = conv2d(x, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer3", activation=tf.nn.relu)
            x = conv2d(x, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer4", activation=tf.nn.relu)
            # x = conv2d(x, 512,kernel_size=(3,3), strides=(2,2), scope="enc_layer5", activation=tf.nn.relu)

            flat = Flatten()(x)
            print(flat.shape)
            h_encode = Dense(self.intermediate_dim, activation='relu')(flat)
            z = dense(h_encode, self.z_dim, scope="mu_layer")


            # generative process
            h_decode = dense(z, self.intermediate_dim, activation=tf.nn.relu)
            h_upsample = dense(h_decode, 512, activation=tf.nn.relu)
            y = Reshape((1,1,512))(h_upsample)

            # for i in range(self.num_conv-1):
            #     y = conv2d_transpose(y, self.filter*np.power(2,self.num_conv-2-i), kernel_size=(2,2),
            #                          strides=(2,2), scope="dec_layer"+"%s" %i, activation=tf.nn.relu)
            #
            # y = conv2d_transpose(y, self.channel, scope="dec_layer"+"%s" %(self.num_conv-1) , kernel_size=(2,2),
            #                          strides=(2,2), activation=tf.nn.relu)
                    # if last_layer_nonelinear: depth_gen -1
            # y = conv2d_transpose(y, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer0", activation=tf.nn.relu)
            y = conv2d_transpose(y, 512, kernel_size=(3,3), strides=(2,2), scope="dec_layer1", activation=tf.nn.relu)
            y = conv2d_transpose(y, 256, kernel_size=(3,3), strides=(2,2), scope="dec_layer2", activation=tf.nn.relu)
            y = conv2d_transpose(y, 128, kernel_size=(3,3), strides=(2,2), scope="dec_layer3", activation=tf.nn.relu)
            y = conv2d_transpose(y, 64, kernel_size=(3,3), strides=(2,2), scope="dec_layer4", activation=tf.nn.relu)
            y = conv2d_transpose(y, 3, kernel_size=(3,3), strides=(2,2), scope="dec_layer5", activation=tf.nn.relu)
            x_recons = y

        loss_recons = self.input_width * self.input_height *binary_crossentropy(K.flatten(x_), K.flatten(x_recons))
        # loss_kl = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mu) + tf.exp(z_log_sigma_sq) - z_log_sigma_sq - 1, 1))
        loss = K.mean(loss_recons)
        # other cases not finished yet
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope))
        ckpt_file = os.path.join(self.ckpt,"dae_%s.ckpt" %scope)
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            start = time.time()
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                _, l = sess.run((train_op, loss), feed_dict={x_:x_batch})
                if i % self.print_size == 0:
                    print("epoches: %d\t loss: %f\t time: %d s"%(i, l, time.time()-start))

            saver.save(sess, ckpt_file)
        else:
            saver.restore(sess, ckpt_file)


