import tensorflow as tf
from tensorbayes.layers import dense, placeholder
import os
from tensorflow.contrib import slim
import numpy as np
import time
class vanilla_vae:
    """
    build a vanilla vae
    you can customize the activation functions pf each layer yourself.
    """

    def __init__(self, input_dim, encoding_dims, z_dim, decoding_dims, loss="cross_entropy", useTranse = False, eps = 1e-10, ckpt_folder="pre_model"):
        # useTranse: if we use trasposed weigths of inference nets
        # eps for numerical stability
        # structural info
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.encoding_dims = encoding_dims
        self.decoding_dims = decoding_dims
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

        encoder = tf.make_template('encoder', encoder_func)
        decoder = tf.make_template('decoder', decoder_func)
        discriminator = tf.make_template('discriminator', discriminator_func)

        with tf.variable_scope(scope):

            eps = tf.random_normal([self.batch_size, self.encoding_dims[0]])
            x_real = placeholder((None, self.input_dim))
            z_sampled = tf.random_normal([self.batch_size, self.z_dim])
            z_inferred = encoder(x_real, eps)
            x_reconstr_logits = decoder(z_inferred)

            Tjoint = discriminator(x_real, z_inferred)
            Tseperate = discriminator(x_real, z_sampled)

        reconstr_err = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=x_real, logits=x_reconstr_logits),
            axis=1
        )

        loss_primal = tf.reduce_mean(reconstr_err + Tjoint)
        loss_dual = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=Tjoint, labels=tf.ones_like(Tjoint))
            + tf.nn.sigmoid_cross_entropy_with_logits(logits=Tseperate, labels=tf.zeros_like(Tseperate))
        )

        optimizer_primal = tf.train.AdamOptimizer(2e-5)
        optimizer_dual = tf.train.AdamOptimizer(1e-4)

        qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope +"/encoder")
        pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope +"/decoder")
        dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope + "/discriminator")

        train_op_primal = optimizer_primal.minimize(loss_primal, var_list=pvars+qvars)
        train_op_dual = optimizer_dual.minimize(loss_dual, var_list=dvars)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope))
        ckpt_file = os.path.join(self.ckpt,"vae_%s.ckpt" %scope)
        if train == True:
            # num_turn = x_input.shape[0] / self.batch_size
            start = time.time()
            for i in range(epochs):
                idx = np.random.choice(x_input.shape[0], batch_size, replace=False)
                x_batch = x_input[idx]
                ELBO_out, g_loss = sess.run([loss_primal, train_op_primal], feed_dict={x_real:x_batch})
                g_loss = sess.run(train_op_primal, feed_dict={x_real:x_batch})
                d_loss = sess.run(train_op_dual, feed_dict={x_real:x_batch})
                if i % self.print_size == 0:
                    print("epoches: %d\t g_loss: %f\t d_loss: %f\t time: %d s"%(i, g_loss, d_loss, time.time()-start))

            saver.save(sess, ckpt_file)
        else:
            saver.restore(sess, ckpt_file)


