import tensorflow as tf
from tensorflow.contrib.layers import softmax, fully_connected
import tensorflow.keras.backend as K
import numpy as np
import os


class Translation:
    def __init__(self, batch_size, dim_A, dim_B, encode_dim_A, decode_dim_A, encode_dim_B, decode_dim_B, adv_dim_A,
                 adv_dim_B, z_dim, share_dim, eps=1e-10, lambda_0=10, lambda_1=0.1, lambda_2=100, lambda_3=0.1,
                 lambda_4=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.encode_dim_A = encode_dim_A
        self.encode_dim_B = encode_dim_B
        self.decode_dim_A = decode_dim_A
        self.decode_dim_B = decode_dim_B
        self.adv_dim_A = adv_dim_A
        self.adv_dim_B = adv_dim_B
        self.z_dim = z_dim
        self.eps = eps
        self.share_dim = share_dim
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.learning_rate = learning_rate

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], softmax, scope="enc_%d"%i)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)):
                x_ = fully_connected(x_, decode_dim[i], softmax, scope="dec_%d" % i)
        return x_

    def adversal(self, x, scope, adv_dim, reuse=False):
        x_ = x
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(adv_dim)):
                x_ = fully_connected(x_, adv_dim[i], softmax, scope="adv_%d" % i)
        return x_

    def share_layer(self, x, scope, dim, reuse=False):
        x_ = x
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(dim)):
                x_ = fully_connected(x_, dim[i], softmax, scope="share_%d"%i)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, softmax, scope="z_mu")
            z_sigma = fully_connected(h, self.z_dim, softmax, scope="z_sigma")
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def encode(self, x, scope, dim, reuse_enc, reuse_share):
        h = self.enc(x, "encode_%s"%scope, dim, reuse_enc)
        h = self.share_layer(h, "encode", self.share_dim, reuse_share)
        z, z_mu, z_sigma = self.gen_z(h, "VAE_%s"%scope)
        return z, z_mu, z_sigma

    def decode(self, x, scope, dim, reuse_dec, reuse_share):
        y = self.share_layer(x, "decode", self.share_dim, reuse_dec)
        y = self.decode(y, "decode_%s"%scope, dim, reuse_share)
        return y

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):
        return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(x, x_recon), axis=1))

    def loss_GAN(self, x, x_fake):
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.ones_like(
            x)))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_fake, labels=tf.zeros_like(
            x_fake)))
        return loss_real + loss_fake

    def build_model(self):
        self.x_A = tf.placeholder(tf.float32, [None, self.dim_A], name='input_A')
        self.x_B = tf.placeholder(tf.float32, [None, self.dim_B], name='input_B')

        x_A = self.x_A
        x_B = self.x_B

        # VAE for domain A
        z_A, z_mu_A, z_sigma_A = self.encode(x_A, "A", self.encode_dim_A, False, False)
        y_AA = self.decode(z_A, "A", self.decode_dim_A, False, False)

        # VAE for domain B
        z_B, z_mu_B, z_sigma_B = self.encode(x_B, "B", self.encode_dim_B, False, True)
        y_BB = self.decode(z_B, "B", self.decode(z_B, "B", self.decode_dim_B, False, True))

        # Adversal
        y_BA = self.decode(z_B, "A", self.decode_dim_A, True, True)
        adv_AA = self.adversal(y_AA, "adv_A", self.adv_dim_A)
        adv_BA = self.adversal(y_BA, "adv_A", self.adv_dim_A)

        y_AB = self.decode(z_A, "B", self.decode_dim_B, True, True)
        adv_BB = self.adversal(y_BB, "adv_B", self.adv_dim_B)
        adv_AB = self.adversal(y_AB, "adv_B", self.adv_dim_B, reuse=True)

        # Cycle - Consistency
        z_ABA, z_mu_ABA, z_sigma_ABA = self.encode(y_AB, "B", self.encode_dim_B, True, True)
        y_ABA = self.decode(z_ABA, "A", self.decode_dim_A, True, True)
        z_BAB, z_mu_BAB, z_sigma_BAB = self.encode(y_BA, "A", self.encode_dim_A, True, True)
        y_BAB = self.decode(z_BAB, "B", self.decode_dim_B, True, True)


        # Loss VAE
        loss_VAE_A = self.lambda_1 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_2 * self.loss_reconstruct(x_A, y_AA)
        loss_VAE_B = self.lambda_1 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_2 * self.loss_reconstruct(x_B, y_BB)

        # Loss GAN
        loss_GAN_A = self.lambda_0 * self.loss_GAN(adv_AA, adv_BA)
        loss_GAN_B = self.lambda_0 * self.loss_GAN(adv_BB, adv_AB)

        # Loss cycle - consistency (CC)
        loss_CC_A = self.lambda_3 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_3 * self.loss_kl(z_mu_ABA, z_sigma_ABA)\
                    + self.lambda_4 * self.loss_reconstruct(x_A, y_ABA)
        loss_CC_B = self.lambda_3 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_3 * self.loss_kl(z_mu_BAB, z_sigma_BAB)\
                    + self.loss_reconstruct(x_B, y_BAB)

        self.loss_val_a = self.loss_reconstruct(x_A, y_BA)
        self.loss_val_b = self.loss_reconstruct(x_B, y_AB)

        self.loss_gen = loss_VAE_A + loss_VAE_B + loss_GAN_A + loss_GAN_B + loss_CC_A + loss_CC_B
        self.loss_dis = loss_GAN_A + loss_GAN_B

        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_gen)
        self.train_op_dis = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_dis)


def create_dataset(num_A, num_B):


    file_A = read_data("data/Health_Clothing/Health_user_product.txt")
    user_A = one_hot_vector(file_A, num_A)

    file_B = read_data("data/Health_Clothing/Clothing_user_product.txt")
    user_B = one_hot_vector(file_B, num_B)

    return user_A, user_B

def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            one_hot_A[i,j] = 1
    return one_hot_A

def main():
    iter = 5000
    batch_size= 500
    clothing_num = 18226
    health_num = 16069
    encoding_dim_A = encoding_dim_B = [1000, 500]
    share_dim = [100]
    decoding_dim_A = [500, 1000, health_num]
    decoding_dim_B = [500, 1000, clothing_num]
    z_dim = 50
    adv_dim_A = adv_dim_B = [200, 100, 1]
    checkpoint_dir = "translation/experiment/exp1/"
    user_A, user_B = create_dataset(health_num, clothing_num)

    assert len(user_A) == len(user_B)
    perm = np.random.permutation(len(user_A))
    train_size = 6000

    user_A = user_A[perm]
    user_B = user_B[perm]

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_test = user_A[train_size:]
    user_B_test = user_B[train_size:]

    model = Translation(batch_size, health_num, clothing_num, encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    for i in range(iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size) + 1):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x_A = user_A_train[list_idx]
            x_B = user_B_train[list_idx]

            feed = {model.x_A: x_A,
                    model.x_B: x_B}

            _, loss_gen = sess.run([model.train_op_gen, model.loss_gen], feed_dict=feed)
            _, loss_dis = sess.run([model.train_op_dis, model.loss_dis], feed_dict=feed)

        print("Loss last batch: loss gen %f, loss dis %f"%(loss_gen, loss_dis))

        # Validation Process
        if i%100 == 0:
            loss_val_a, loss_val_b = sess.run([model.loss_val_a, model.loss_val_b], feed_dict={model.x_A:user_A_test[:
                200], model.x_B:user_B_test[:200]})
            print("Loss val a: %f, Loss val b: %f"%(loss_val_a, loss_val_b))
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'), i)

    loss_test_a, loss_test_b = sess.run([model.loss_val_a, model.loss_val_b], feed_dict={model.x_A: user_A_test[200:],
                                                                                       model.x_B: user_B_test[200:]})
    print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

if __name__ == '__main__':
    main()




