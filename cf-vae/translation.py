import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import bottleneck as bn



class Translation:
    def __init__(self, batch_size, dim_A, dim_B, encode_dim_A, decode_dim_A, encode_dim_B, decode_dim_B, adv_dim_A,
                 adv_dim_B, z_dim, share_dim, z_A=None, z_B=None, eps=1e-10, lambda_0=10, lambda_1=0.1, lambda_2=100,
                 lambda_3=0.1,
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
        self.active_function = tf.nn.relu
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x
        # ids = argsort(x_, 1)[::-1][:, :200]
        # if "A" in scope:
        #     #x_ = tf.multiply(tf.expand_dims(self.z_A, 0), tf.expand_dims(x_, 2))
        #     x_ = tf.nn.embedding_lookup(self.z_A, ids)
        # else:
        #     #x_ = tf.multiply(tf.expand_dims(self.z_B, 0), tf.expand_dims(x_, 2))
        #     x_ = tf.nn.embedding_lookup(self.z_B, ids)
        # x_ = flatten(x_)
        # x_ = tf.reshape(x_, (-1, 10000))

        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], self.active_function, scope="enc_%d"%i,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=98765),
                                     weights_regularizer=self.regularizer)
                # x_ = batch_norm(x_, decay=0.995)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.3)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)-1):
                x_ = fully_connected(x_, decode_dim[i], self.active_function, scope="dec_%d" % i,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=98765),
                                     weights_regularizer=self.regularizer)
            if self.train:
                x_ = fully_connected(x_, decode_dim[-1], tf.nn.sigmoid, scope="dec_last", \
                                                                            weights_regularizer=self.regularizer)
            else:
                x_ = fully_connected(x_, decode_dim[-1], scope="dec_last", \
                                     weights_regularizer=self.regularizer)
        return x_

    def adversal(self, x, scope, adv_dim, reuse=False):
        x_ = x

        with tf.variable_scope(scope, reuse=reuse):
            # if self.train:
            #     x_ = tf.nn.dropout(x_, 0.3)
            for i in range(len(adv_dim)-1):
                x_ = fully_connected(x_, adv_dim[i], self.active_function, scope="adv_%d" % i)
            x_ = fully_connected(x_, adv_dim[-1], scope="adv_last")
        return x_

    def share_layer(self, x, scope, dim, reuse=False):
        x_ = x
        # if self.train:
        #     x_ = tf.nn.dropout(x_, 0.3)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(dim)):
                x_ = fully_connected(x_, dim[i], self.active_function, scope="share_%d"%i,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(seed=98765),
                                     weights_regularizer=self.regularizer)
        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim,  scope="z_mu")
            z_sigma = fully_connected(h, self.z_dim,  scope="z_sigma")
            e = tf.random_normal(tf.shape(z_mu))
            if self.train:
                z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
            else:
                z = z_mu
        return z, z_mu, z_sigma

    def encode(self, x, scope, dim, reuse_enc, reuse_share, reuse_z=False):
        h = self.enc(x, "encode_%s"%scope, dim, reuse_enc)
        h = self.share_layer(h, "encode", self.share_dim, reuse_share)
        z, z_mu, z_sigma = self.gen_z(h, "encode", reuse=reuse_z)
        return z, z_mu, z_sigma

    def decode(self, x, scope, dim, reuse_dec, reuse_share):
        y = self.share_layer(x, "decode", self.share_dim[::-1], reuse_share)
        y = self.dec(y, "decode_%s"%scope, dim, reuse_dec)
        return y

    def loss_kl(self, mu, sigma):
        return tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):
        # return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(x, x_recon), axis=1))
        return tf.reduce_mean(tf.abs(x - x_recon))
        # return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x))


    def loss_recsys(self, pred, label):
        return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(label, pred), axis=1))

    def loss_discriminator(self, x, x_fake):
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.ones_like(
            x)))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_fake, labels=tf.zeros_like(
            x_fake)))
        return loss_real + loss_fake
    def loss_generator(self, x):
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=tf.ones_like(
            x)))
        return loss_real

    def build_model(self):
        self.x_A = tf.placeholder(tf.float32, [None, self.dim_A], name='input_A')
        self.x_B = tf.placeholder(tf.float32, [None, self.dim_B], name='input_B')

        x_A = self.x_A
        x_B = self.x_B

        # VAE for domain A
        z_A, z_mu_A, z_sigma_A = self.encode(x_A, "A", self.encode_dim_A, False, False)
        y_AA = self.decode(z_A, "A", self.decode_dim_A, False, False)

        # VAE for domain B
        z_B, z_mu_B, z_sigma_B = self.encode(x_B, "B", self.encode_dim_B, False, True, True)
        y_BB = self.decode(z_B, "B", self.decode_dim_B, False, True)

        # Adversal
        y_BA = self.decode(z_B, "A", self.decode_dim_A, True, True)
        adv_AA = self.adversal(x_A, "adv_A", self.adv_dim_A)
        adv_BA = self.adversal(y_BA, "adv_A", self.adv_dim_A, reuse=True)

        y_AB = self.decode(z_A, "B", self.decode_dim_B, True, True)
        adv_BB = self.adversal(x_B, "adv_B", self.adv_dim_B)
        adv_AB = self.adversal(y_AB, "adv_B", self.adv_dim_B, reuse=True)

        # Cycle - Consistency
        z_ABA, z_mu_ABA, z_sigma_ABA = self.encode(y_AB, "B", self.encode_dim_B, True, True, True)
        y_ABA = self.decode(z_ABA, "A", self.decode_dim_A, True, True)
        z_BAB, z_mu_BAB, z_sigma_BAB = self.encode(y_BA, "A", self.encode_dim_A, True, True, True)
        y_BAB = self.decode(z_BAB, "B", self.decode_dim_B, True, True)


        # Loss VAE
        loss_VAE_A = self.lambda_1 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_2 * self.loss_reconstruct(x_A, y_AA)
        loss_VAE_B = self.lambda_1 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_2 * self.loss_reconstruct(x_B, y_BB)
        self.loss_VAE = loss_VAE_A + loss_VAE_B

        # Loss GAN
        loss_d_A = self.lambda_0 * self.loss_discriminator(adv_AA, adv_BA)
        loss_d_B = self.lambda_0 * self.loss_discriminator(adv_BB, adv_AB)
        self.loss_d= loss_d_A + loss_d_B
        self.adv_AA = adv_AA
        self.adv_AB = adv_BA

        # Loss cycle - consistency (CC)
        loss_CC_A = self.lambda_3 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_3 * self.loss_kl(z_mu_ABA, z_sigma_ABA)\
                    + self.lambda_4 * self.loss_reconstruct(x_A, y_ABA)
        loss_CC_B = self.lambda_3 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_3 * self.loss_kl(z_mu_BAB, z_sigma_BAB)\
                    + self.lambda_4 * self.loss_reconstruct(x_B, y_BAB)

        self.loss_CC = loss_CC_A + loss_CC_B

        self.loss_val_a = self.loss_recsys(x_A, y_BA)
        self.loss_val_b = self.loss_recsys(x_B, y_AB)
        self.y_BA = y_BA
        self.y_AB = y_AB

        self.loss_gen = loss_VAE_A + loss_VAE_B + loss_CC_A + loss_CC_B + tf.losses.get_regularization_loss() + \
                        10*self.loss_generator(y_AB) + \
                        10*self.loss_generator(y_BA)
        self.loss_dis = loss_d_A + loss_d_B
        self.loss_rec = 10 * self.loss_val_a + 10*self.loss_val_b

        # self.train_op_VAE = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_VAE)

        gen_var = [var for var in tf.all_variables() if 'enc' in var.name or 'dec' in var.name]
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.loss_gen, var_list=gen_var)
        adv_varlist = [var for var in tf.all_variables() if 'adv' in var.name]
        print(adv_varlist)
        self.train_op_dis = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.loss_dis,
                                                                                         var_list=adv_varlist)
        self.train_op_rec = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.loss_rec, var_list=gen_var)


def create_dataset(num_A, num_B, A="Health", B="Clothing"):
    dense_A = read_data("data/%s_%s/%s_user_product.txt"%(A,B,A))
    user_A = one_hot_vector(dense_A, num_A)

    dense_B = read_data("data/%s_%s/%s_user_product.txt"%(A, B, B))
    user_B = one_hot_vector(dense_B, num_B)

    return user_A, user_B, dense_A, dense_B

def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f

def read_data2(filename):
    data = list(open(filename).readlines())
    data = data[1:]
    n_data = len(data)
    print(len(data))
    data = [d.strip() for d in data]
    data = [d.split(", ") for d in data]
    data = [d[:3] for d in data]
    data = np.array(data).reshape(n_data, 3).astype(np.int32)
    return data

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j!= num_product:
                one_hot_A[i,j] = 1
    return one_hot_A

def one_hot_vector2(A, num_product):
    one_hot = np.zeros((6557, num_product))
    for i in A:
        one_hot[i[0], i[1]] = i[2]
    return one_hot

def calc_recall(pred, test):
    pred_ab = np.argsort(pred)[:,::-1][:, :100]
    recall = []
    for i in range(len(pred_ab)):
        hits = set(test[i]) & set(pred_ab[i])
        recall_val = float(len(hits)) / len(test[i])
        recall.append(recall_val)
    return np.mean(np.array(recall))

def calc_rmse(pred, test):
    idx = np.where(test != 0)
    pred = pred[idx]
    test = test[idx]
    return np.sqrt(np.mean((test-pred)**2))

def main():
    iter = 3000
    batch_size= 500
    A = "Health"
    B = "Grocery"
    health_num = 15084
    clothing_num = 8364
    encoding_dim_A = [1000, 500]
    encoding_dim_B = [1000, 500]
    share_dim = [100]
    decoding_dim_A = [500, 1000, health_num]
    decoding_dim_B = [500, 1000, clothing_num]
    z_dim = 100
    adv_dim_A = adv_dim_B = [100, 1]
    checkpoint_dir = "translation/%s_%s/"%(A,B)
    user_A, user_B, dense_A, dense_B = create_dataset(health_num, clothing_num, A, B)
    # test_A = list(open("data/Health_Clothing/test_A.txt").readlines())
    # test_A = [t.strip() for t in test_A]
    # if test_A[-1] == '':
    #     test_A = test_A[:-1]
    # test_A = [int(t) for t in test_A]
    # test_B = list(open("data/Health_Clothing/test_B.txt").readlines())
    # test_B = [t.strip() for t in test_B]
    # if test_B[-1] == '':
    #     test_B = test_B[:-1]
    # test_B = [int(t) for t in test_B]
    # z = np.load(os.path.join(checkpoint_dir, "text.npz"))
    # z = z['arr_0']
    # print(z.shape)
    # z_A = z[:health_num]
    # z_B = z[health_num:]
    assert len(user_A) == len(user_B)
    perm = np.random.permutation(len(user_A))
    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    # user_A = user_A[perm]
    # user_B = user_B[perm]

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_val = user_A[train_size:train_size+val_size]
    user_B_val = user_B[train_size:train_size+val_size]
    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]
    # dense_A_test = np.array(dense_A)[test_A]
    # dense_B_test = np.array(dense_B)[test_B]
    # test_A = [t - train_size - val_size for t in test_A]
    # test_B = [t - train_size - val_size for t in test_B]

    model = Translation(batch_size, health_num, clothing_num, encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0
    dense_A_val = dense_A[train_size:train_size+val_size]
    dense_B_val = dense_B[train_size:train_size+val_size]

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x_A = user_A_train[list_idx]
            x_B = user_B_train[list_idx]

            feed = {model.x_A: x_A,
                    model.x_B: x_B}


            _, loss_gen, loss_vae, loss_cc = sess.run([model.train_op_gen, model.loss_gen, model.loss_VAE,
                                                model.loss_CC], feed_dict=feed)
            _, loss_dis, adv_AA, adv_AB = sess.run([model.train_op_dis, model.loss_dis, model.adv_AA, model.adv_AB],
                                        feed_dict=feed)
            # print(adv_AA, adv_AB)
            # _, loss_dis = sess.run([model.train_op_dis, model.loss_dis], feed_dict=feed)
            # _, loss_rec = sess.run([model.train_op_rec, model.loss_rec], feed_dict=feed)

        # print("Loss last batch: loss gen %f, loss dis %f, loss vae %f, loss rec %f, loss cc %f"%(loss_gen, loss_dis,
        #                                                                         loss_vae, loss_rec, loss_cc))

        # Validation Process
        if i%10 == 0:
            model.train = False
            print("Loss last batch: loss gen %f, loss dis %f, loss vae %f,loss cc %f" % (
            loss_gen, loss_dis, loss_vae, loss_cc))
            #                                                                         loss_vae, loss_gan, loss_cc))
            loss_gen, loss_val_a, loss_val_b, y_ba, y_ab = sess.run([model.loss_gen, model.loss_val_a,
                                                                     model.loss_val_b, model.y_BA, model.y_AB],
                                              feed_dict={model.x_A:user_A_val, model.x_B:user_B_val})

            recall = calc_recall(y_ba, dense_A_val) + calc_recall(y_ab, dense_B_val)
            print("Loss gen: %f, Loss val a: %f, Loss val b: %f, recall %f" % (loss_gen, loss_val_a, loss_val_b,
                                                                               recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'), i)
                loss_test_a, loss_test_b, y_ab, y_ba = sess.run(
                    [model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
                 feed_dict={model.x_A: user_A_test, model.x_B: user_B_test})
                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]

                print("recall B: %f" % (calc_recall(y_ab, dense_B_test)))
                print("recall A: %f" % (calc_recall(y_ba, dense_A_test)))


            model.train = True
        if i%50 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)

            # pred = np.array(y_ab).flatten()
            # test = np.array(user_B_val).flatten()
            # rmse = calc_rmse(pred, test)
            # print("Loss val a: %f, Loss val b: %f, rmse %f" % (loss_val_a, loss_val_b, rmse))
            # if rmse < max_recall:
            #     max_recall = rmse
            #     saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'), i)

    print(max_recall)
    # model.train = False
    # loss_test_a, loss_test_b, y_ab, y_ba = sess.run([model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
    #                         feed_dict={model.x_A: user_A_test[200:],model.x_B: user_B_test[200:]})
    # print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))
    # model.train = True
    #
    # dense_A_test = dense_A[(train_size+200):]
    # dense_B_test = dense_B[(train_size+200):]
    #
    #
    # print("recall B: %f"%(calc_recall(y_ab, dense_B_test)))
    # print("recall A: %f" % (calc_recall(y_ba, dense_A_test)))

    # pred_a = np.array(y_ba).flatten()
    # test_a = np.array(user_A_test).flatten()
    # print("rmse A %f"%calc_rmse(pred_a, test_a))
    #
    # pred_a = np.array(y_ab).flatten()
    # test_a = np.array(user_B_test).flatten()
    # print("rmse B %f" % calc_rmse(pred_a, test_a))


if __name__ == '__main__':
    main()



