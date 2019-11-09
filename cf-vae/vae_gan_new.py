import tensorflow as tf
import numpy as np
import os
import argparse
from translation import Translation, create_dataset, calc_recall


def loss_discriminator(A, B):
    loss_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A, tf.zeros_like(A)))
    loss_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(B, tf.ones_like(B)))
    return loss_A + loss_B


def build_model(d2d):
    d2d.x_A = tf.placeholder(tf.float32, [None, d2d.dim_A], name='input_A')
    d2d.x_B = tf.placeholder(tf.float32, [None, d2d.dim_B], name='input_B')

    x_A = d2d.x_A
    x_B = d2d.x_B

    # VAE for domain A
    z_A, z_mu_A, z_sigma_A = d2d.encode(x_A, "A", d2d.encode_dim_A, False, False)
    y_AA = d2d.decode(z_A, "A", d2d.decode_dim_A, False, False)

    # VAE for domain B
    z_B, z_mu_B, z_sigma_B = d2d.encode(x_B, "B", d2d.encode_dim_B, False, True, True)
    y_BB = d2d.decode(z_B, "B", d2d.decode_dim_B, False, True)

    # Fake
    d2d.y_AB = d2d.decode(z_A, "B", d2d.decode_dim_B, True, True)
    d2d.y_BA = d2d.decode(z_B, "A", d2d.decode_dim_A, True, True)


    # Loss VAE
    loss_VAE_A = d2d.lambda_1 * d2d.loss_kl(z_mu_A, z_sigma_A) + d2d.lambda_2 * d2d.loss_reconstruct(x_A, y_AA) +\
        d2d.lambda_2 * d2d.loss_reconstruct(x_B, d2d.y_AB)
    loss_VAE_B = d2d.lambda_1 * d2d.loss_kl(z_mu_B, z_sigma_B) + d2d.lambda_2 * d2d.loss_reconstruct(x_B, y_BB) +\
        d2d.lambda_2 * d2d.loss_reconstruct(x_A, d2d.y_BA)
    d2d.loss_VAE = loss_VAE_A + loss_VAE_B

    # GAN
    av_A = d2d.adversal(z_A, [20,1], scope="adv")
    av_B = d2d.adversal(z_B, [20, 1], scope="adv", reuse=True)

    # Loss GAN
    d2d.loss_gen = loss_discriminator(av_B, av_A)
    d2d.loss_dis = loss_discriminator(av_A, av_B)

    adv_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv")
    vae_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vae")
    d2d.train_op_gen = tf.train.AdamOptimizer(d2d.learning_rate).minimize(d2d.loss_gen + d2d.loss_VAE,
                                                                                     var_list=vae_vars)
    d2d.train_op_dis = tf.train.AdamOptimizer(d2d.learning_rate).minimize(d2d.loss_dis,
                                                                              var_list=adv_vars)

def main():
    iter = 300
    batch_size= 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/"%(A,B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)
    z_dim = 50
    adv_dim_A = adv_dim_B = [100, 1]

    if A == "Drama" or A == "Romance":
        k = [10, 20, 30, 40, 50]
        dim = 200
        share = 100
    else:
        k = [50, 100, 150, 200, 250, 300]
        dim = 600
        share = 200

    print(k)

    encoding_dim_A = [dim]
    encoding_dim_B = [dim]
    share_dim = [share]
    decoding_dim_A = [dim, num_A]
    decoding_dim_B = [dim, num_B]


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


    model = Translation(batch_size, num_A, num_B, encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim, learning_rate=1e-3, lambda_2=1,
                        lambda_4=0.1)
    build_model(model)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
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

            if i <50:
                _, loss_vae = sess.run([model.train_op_VAE_A, model.loss_VAE], feed_dict=feed)
                _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
                loss_gen = loss_dis = loss_cc = 0
            # elif i>=50 and i < 100:
            #     _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
            #     loss_gen = loss_dis = loss_cc = 0
            else:
                model.freeze = False
                _, loss_gen, loss_vae = sess.run([model.train_op_gen, model.loss_gen, model.loss_VAE], feed_dict=feed)
                _, loss_dis = sess.run([model.train_op_dis, model.loss_dis], feed_dict=feed)

        if i%10 == 0:
            model.train = False
            print("Loss last batch: loss gen %f, loss dis %f, loss vae %f" % (loss_gen, loss_dis, loss_vae))
            loss_gen, loss_val_a, loss_val_b, y_ba, y_ab = sess.run([model.loss_gen, model.loss_val_a,
                                                                     model.loss_val_b, model.y_BA, model.y_AB],
                                              feed_dict={model.x_A:user_A_val, model.x_B:user_B_val})


            recall = calc_recall(y_ba, dense_A_val, [50]) + calc_recall(y_ab, dense_B_val, [50])
            print("Loss gen: %f, Loss val a: %f, Loss val b: %f, recall %f" % (loss_gen, loss_val_a, loss_val_b,
                                                                               recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'))
                loss_test_a, loss_test_b, y_ab, y_ba = sess.run(
                    [model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
                 feed_dict={model.x_A: user_A_test, model.x_B: user_B_test})
                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]

                calc_recall(y_ba, dense_A_test, k, type="A")
                calc_recall(y_ab, dense_B_test, k, type="B")
                pred = np.argsort(-y_ba)[:, :10]
                f = open(os.path.join(checkpoint_dir, "predict_%s.txt"%A), "w")
                for p in pred:
                    w = [str(i) for i in p]
                    f.write(','.join(w))
                    f.write("\n")
                f.close()
                pred = np.argsort(-y_ab)[:, :10]
                f = open(os.path.join(checkpoint_dir, "predict_%s.txt" % B), "w")
                for p in pred:
                    w = [str(i) for i in p]
                    f.write(','.join(w))
                    f.write("\n")
                f.close()

            model.train = True
        if i%100 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)

    print(max_recall)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k',  type=int, default=100,
                   help='top-K')
if __name__ == '__main__':
    main()



