from multiVAE import Translation, create_dataset, calc_recall, one_hot_vector
import tensorflow as tf
import numpy as np
import argparse
import os
from multi_VAE_single import calc_recall_same_domain


def main():
    iter = 300
    batch_size = 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/" % (A, B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)

    print(num_A, num_B)
    if A == "Drama" or A == "Romance":
        k = [10, 20, 30, 40, 50]
        encoding_dim = [200, 100]
        decoding_dim = [100, 200, num_A + num_B]
    else:
        k = [50, 100, 150, 200, 250, 300]
        encoding_dim = [600, 200]
        decoding_dim = [200, 600, num_A + num_B]

    z_dim = 50

    perm = np.random.permutation(len(user_A))
    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    # user_A = user_A[perm]
    # user_B = user_B[perm]
    user_A = np.array(user_A)
    user_B = np.array(user_B)

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_val = user_A[train_size:train_size + val_size]
    user_B_val = user_B[train_size:train_size + val_size]
    user_A_test = user_A[train_size + val_size:]
    user_B_test = user_B[train_size + val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]

    user_train = np.concatenate((user_A_train, user_B_train), axis=1)
    user_val_A = np.concatenate((user_A_val, np.zeros(shape=user_B_val.shape)), axis=1)
    print(user_val_A.shape)
    user_val_B = np.concatenate((np.zeros(shape=user_A_val.shape), user_B_val), axis=1)
    user_test_A = np.concatenate((user_A_test, np.zeros(shape=user_B_test.shape)), axis=1)
    user_test_B = np.concatenate((np.zeros(shape=user_A_test.shape), user_B_test), axis=1)

    train_A_same_domain = one_hot_vector([i[:-args.n_predict] for i in dense_A_test], num_A)
    train_B_same_domain = one_hot_vector([i[:-args.n_predict] for i in dense_B_test], num_B)
    train_A_same_domain = np.concatenate((train_A_same_domain, np.zeros_like(train_B_same_domain)), axis=1)
    train_B_same_domain = np.concatenate((np.zeros((train_B_same_domain.shape[0], num_A)), train_B_same_domain), axis=1)

    # dense_A_test = np.array(dense_A)[test_A]
    # dense_B_test = np.array(dense_B)[test_B]
    # test_A = [t - train_size - val_size for t in test_A]
    # test_B = [t - train_size - val_size for t in test_B]

    model = Translation(batch_size, num_A + num_B, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    saver.restore(sess, os.path.join(checkpoint_dir, args.ckpt))
    loss_test_a, y_b = sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_A})
    loss_test_b, y_a = sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_B})
    print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

    calc_recall(y_a[:, :num_A], dense_A_test, k, "A")
    calc_recall(y_b[:, num_A:], dense_B_test, k, "B")
    y_aa = sess.run(model.x_recon, feed_dict={model.x: train_A_same_domain})
    y_bb = sess.run(model.x_recon, feed_dict={model.x: train_B_same_domain})
    calc_recall_same_domain(y_aa[:, :num_A], dense_A_test, k, "same A")
    calc_recall_same_domain(y_bb[:, num_A:], dense_B_test, k, "same B")
    pred = np.argsort(-y_a[:, :num_A])[:, :10]
    f = open(os.path.join(checkpoint_dir, "predict_%s_multiVAE.txt" % A), "w")
    for p in pred:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()
    pred = np.argsort(-y_b[:, num_A:])[:, :10]
    f = open(os.path.join(checkpoint_dir, "predict_%s_multiVAE.txt" % B), "w")
    for p in pred:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()

    pred = np.argsort(-y_aa[:, :num_A])[:, :10]
    f = open(os.path.join(checkpoint_dir, "predict_%s_multiVAE_samedomain.txt" % A), "w")
    for p in pred:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()
    pred = np.argsort(-y_bb[:, num_A:])[:, :10]
    f = open(os.path.join(checkpoint_dir, "predict_%s_multiVAE_samedomain.txt" % B), "w")
    for p in pred:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k',  type=int, default=100,
                   help='top-K')
parser.add_argument('--ckpt',  type=str, default="multi-VAE-model",
                   help='top-K')
parser.add_argument('--n_predict',  type=int, default=5,
                   help='num predict')
if __name__ == '__main__':
    main()
