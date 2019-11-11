import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse
from multiVAE import Translation, read_data, one_hot_vector, calc_recall


def create_dataset(A="Health", B="Clothing"):
    dense_A = read_data("data/%s_%s/%s_user_product.txt" % (A, B, A))
    num_A = 0
    for i in dense_A:
        if num_A < max(i):
            num_A = max(i)
    num_A += 1
    user_A = one_hot_vector(dense_A, num_A)

    dense_B = read_data("data/%s_%s/%s_user_product.txt" % (A, B, B))
    num_B = 0
    for i in dense_B:
        if num_B < max(i):
            num_B = max(i)
    num_B += 1
    user_B = one_hot_vector(dense_B, num_B)

    return user_A, user_B, dense_A, dense_B, num_A, num_B


def main():
    iter = 300
    batch_size= 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/" % (A, B)
    if args.switch:
        user_B, user_A, dense_B, dense_A, num_B, num_A = create_dataset(A, B)
    else:
        user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)

    print(num_A, num_B)
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

    # user_A_val = user_A[train_size:train_size+val_size]
    # user_B_val = user_B[train_size:train_size+val_size]
    # user_A_test = user_A[train_size+val_size:]
    # user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]
    dense_A_val = dense_A[train_size:train_size + val_size]
    dense_B_val = dense_B[train_size:train_size + val_size]
    user_A_val = one_hot_vector([i[:-10] for i in dense_A_val], num_A)
    user_A_test = one_hot_vector([i[:-10] for i in dense_A_test], num_A)
    dense_A_val = [i[-10:] for i in dense_A_val]
    dense_A_test = [i[-10:] for i in dense_A_test]

    print("Train A")
    if A == "Drama" or A=="Romance":
        k = [10, 20, 30, 40, 50]
        encoding_dim = [200, 100]
        decoding_dim = [100, 200, num_A]
    else:
        k = [50, 100, 150, 200, 250, 300]
        encoding_dim = [600, 200]
        decoding_dim = [200, 600, num_A]
    model = Translation(batch_size, num_A, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    max_recall = 0

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x = user_A_train[list_idx]

            feed = {model.x: x}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)


        # Validation Process
        if i%10 == 0:
            model.train = False
            loss_val_a, y_b = sess.run([model.loss, model.x_recon],
                                              feed_dict={model.x:user_A_val})
            print(len(y_b[0]))
            recall = calc_recall(y_b, dense_A_val, [50])
            print("Loss val a: %f, recall %f" % (loss_val_a, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'multi-VAE-model'))
                loss_test_a, y_b= sess.run([model.loss, model.x_recon], feed_dict={model.x: user_A_test})
                print("Loss test a: %f" % (loss_test_a))


                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]
                calc_recall(y_b, dense_A_test, k, "A")
            model.train = True
        if i%100 == 0:
            model.learning_rate /= 2
            print("decrease lr to %f"%model.learning_rate)


    print(max_recall)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k', type=int, default=100, help='top-K')
parser.add_argument('--switch', type=bool, default=False)



if __name__ == '__main__':
    main()



