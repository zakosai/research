from translation import Translation, create_dataset, calc_recall
import tensorflow as tf
import numpy as np
import argparse
import os


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
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    saver.restore(sess, os.path.join(checkpoint_dir, args.ckpt))
    loss_test_a, loss_test_b, y_ab, y_ba = sess.run(
        [model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
        feed_dict={model.x_A: user_A_test, model.x_B: user_B_test})
    print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))
    calc_recall(y_ba, dense_A_test, k, type="A")
    calc_recall(y_ab, dense_B_test, k, type="B")
    pred = np.argsort(-y_ba)[:, :10]
    f = open(os.path.join(checkpoint_dir, "predict_%s.txt" % A), "w")
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





parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k',  type=int, default=100,
                   help='top-K')
parser.add_argument('--ckpt',  type=str, default="translation",
                   help='top-K')
if __name__ == '__main__':
    main()
