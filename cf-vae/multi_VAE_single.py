import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse
from multiVAE import Translation, read_data, one_hot_vector, dcg_score, calc_recall


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


def calc_recall_same_domain(pred, test, m=[100], type=None, f=None):
    for k in m:
        pred_ab = np.argsort(-pred)
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            num_train = -5
            u_train = test[i][:num_train]
            u_test = test[i][num_train:]
            p = list(pred_ab[i, :(k + len(u_train))])
            for t in u_train:
                if t in p:
                    p.remove(t)
            p = p[:k]

            hits = set(u_test) & set(p)

            #recall
            recall_val = float(len(hits)) / len(u_test)
            recall.append(recall_val)

            #ncdg
            score = []
            for j in range(k):
                if p[j] in hits:
                    score.append(1)
                else:
                    score.append(0)
            actual = dcg_score(score, pred[i, p], k)
            best = dcg_score(score, score, k)
            if best == 0:
                ndcg.append(0)
            else:
                ndcg.append(float(actual) / best)

        if f != None:
            f.write("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))
        else:
            print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))
    return np.mean(np.array(recall))


def main(args):
    iter = 300
    batch_size= 500
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/" % (A, B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)

    print(num_A, num_B)
    if A == "Drama" or A=="Romance":
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

    user_A_val = user_A[train_size:train_size+val_size]
    user_B_val = user_B[train_size:train_size+val_size]
    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]

    user_train = np.concatenate((user_A_train, user_B_train), axis=1)
    user_val_A = np.concatenate((user_A_val, np.zeros(shape=user_B_val.shape)), axis=1)
    print(user_val_A.shape)
    user_val_B = np.concatenate((np.zeros(shape=user_A_val.shape), user_B_val), axis=1)
    user_test_A = np.concatenate((user_A_test, np.zeros(shape=user_B_test.shape)), axis=1)
    user_test_B = np.concatenate((np.zeros(shape=user_A_test.shape), user_B_test), axis=1)

    model = Translation(batch_size, num_A + num_B, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    max_recall = 0
    dense_A_val = dense_A[train_size:train_size+val_size]
    dense_B_val = dense_B[train_size:train_size+val_size]

    train_A_same_domain = one_hot_vector([i[:-args.n_predict] for i in dense_A_test], num_A)
    train_B_same_domain = one_hot_vector([i[:-args.n_predict] for i in dense_B_test], num_B)
    train_A_same_domain = np.concatenate((train_A_same_domain, np.zeros_like(train_B_same_domain)), axis=1)
    train_B_same_domain = np.concatenate((np.zeros((train_B_same_domain.shape[0], num_A)),
                                          train_B_same_domain), axis=1)

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x = user_train[list_idx]

            feed = {model.x: x}

            _, loss = sess.run([model.train_op, model.loss], feed_dict=feed)

        # Validation Process
        if i%10 == 0:
            model.train = False
            loss_val_a, y_b = sess.run([model.loss, model.x_recon],
                                              feed_dict={model.x:user_val_A})
            loss_val_b, y_a = sess.run([model.loss, model.x_recon],
                                       feed_dict={model.x: user_val_B})
            print(len(y_a[0]), len(y_b[0]))
            recall = calc_recall(y_b[:, num_A:], dense_B_val, [50]) + calc_recall(y_a[:, :num_A], dense_A_val, [50])
            print("Loss val a: %f, Loss val b: %f, recall %f" % (loss_val_a, loss_val_b, recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'multi-VAE-model'))
                loss_test_a, y_b = sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_A})
                loss_test_b, y_a = sess.run([model.loss, model.x_recon], feed_dict={model.x: user_test_B})
                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))
                calc_recall(y_a[:, :num_A], dense_A_test, k, "diff A")
                calc_recall(y_b[:, num_A:], dense_B_test, k, "diff B")
                # print("Recall in different domain: A:%f, B:%f" % (recall_ba, recall_ab))

                y_aa = sess.run(model.x_recon, feed_dict={model.x: train_A_same_domain})
                y_bb = sess.run(model.x_recon, feed_dict={model.x: train_B_same_domain})
                calc_recall_same_domain(y_aa[:, :num_A], dense_A_test, k, "same A")
                calc_recall_same_domain(y_bb[:, num_A:], dense_B_test, k, "same B")

            model.train = True
        if i%100 == 0:
            model.learning_rate /= 2
            print("decrease lr to %f"%model.learning_rate)


    print(max_recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--A', type=str, default="Health", help='domain A')
    parser.add_argument('--B', type=str, default='Grocery', help='domain B')
    parser.add_argument('--k', type=int, default=100, help='top-K')
    parser.add_argument('--switch', type=bool, default=False)
    parser.add_argument('--n_predict', type=int, default=5)
    args = parser.parse_args()

    main(args)



