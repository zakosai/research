from translation2 import Translation, create_dataset, dcg_score, calc_recall_same_domain, one_hot_vector
import tensorflow as tf
import numpy as np
import argparse
import os


def main():
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

    encoding_dim_A = [600]
    encoding_dim_B = [600]
    share_dim = [200]
    decoding_dim_A = [600, num_A]
    decoding_dim_B = [600, num_B]

    assert len(user_A) == len(user_B)
    perm = np.random.permutation(len(user_A))
    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]

    train_A_same_domain = [i[:-5] for i in dense_A_test]
    train_A_same_domain = one_hot_vector(train_A_same_domain, num_A)
    train_B_same_domain = [i[:-5] for i in dense_B_test]
    train_B_same_domain = one_hot_vector(train_B_same_domain, num_B)

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
    jaccard = np.matmul(user_A[:train_size].T, user_B[:train_size])

    calc_recall(y_ba, dense_A_test, dense_B_test, jaccard.T, k, type="A")
    calc_recall(y_ab, dense_B_test, dense_A_test, jaccard, k, type="B")

    y_aa, y_bb = sess.run([model.y_AA, model.y_BB],
                          feed_dict={model.x_A: train_A_same_domain, model.x_B: train_B_same_domain})
    recall_aa, predict_A = calc_recall_same_domain(y_aa, dense_A_test, [50], type="A")
    recall_bb, predict_B = calc_recall_same_domain(y_bb, dense_B_test, [50], type="B")
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

    f = open(os.path.join(checkpoint_dir, "predict_%s_samedomain.txt" % A), "w")
    for p in predict_A:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()
    f = open(os.path.join(checkpoint_dir, "predict_%s_samedomain.txt" % B), "w")
    for p in predict_B:
        w = [str(i) for i in p]
        f.write(','.join(w))
        f.write("\n")
    f.close()


def calc_recall(pred, test, train, jaccard, m=[100], type=None):

    for k in m:
        recall = []
        ndcg = []
        for i in range(len(pred)):
            similar_score = jaccard[train[i], :].sum(axis=0)
            p = np.argsort(-pred[i] * similar_score)[:k]

            hits = set(test[i]) & set(p)

            #recall
            recall_val = float(len(hits)) / len(test[i])
            # if recall_val > 0.5:
            #     print(i, p, hits, type)
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

        print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))
    return np.mean(np.array(recall))


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
