import tensorflow as tf
import numpy as np
import os
import argparse
from translation import Translation, create_dataset, dcg_score, one_hot_vector


def calc_recall(pred, test, m=[100], type=None, n_predict=5):
    for k in m:
        recall = []
        ndcg = []
        for i in range(len(pred)):
            train_user_i = test[i][:-n_predict]
            test_user_i = test[i][-n_predict:]
            pred[i, train_user_i] = np.min(pred[i])
            p = np.argsort(-pred[i])[:k]
            hits = set(test_user_i) & set(p)

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


def main(args):
    iter = 50
    batch_size= 500

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

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]
    dense_A_val = dense_A[train_size:train_size+val_size]
    dense_B_val = dense_B[train_size:train_size+val_size]

    user_A_val = one_hot_vector([i[:-args.n_predict] for i in dense_A_val], num_A)
    user_A_test = one_hot_vector([i[:-args.n_predict] for i in dense_A_test], num_A)
    user_B_val = one_hot_vector([i[:-args.n_predict] for i in dense_B_val], num_B)
    user_B_test = one_hot_vector([i[:-args.n_predict] for i in dense_B_test], num_B)

    model = Translation(batch_size, num_A, num_B, encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim, learning_rate=1e-3, lambda_2=1,
                        lambda_4=0.1)
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
            x_A = user_A_train[list_idx]
            x_B = user_B_train[list_idx]

            feed = {model.x_A: x_A,
                    model.x_B: x_B}

            if i <20:
                _, loss_vae = sess.run([model.train_op_VAE_A, model.loss_VAE], feed_dict=feed)
                _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
                loss_gen = loss_dis = loss_cc = 0
            # elif i>=50 and i < 100:
            #     _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
            #     loss_gen = loss_dis = loss_cc = 0
            else:
                model.freeze = False
                _, loss_gen, loss_vae, loss_cc = sess.run([model.train_op_gen, model.loss_gen, model.loss_VAE,
                                                        model.loss_CC], feed_dict=feed)

                sess.run([model.train_op_dis_A],feed_dict=feed)
                # _, loss_gen, loss_vae, loss_cc = sess.run([model.train_op_gen_B, model.loss_gen, model.loss_VAE,
                #                                            model.loss_CC], feed_dict=feed)
                sess.run([model.train_op_dis_B], feed_dict=feed)
                loss_dis = 0
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
            loss_gen, loss_val_a, loss_val_b, y_ba, y_aa, y_ab, y_bb = sess.run([model.loss_gen, model.loss_val_a,
                                                                     model.loss_val_b, model.y_BA, model.y_AA, model.y_AB, model.y_BB],
                                              feed_dict={model.x_A:user_A_val, model.x_B:user_B_val})
            y_ab = y_bb + y_ab
            y_ba = y_aa + y_ba
            recall = calc_recall(y_ba, dense_A_val, [50], args.n_predict) + \
                     calc_recall(y_ab, dense_B_val, [50], args.n_predict)
            print("Loss gen: %f, Loss val a: %f, Loss val b: %f, recall %f" % (loss_gen, loss_val_a, loss_val_b,
                                                                               recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'))
                loss_test_a, loss_test_b, y_ab, y_ba, y_aa, y_bb = sess.run(
                    [model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA, model.y_AA, model.y_BB],
                 feed_dict={model.x_A: user_A_test, model.x_B: user_B_test})

                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]

                calc_recall(y_ba, dense_A_test, k, type="A", n_predict=args.n_predict)
                calc_recall(y_ab, dense_B_test, k, type="B", n_predict=args.n_predict)
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
        if i%50 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)

    print(max_recall)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--A', type=str, default="Health", help='domain A')
    parser.add_argument('--B', type=str, default='Grocery', help='domain B')
    parser.add_argument('--k', type=int, default=100, help='top-K')
    parser.add_argument('--n_predict', type=int, default=5, help='prediction number')
    args = parser.parse_args()
    main(args)
