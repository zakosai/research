import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm
import pickle
import argparse
import os


def calc_recall(pred, test, m=[100], type=None):
    result = {}
    for k in m:
        pred_ab = np.argsort(-pred)[:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
            if len(test[i]) != 0:
                hits = set(test[i]) & set(p)

                #recall
                recall_val = float(len(hits)) / len(test[i])
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
        result['recall@%d'%k] = np.mean(recall)
        result['ndcg@%d'%k] = np.mean(ndcg)


    return np.mean(np.array(recall)), result

def dcg_score(y_true, y_score, k=50):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)
def main():
    args = parser.parse_args()
    f = open(args.data, 'rb')
    dataset = pickle.load(f)

    X = []
    y = []
    for d in dataset['train']:
        X.append({'user': str(d[0]), 'item':str(d[1])})
        y.append(1)
        idx = np.random.randint(0, 100)
        idx = dataset['user_neg'][d[0]][idx]
        X.append({'user': str(d[0]), 'item': str(idx)})
        y.append(0)

    v = DictVectorizer()
    X_train = v.fit_transform(X)
    y_train = np.array(y)


    # Build and train a Factorization Machine
    fm = pylibfm.FM(num_factors=50, num_iter=3000, verbose=True, task="classification", initial_learning_rate=0.001,
                    learning_rate_schedule="optimal")

    fm.fit(X_train,y_train)

    y_pred = []
    for i in dataset['user_item_test'].keys():
        X_test = []
        for j in range(dataset['item_no']):
            X_test.append({'user': str(i), 'item': str(j)})
        X_test = v.transform(X_test)
        predict = fm.predict(X_test)
        y_pred.append(predict)

    recall, _ = calc_recall(np.array(y_pred), dataset['user_item_test'].values(), [50])
    print(recall)
    if recall < 0.1:
        _, result = calc_recall(np.array(y_pred), dataset['user_item_test'].values(), [50, 100, 150, 200, 250, 300])
    else:
        _, result = calc_recall(np.array(y_pred), dataset['user_item_test'].values(), [10, 20, 30, 40, 50, 60])

    f = open(os.path.join(args.ckpt, "result_sum.txt"), "a")
    f.write("Best recall FM: %f\n" % recall)
    np.save(os.path.join(args.ckpt, "result_FM.npy"), result)



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data',  type=str, default="Tool",
                   help='dataset name')
parser.add_argument('--ckpt',  type=str, default="experiment/delicious",
                   help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')


if __name__ == '__main__':
    main()
