import pickle
import numpy as np

class Dataset(object):
    def __init__(self, folder, seq_len, swap=False):
        self.n_user = len(list(open("%s/user.txt"%folder)))
        self.n_item_A = len(list(open("%s/itemA.txt"%folder)))
        self.n_item_B = len(list(open("%s/itemB.txt"%folder)))
        self.dataset = pickle.load(open("%s/dataset.obj"%folder, "rb"))

        if swap:
            tmp = self.n_item_A
            self.n_item_A = self.n_item_B
            self.n_item_B = tmp
            tmp = self.dataset['rating_A']
            self.dataset['rating_A'] = self.dataset['rating_B']
            self.dataset['rating_B'] = tmp
            tmp = 0

        self.seq_len = seq_len
        self.max_target_sequence = max([len(i) for i in self.dataset['rating_B']])

        self.eos_A = self.n_item_A
        self.eos_B = self.n_item_B
        self.go = self.n_item_B
        self.n_item_B += 2
        self.n_item_A += 1

        self.emb_A = self.create_emb(self.n_item_A, self.dataset['rating_A'])
        self.emb_B = self.create_emb(self.n_item_B, self.dataset['rating_B'], 1)


    def create_emb(self, n_item, rating, type=0):
        emb = np.zeros((n_item, self.n_user))

        for i, r in enumerate(rating):
            for j in range(len(r)-type):
                emb[i, j] = 1
        return emb


    def create_batch(self, idx):
        target_batch = []
        input_emb_batch = []
        target_emb_batch = []
        target_sequence = []
        max_input_length = max([len(self.dataset['rating_A'][i]) for i in idx])
        max_target_legth = max([len(self.dataset['rating_B'][i]) for i in idx])

        for i in idx:
            tmp_input = self.dataset['rating_A'][i]
            tmp_input = tmp_input + [self.eos_A]*(max_input_length - len(tmp_input))
            input_emb_batch.append(self.emb_A[tmp_input])

            tmp_target = self.dataset['rating_B'][i]
            target_sequence.append(len(tmp_target))
            tmp_target = tmp_target + [self.eos_B]*(max_target_legth - len(tmp_target))
            target_batch.append(tmp_target)
            tmp_target = [self.go] + tmp_target[:-1]

            # for k, j in enumerate(tmp_target):
            #     try:
            #         tmp = self.emb_B[j]
            #     except:
            #         print(j, k)
            #         print(i)
            target_emb_batch.append(self.emb_B[tmp_target])


        return (np.array(input_emb_batch).reshape((len(idx), max_input_length, self.n_user)),
                np.array(target_batch).reshape((len(idx), max_target_legth)),
                np.array(target_emb_batch).reshape((len(idx), max_target_legth, self.n_user)),
                target_sequence)


def calc_recall(pred, train, test, k=10, type=None):
    pred_ab = np.argsort(-pred)
    recall = []
    ndcg = []
    hit = 0
    for i in range(len(pred_ab)):
        p = pred_ab[i, :k + len(train[i])]
        p = p.tolist()
        for u in train[i]:
            if u in p:
                p.remove(u)
        p = p[:k]
        hits = set(test[i]) & set(p)
        # print(test[i], p, hits)

        # recall
        recall_val = float(len(hits)) / len(test[i])
        recall.append(recall_val)

        # hit
        hits_num = len(hits)
        if hits_num > 0:
            hit += 1

        # ncdg
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

    # print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))

    return np.mean(np.array(recall)), float(hit) / len(pred_ab), np.mean(ndcg)

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






