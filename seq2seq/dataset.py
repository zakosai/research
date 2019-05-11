import numpy as np

class Dataset(object):
    def __init__(self, n_item, folder):
        self.w_size = 10

        train_file = "%s/train.txt"%folder
        self.train, self.infer1 = self.read_file(train_file)
        test_file = "%s/test.txt"%folder
        tmp_test, self.infer2 = self.read_file(test_file)
        self.create_val_test(tmp_test)

        self.n_user = len(self.train)
        self.n_item = n_item


    def read_file(self, filename):
        train = []
        infer = []
        for line in open(filename):
            a = line.strip().split()
            if a == []:
                l = []
            else:
                l = [int(x) for x in a[1:-1]]
            train.append(l)
            infer.append([int(a[-1])])
        return train, infer


    def create_train_iter(self):
        self.X_iter = []
        self.y_iter = []
        self.item_emb = np.zeros((self.n_item, self.n_user))
        for i, tr in enumerate(self.train):
            n = np.random.randint(len(tr)-self.w_size-1)
            self.X_iter.append(tr[n:n+self.w_size])
            self.y_iter.append(tr[n+self.w_size])
            self.item_emb[tr, [i]*len(tr)] = 1

        self.X_iter = np.reshape(self.X_iter, (self.n_user, self.w_size))
        self.y_iter = np.array(self.y_iter)


    def create_batch(self, idx, X_iter, y_iter):
        n_batch = len(idx)
        X_batch = np.zeros((n_batch, self.w_size, self.item_emb.shape[1]))
        y_batch = np.zeros((n_batch, self.n_item))
        for i in range(n_batch):
            X_batch[i, :, :] = self.item_emb[X_iter[idx[i]]]
            y_batch[i, y_iter[idx[i]]] = 1

        return X_batch, y_batch



    def create_val_test(self, tmp_test):
        self.val = []
        self.val_infer = []
        self.test = []
        for i, tr in enumerate(tmp_test):
            n = np.random.randint((len(tr)-self.w_size-1))
            self.val.append(tr[n:n+self.w_size])
            self.val_infer.append([tr[n+self.w_size]])
            self.test.append(tr[-self.w_size:])

        self.val = np.reshape(self.val, (len(self.val), self.w_size))
        self.test = np.reshape(self.test, (len(self.test), self.w_size))




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



