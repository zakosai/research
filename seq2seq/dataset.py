import numpy as np
from datetime import datetime

class Dataset(object):
    def __init__(self, n_item, folder, w_size=10, time=False, cat=False):
        self.w_size = w_size

        train_file = "%s/train.txt"%folder
        self.train, self.infer1 = self.read_file(train_file)
        test_file = "%s/test.txt"%folder
        tmp_test, self.infer2 = self.read_file(test_file)

        self.n_user = len(self.train)
        self.n_item = n_item
        self.cat_dim = 18
        self.hybrid = False
        self.time_dim = 23
        self.time = time
        time_test = None
        if self.time:
            time_test = self.create_time(folder)
        self.cat = cat
        if self.cat:
            self.create_item_cat(folder)
        self.create_val_test(tmp_test, time_test)



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


    def create_train_iter(self, text=[]):
        self.X_iter = []
        self.y_iter = []
        self.val2 = []
        self.item_emb = np.zeros((self.n_item, self.n_user))
        if self.time:
            self.time_emb = np.zeros((len(self.train), self.w_size, self.time_dim))
        for i, tr in enumerate(self.train):
            if len(tr) > self.w_size:
                n = np.random.randint(len(tr)-self.w_size)
                self.X_iter.append(tr[n:n+self.w_size])
                self.y_iter.append(tr[n+self.w_size])
            else:
                n = 0
                self.X_iter.append(tr)
                self.y_iter.append(self.infer1[i][0])

            if self.time:
                for j in range(n+1, n+self.w_size+1):
                    self.time_emb[i, j-n-1, :] = self.convert_time(self.time_train[i][j])
            self.item_emb[tr, [i]*len(tr)] = 1
            self.val2.append(tr[-self.w_size:])


        self.X_iter = np.reshape(self.X_iter, (self.n_user, self.w_size))
        self.y_iter = np.array(self.y_iter)
        self.val2 = np.array(self.val2)
        if self.cat:
            self.item_emb = np.concatenate((self.item_emb, self.item_cat), axis=1)
        if self.hybrid:
            self.text = text


    def create_batch(self, idx, X_iter, y_iter, time=None):
        n_batch = len(idx)
        X_batch = np.zeros((n_batch, self.w_size, self.item_emb.shape[1]))
        y_batch = np.zeros((n_batch, self.n_item))
        if self.hybrid:
            t_batch = np.zeros((n_batch, self.w_size, self.text.shape[1]))
        for i in range(n_batch):
            X_batch[i, :, :] = self.item_emb[X_iter[idx[i]]]
            y_batch[i, y_iter[idx[i]]] = 1
            if self.hybrid:
                t_batch[i, :, :] = self.text[X_iter[idx[i]]]

        if self.hybrid:
            return X_batch, y_batch, t_batch
        if self.time:
            X_batch = np.concatenate((X_batch, time[idx]), axis=-1)
        return X_batch, y_batch


    def create_val_test(self, tmp_test, time_test):
        self.val = []
        self.val_infer = []
        self.test = []
        list_u = []
        self.time_emb_val = []
        self.time_emb_test = []
        for i, tr in enumerate(tmp_test):
            if len(tr) > self.w_size+1:
                n = np.random.randint((len(tr)-self.w_size-1))
                self.val.append(tr[n:n + self.w_size])
                self.val_infer.append([tr[n + self.w_size]])
                list_u.append(i)
                if self.time:
                    time = []
                    for j in range(n+1, n+self.w_size+1):
                        time.append(self.convert_time(time_test[i][j]))

                    self.time_emb_val.append(time)
            self.test.append(tr[-self.w_size:])
            if self.time:
                time = []
                for j in range(len(tr)-self.w_size, len(tr)):
                    time.append(self.convert_time(time_test[i][j]))

                self.time_emb_test.append(time)

        self.val = np.reshape(self.val, (len(self.val), self.w_size))
        self.test = np.reshape(self.test, (len(self.test), self.w_size))
        self.list_u = list_u

        if self.time:
            self.time_emb_val = np.reshape(self.time_emb_val,
                                           (len(self.time_emb_val), self.w_size, self.time_dim))
            self.time_emb_test = np.reshape(self.time_emb_test,
                                        (len(self.time_emb_test), self.w_size, self.time_dim))


    def create_item_cat(self, folder):
        item_cat = list(open("%s/categories.txt"%folder))
        item_cat = [i.strip() for i in item_cat]
        item_cat = [i.split(",") for i in item_cat]
        self.item_cat = np.array(item_cat).astype(np.uint8)

    def create_user_info(self, folder):
        user_info = list(open("%s/user_info_train.txt"%folder))
        user_info = [u.strip() for u in user_info]
        user_info = [u.split(",")[1:] for u in user_info]
        self.user_info_train = np.array(user_info).astype(np.float32)
        col = [0] + list(range(6, self.user_info_train.shape[1]-1))
        self.user_info_train = self.user_info_train[:, col]

        user_info = list(open("%s/user_info_test.txt" % folder))
        user_info = [u.strip() for u in user_info]
        user_info = [u.split(",")[1:] for u in user_info]
        self.user_info_test = np.array(user_info).astype(np.float32)
        self.user_info_test = self.user_info_test[:, col]
        self.user_info_val = self.user_info_test[self.list_u]


    def convert_time(self, t):
        time = datetime.fromtimestamp(int(t))
        hour = [0]*4
        weekday = [0]*7
        month = [0]*12

        # hour
        hour[int(time.hour/6)] = 1
        weekday[time.weekday()] = 1
        month[time.month-1] = 1

        time = hour + weekday + month
        return time

    def create_time(self, folder):
        filename = "%s/time_train.txt"%folder
        self.time_train = []
        for line in open(filename):
            a = line.strip().split(",")
            if a == []:
                l = []
            else:
                l = [x for x in a[1:]]
            self.time_train.append(l)

        filename = "%s/time_test.txt" % folder
        time_test = []
        for line in open(filename):
            a = line.strip().split(",")
            if a == []:
                l = []
            else:
                l = [x for x in a[1:]]
            time_test.append(l)
        return time_test





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




