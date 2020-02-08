import numpy as np


class Dataset:
    def __init__(self, domain_list):
        self.onehot_train, self.onehot_test = [], []
        self.label = []
        self.input_size_list = []
        self.data_test = []
        source_dir = '_'.join(domain_list)

        for i in range(len(domain_list)-1):
            for j in range(i+1, len(domain_list)):
                in_data_train, in_data_test, item_no, in_test = self.read_data("data/%s/%s_%s_user_product.txt"%
                                                   (source_dir, domain_list[i], domain_list[j]))
                if i == 0 and j == 1:
                    self.input_size_list.append(item_no)
                out_data_train, out_data_test, item_no, out_test = self.read_data("data/%s/%s_%s_user_product.txt"%
                                                    (source_dir, domain_list[j], domain_list[i]))
                if i == 0:
                    self.input_size_list.append(item_no)
                self.label.append([i, j])
                self.onehot_train.append([in_data_train, out_data_train])
                self.onehot_test.append([in_data_test, out_data_test])
                self.data_test.append([in_test, out_test])

    def read_data(self, dataset):
        data = list(open(dataset))
        data = [d.strip().split(' ') for d in data]
        data = [[int(i) for i in d[1:]] for d in data]
        item_no = max([max(d) for d in data]) + 1
        one_hot = np.zeros((len(data), item_no))
        for j, d in enumerate(data):
            for i in d:
                one_hot[j, i] = 1

        one_hot_train = one_hot[:int(len(one_hot)*0.8), :]
        one_hot_test = one_hot[int(len(one_hot)*0.8):, :]
        datatest = data[int(len(one_hot)*0.8):]
        return one_hot_train, one_hot_test, item_no, datatest

    def random_iter(self, batch_size):
        domain = []
        ids = []
        for i in range(len(self.label)):
            shuffle_idx = np.random.permutation(range(len(self.onehot_train[i][0])))
            for j in range(len(shuffle_idx)//batch_size+1):
                domain.append(i)
                ids.append(shuffle_idx[i*batch_size:(i+1)*batch_size])
        return domain, ids

    def get_batch_train(self, domain_id, user_ids):
        return (self.onehot_train[domain_id][0][user_ids],
                self.onehot_train[domain_id][1][user_ids],
                self.label[domain_id])

    def get_batch_test(self, domain_id, user_ids):
        # in_domain = [self.data_test[domain_id][0][i] for i in user_ids]
        # out_domain = [self.data_test[domain_id][1][i] for i in user_ids]
        in_domain = self.data_test[0][0]
        out_domain = self.data_test[0][1]
        return (self.onehot_test[domain_id][0][user_ids],
                self.onehot_test[domain_id][1][user_ids],
                self.label[domain_id], in_domain, out_domain)


def calc_recall(pred, test, m=[100], type=None):

    for k in m:
        pred_ab = np.argsort(-pred)[:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
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


    return np.mean(np.array(recall))


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


