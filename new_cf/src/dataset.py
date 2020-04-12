import numpy as np
from scipy.sparse import load_npz
import pandas as pd
import os


class Dataset:
    def __init__(self, data_dir, data_type):
        data = self.load_cvae_data(data_dir, data_type)
        self.item_info = data['train_items']
        self.user_info = data['train_users']
        self.train = data['train_users']
        self.test = data['test_users']
        self.no_item, self.item_size, = self.item_info.shape
        self.no_user, self.user_size = self.user_info.shape

        cf_in = self.gen_cf_matrix()
        self.transaction = cf_in
        print(self.item_info.shape, self.user_info.shape, self.transaction.shape)
        # self.item_info = cf_data.T
        # self.user_info = cf_data
        # self.item_size = self.item_info.shape[1]
        # self.user_size = self.user_info.shape[1]
        # self.item_info = np.concatenate((self.item_info, cf_data.T), axis=1)
        # self.user_info = np.concatenate((self.user_info, cf_data), axis=1)
        # self.item_size += self.no_user
        # self.user_size += self.no_item

        # calculate score
        # self.transaction = list(open(data_dir + "review_info.txt"))
        # cols = self.transaction[0].split(', ')[:3]
        # self.transaction = [i.strip().split(', ')[:3] for i in self.transaction[1:]]
        # self.transaction = pd.DataFrame(self.transaction, columns=cols).astype('int')
        # self.transaction['train'] = False
        # for i in range(self.no_user):
        #     self.transaction.train[(self.transaction.u_id == i) & (self.transaction.p_id.isin(self.train[i]))] = True
        # self.transaction = self.transaction[self.transaction.train][cols].to_numpy()

    def load_cvae_data(self, data_dir, data_type):
        data = {}
        variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
        data["content"] = variables.toarray()
        user = np.load(os.path.join("data", data_dir.split("/")[-2], "user_info_%s.npy" % data_type[0]))
        # user = np.load(os.path.join(data_dir,  "user_info_%s.npy" % data_type[0]))
        data["user"] = user
        data["train_users"] = self.load_rating(data_dir + "cf-train-%s-users.dat" % data_type)
        data["train_items"] = self.load_rating(data_dir + "cf-train-%s-items.dat" % data_type)
        data["test_users"] = self.load_rating(data_dir + "cf-test-%s-users.dat" % data_type)
        data["test_items"] = self.load_rating(data_dir + "cf-test-%s-items.dat" % data_type)
        return data

    def load_rating(self, path):
        arr = []
        for line in open(path):
            a = line.strip().split()
            if a[0] == 0:
                l = []
            else:
                l = [int(x) for x in a[1:]]
            arr.append(l)
        return arr

    def gen_cf_matrix(self):
        arr = np.zeros((self.no_user, self.no_item))
        for i in range(self.no_user):
            arr[i, self.train[i]] = 1
        return arr

    def gen_epoch(self):
        user = []
        neg_item = []
        pos_item = []
        for i in range(self.no_user):
            if len(self.train[i]) > 0:
                user += [i] * len(self.train[i])
                neg_item_tmp = list(set(range(self.no_item)) - set(self.train[i]))
                neg_item_tmp = np.random.permutation(neg_item_tmp)[:len(self.train[i])].tolist()
                neg_item += neg_item_tmp[:len(self.train[i])]
                pos_item += self.train[i]
        train = np.column_stack((user, pos_item, neg_item))
        return np.random.permutation(train)

    def gen_batch(self, transaction_batch):
        user = self.user_info[transaction_batch[:, 0]]
        user = np.concatenate((user, user))
        item = self.item_info[transaction_batch[:, 1:].flatten()]
        label = np.concatenate((np.ones(len(transaction_batch)),
                                np.zeros(len(transaction_batch))))

        return user, item, label, transaction_batch

    def gen_batch_rating(self, transaction_batch):
        user = self.user_info[transaction_batch[:, 0]]
        item = self.item_info[transaction_batch[:, 1]]
        return user, item, transaction_batch[:, 2] - 1


def recallK(train, test, predict, k=50):
    recall = []
    for i in range(len(train)):
        pred = np.argsort(predict[i])[::-1][:k+len(train[i])]
        pred = [item for item in pred if item not in train[i]]
        pred = pred[:k]
        hits = len(set(pred) & set(test[i]))
        recall.append(float(hits)/len(test[i]))
    return np.mean(recall)



