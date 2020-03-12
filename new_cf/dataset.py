import numpy as np
from scipy.sparse import load_npz
import os


class Dataset:
    def __init__(self, data_dir, data_type):
        data = self.load_cvae_data(data_dir, data_type)
        self.item_info = data['content']
        self.user_info = data['user']
        self.train = data['train_users']
        self.test = data['test_users']
        self.no_item, self.item_size, = self.item_info.shape
        self.no_user, self.user_size = self.user_info.shape

        # cf_data = self.gen_cf_matrix()
        # self.item_info = np.concatenate((self.item_info, cf_data.T), axis=1)
        # self.user_info = np.concatenate((self.user_info, cf_data), axis=1)
        # self.item_size += self.no_user
        # self.user_size += self.no_item

    def load_cvae_data(self, data_dir, data_type):
        data = {}
        # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
        # data["content"] = variables['X']
        variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
        data["content"] = variables.toarray()
        # variables = np.load(os.path.join(data_dir, "structure.npy"))
        # data["structure"] = variables
        user = np.load(os.path.join(data_dir, "user_info_%s.npy" % data_type))
        # user = user[:, 7:30]
        data["user"] = user
        data["train_users"] = self.load_rating(data_dir + "cf-train-%sp-users.dat" % data_type)
        data["train_items"] = self.load_rating(data_dir + "cf-train-%sp-items.dat" % data_type)
        data["test_users"] = self.load_rating(data_dir + "cf-test-%sp-users.dat" % data_type)
        data["test_items"] = self.load_rating(data_dir + "cf-test-%sp-items.dat" % data_type)
        # data["train_users_rating"] = load_rating(data_dir + "train-%s-users-rating.dat"%data_type)
        # data["train_items_rating"] = load_rating(data_dir + "train-%s-items-rating.dat"%data_type)
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
                neg_item += neg_item_tmp
                pos_item += self.train[i]
        train = np.column_stack((user, pos_item, neg_item))
        return np.random.permutation(train)

    def gen_batch(self, transaction_batch):
        user = self.user_info[transaction_batch[:, 0]]
        user = np.concatenate((user, user))
        item = np.concatenate((self.item_info[transaction_batch[:, 1]], self.item_info[transaction_batch[:, 2]]))
        label = np.concatenate((np.ones(len(transaction_batch)), np.zeros(len(transaction_batch))))

        return user, item, label, transaction_batch


def recallK(train, test, predict, k=50):
    recall = []
    for i in range(len(train)):
        pred = np.argsort(predict[i])[::-1][:k+len(train[i])]
        pred = [item for item in pred if item not in train[i]]
        pred = pred[:k]
        hits = len(set(pred) & set(test[i]))
        recall.append(float(hits)/len(test[i]))
    return np.mean(recall)



