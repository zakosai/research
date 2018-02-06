__author__ = 'linh'

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from operator import add
import matplotlib.pyplot as plt


class MF(object):
    """docstring for CF"""
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None,
                 learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 0, b=None):
        self.Y_raw = Y_data.copy()
        self.Y_data = Y_data.copy()
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = 8000
        self.n_items = 16000

        if Xinit is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit

        if Winit is None:
            self.W = np.random.randn(K, self.n_users)
        else:
            self.W = Winit

        # item biases
        if b is None:
            self.b = np.random.randn(self.n_items)
        self.d = np.random.randn(self.n_users)
        #self.all_users = self.Y_data[:,0] # all users (may be duplicated)
        self.n_ratings = Y_data.shape[0]
#         self.mu = np.mean(Y_data[:, 2])
        self.mu = 0


    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:
            user_col = 1
            item_col = 0
            n_objects = self.n_items

        users = self.Y_data[:, user_col]
        self.muu = np.zeros((n_objects,))
        for n in xrange(n_objects):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, item_col]
            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings)
#             print m
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.muu[n] = m
            # normalize
            self.Y_data[ids, 2] = ratings - m


    def loss(self):
        L = 0
        for i in xrange(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data[i, 0]), int(self.Y_data[i, 1]), self.Y_data[i, 2]
            L += 0.5*(self.X[m, :].dot(self.W[:, n]) + self.b[m] + self.d[n] + self.mu - rate)**2

        # regularization, don't ever forget this
        L /= self.n_ratings
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') + \
                          np.linalg.norm(self.b) + np.linalg.norm(self.d))
        return L


    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user n, and the corresponding ratings
        """
        # y = self.Y_data_n[:,0] # all users (may be duplicated)
        # item indices rated by user_id
        # we need to +1 to user_id since in the rate_matrix, id starts from 1
        # while index in python starts from 0
        ids = np.where(self.Y_data[:,0] == user_id)[0]
        item_ids = self.Y_data[ids, 1].astype(np.int32) # index starts from 0
        ratings = self.Y_data[ids, 2]
        return (item_ids, ratings)


    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item m and get the corresponding ratings
        """
        ids = np.where(self.Y_data[:,1] == item_id)[0]
        user_ids = self.Y_data[ids, 0].astype(np.int32)
        ratings = self.Y_data[ids, 2]
        return (user_ids, ratings)

    def updateX(self):
        for m in xrange(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)

            Wm = self.W[:, user_ids]
            dm = self.d[user_ids]
            xm = self.X[m, :]

            error = xm.dot(Wm) + self.b[m] + dm + self.mu - ratings

            grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
            grad_bm = np.sum(error)/self.n_ratings + self.lam*self.b[m]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
            self.b[m]    -= self.learning_rate*grad_bm

    def updateW(self):
        for n in xrange(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            bn = self.b[item_ids]
            wn = self.W[:, n]

            error = Xn.dot(wn) + bn + self.mu + self.d[n] - ratings
            grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
            grad_dn = np.sum(error)/self.n_ratings + self.lam*self.d[n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
            self.d[n]    -= self.learning_rate*grad_dn

    def fit(self):
        self.normalize_Y()
        for it in xrange(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw)
                print 'iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train

    def save_model(self, dir_save):
        np.savetxt(dir_save+'/final-U.dat',self.W,fmt='%.5f',comments='')
        np.savetxt(dir_save+'/final-V.dat',self.X,fmt='%.5f',comments='')
        np.savetxt(dir_save+'/final-b.dat',self.b,fmt='%.5f',comments='')

    def pred(self, u, i):
        """
        predict the rating of user u for item i
        if you need the un
        """
        u = int(u)
        i = int(i)
        if self.user_based == 1:
            bias = self.muu[u]
        else:
            bias = self.muu[i]

        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u] + bias
#         if pred < 0:
#             return 0
#         if pred > 5:
#             return 5
#         return pred
        return max(0, min(5, pred))


    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()

        y_pred = self.X.dot(self.W[:, user_id])
        predicted_ratings= []
        for i in xrange(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))

        return predicted_ratings

    def pred_all(self):
        y_pred = self.X.dot(self.W) + self.b.reshape((self.n_items, 1))
        return y_pred

    def predict(self, train_users, test_users, M):
        user_all = map(add, train_users, test_users)
        # user_all = np.array(user_all)    # item idex from 1
        ground_tr_num = [len(user) for user in user_all]


        pred_all = self.pred_all()
        pred_all = list(pred_all)

        recall_avgs = []
        for m in range(5, M, 5):
            print "m = " + "{:>10d}".format(m) + "done"
            recall_vals = []
            for i in range(len(user_all)):
                top_M = np.argsort(-pred_all[i])[0:m]
                hits = set(top_M) & set(user_all[i])   # item idex from 0
                hits_num = len(hits)
                recall_val = float(hits_num) / float(ground_tr_num[i])
                recall_vals.append(recall_val)
            recall_avg = np.mean(np.array(recall_vals))
            print recall_avg
            recall_avgs.append(recall_avg)
        return recall_avgs

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in xrange(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
#             print pred, rate_test[n, 2]
            SE += (pred - rate_test[n, 2])**2

        RMSE = np.sqrt(SE/n_tests)
        return RMSE





def load_cvae_data():
  data = {}
  data_dir = "/cf-vae/data/amazon/"

  data["train_users"] = load_rating(data_dir + "cf-train-1-users-small.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users-small.dat")

  return data

def load_rating(path):
  arr = []
  for line in open(path):
    a = line.strip().split()
    if a[0]==0:
      l = []
    else:
      l = [int(x) for x in a[1:]]
    arr.append(l)
  return arr

# if __name__ == '__main__':
#
#     ratings = pd.read_csv("cf-vae/data/amazon/ratings_MF.csv", header=None)
#     Y_data = ratings.as_matrix()
#
#
#     rs = MF(Y_data, K = 50, max_iter = 1000, print_every = 100, lam = 0.1)
#
#     rs.fit()
#     rs.save_model("MF-model")
#
#     data = load_cvae_data()
#     recalls = rs.predict(data["train_users"], data["test_users"], 30)
#
#     plt.figure()
#     plt.ylabel("Recall@M")
#     plt.xlabel("M")
#     plt.plot(np.arange(5, 30, 5),recalls)
#     plt.savefig("/cf-vae/result/MF-result.png")


