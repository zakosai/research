from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
from operator import add

def load_data():
  data = {}
  data_dir = "data/citeulike-a/"

  data["train_users"] = load_rating(data_dir + "cf-train-1-users.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users.dat")

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


def cal_rec(train_users, test_users, M):
    # user_all = map(add, train_users, test_users)
    U = np.mat(np.loadtxt('cdl10/final-U.dat'))
    V = np.mat(np.loadtxt('cdl10/final-V.dat'))
    user_all = test_users
    ground_tr_num = [len(user) for user in user_all]


    pred_all = np.dot(U, V.T)
    pred_all = pred_all.tolist()

    recall_avgs = []
    for m in [50, 300]:
        print "m = " + "{:>10d}".format(m) + "done"
        recall_vals = []
        for i in range(len(user_all)):
            top_M = list(np.argsort(pred_all[i])[-(m +1):])
            if train_users[i] in top_M:
                top_M.remove(train_users[i])
            else:
                top_M = top_M[:-1]
            if len(top_M) != m:
                print(top_M, train_users[i])
            if len(train_users[i]) != 1:
                print(i)
            hits = set(top_M) & set(user_all[i])   # item idex from 0
            hits_num = len(hits)
            try:
                recall_val = float(hits_num) / float(ground_tr_num[i])
            except:
                recall_val = 1
            recall_vals.append(recall_val)
        recall_avg = np.mean(np.array(recall_vals))
        print recall_avg
        recall_avgs.append(recall_avg)
    return recall_avgs


if __name__ == '__main__':

    # give the same p as given in cdl.py
    data = load_data()
    recalls = cal_rec(data['train_users'], data['test_users'], 30)

    plt.figure()
    plt.xlabel("M")
    plt.ylabel("Recall@M")
    plt.plot(np.arange(5, 30, 5),recalls)
    plt.savefig("../cf-vae/result/cdl-result.png")
