from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
from operator import add

def load_data():
  data = {}
  data_dir = "../cf-vae/data/amazon/"

  data["train_users"] = load_rating(data_dir + "cf-train-1-users-small.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items-small.dat")
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


def cal_rec(train_users, test_users, M):
    user_all = map(add, train_users, test_users)
    U = np.mat(np.loadtxt('cdl10/final-U.dat'))
    V = np.mat(np.loadtxt('cdl10/final-V.dat'))

    ground_tr_num = [len(user) for user in user_all]

    pred_all = U * V.T
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


if __name__ == '__main__':

    # give the same p as given in cdl.py
    data = load_data()
    recalls = cal_rec(data['train_users'], data['test_users'], 30)

    plt.figure()
    plt.xlabel("M")
    plt.ylabel("Recall@M")
    plt.plot(np.arange(5, 30, 5),recalls)
    plt.savefig("../cf-vae/result/cdl-result.png")
