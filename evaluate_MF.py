__author__ = 'linh'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from Matrix_Fatorization import MF
def load_cvae_data():
  data = {}
  data_dir = "cf-vae/data/amazon/"

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



U = np.mat(np.loadtxt('MF-model/final-U.dat'))
V = np.mat(np.loadtxt('MF-model/final-V.dat'))
b = np.mat(np.loadtxt('MF-model/final-b.dat'))

data = load_cvae_data()

ratings = pd.read_csv("cf-vae/data/amazon/ratings_MF.csv", header=None)
Y_data = ratings.as_matrix()
rs = MF(Y_data, K = 50, max_iter = 1000, print_every = 100, lam = 0.1, Xinit=V, Winit=U, b=b)
recalls = rs.predict(data["train_users"], data["test_users"], 30)

plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(5, 30, 5),recalls)
plt.savefig("/cf-vae/result/MF-result.png")

