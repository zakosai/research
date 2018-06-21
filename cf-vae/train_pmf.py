import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cvae_user import cf_vae_extend, params
from scipy.sparse import load_npz
import argparse
import os
import csv
from PMF import PMF

np.random.seed(0)
tf.set_random_seed(0)


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model',  type=int, default=0,
                   help='type of model: 0-only text, 1-text+image, 2-text+image+structure, 3-text+structure')
parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--initial',  type=bool, default=True,
                   help='where model is stored')
parser.add_argument('--iter',  type=int, default=30,
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--zdim',  type=int, default=50,
                   help='where model is stored')
parser.add_argument('--gridsearch',  type=int, default=0,
                   help='gridsearch or not')
parser.add_argument('--data_type',  type=str, default='5',
                   help='gridsearch or not')
parser.add_argument('--user_dim',  type=int, default=9975,
                   help='gridsearch or not')
parser.add_argument('--user_no',  type=int, default=6040,
                   help='gridsearch or not')
parser.add_argument('--item_no',  type=int, default=3883,
                   help='gridsearch or not')
args = parser.parse_args()
model_type = args.model
ckpt = args.ckpt_folder
initial = args.initial
iter = args.iter
data_dir = args.data_dir
zdim = args.zdim
gs = args.gridsearch
data_type = args.data_type
print(model_type)

def load_cvae_data(data_dir):
  data = {}
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']
  variables = load_npz(os.path.join(data_dir,"mult_nor.npz"))
  data["content"] = variables.toarray()
  variables = np.load(os.path.join(data_dir, "structure.npy"))
  data["structure"] = variables
  user = np.load(os.path.join(data_dir, "user_info_%s4.npy"%data_type))
  # user = np.delete(user, [7,8,9,10,11], axis=1)
  data["user"] = user
  data["train_users"] = load_rating(data_dir + "cf-train-%s-users.dat"%data_type)
  data["train_items"] = load_rating(data_dir + "cf-train-%s-items.dat"%data_type)
  data["test_users"] = load_rating(data_dir + "cf-test-%s-users.dat"%data_type)
  data["test_items"] = load_rating(data_dir + "cf-test-%s-items.dat"%data_type)

  data["train_vec"] = load_rating2(os.path.join(data_dir, "train-%s.csv"%data_type))
  data["val_vec"] =load_rating2(os.path.join(data_dir, "val-%s.csv"%data_type))
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

def load_rating2(path):
    arr = list(open(path).readlines())
    arr = arr[1:]
    arr = [a.strip() for a in arr]
    arr = [a.split(",") for a in arr]
    arr = np.array(arr).astype(np.int32)
    return arr



params = params()
params.lambda_u = 1
params.lambda_v = 10
params.lambda_r = 0.1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1
params.EM_iter = args.iter
params.num_iter = 150

C = [0.1, 1, 10]

# # for updating W and b in vae
# self.learning_rate = 0.001
# self.batch_size = 500
# self.num_iter = 3000
# self.EM_iter = 100

data = load_cvae_data(data_dir)
np.random.seed(0)
tf.set_random_seed(0)

images = np.fromfile(os.path.join(data_dir,"images.bin"), dtype=np.uint8)
img = images.reshape((13791, 32, 32, 3))
img = img.astype(np.float32)/255
num_factors = zdim
i = 0
e = 3
l = 0.03
m = 0.2
for e in [0.01, 0.03, 0.1, 1, 3, 10, 30, 100]:
    for l in [0.01, 0.03, 0.1, 1, 3, 10, 30, 100]:
        for m in [0.01, 0.03, 0.1, 1, 3, 10, 30, 100]:
            model = PMF(epsilon=e, _lambda=l, momentum=m, num_feat=50, maxepoch=50, num_batches=43)
            model.fit(data["train_vec"], data["val_vec"], train_users=data["train_users"], test_users=data["test_users"])
            # model.save_model(os.path.join(ckpt,"pmf_%d.mat"%(i)))
                    # model.load_model("cf_vae.mat")
            f = open(os.path.join(ckpt, "result_pmf_%d.txt"%model_type), 'w')
            f.write("-----------%f----------%f----------%f\n"%(e,l,m))
            model.predict_val(data["train_users"], data["test_users"], f)
            f.write("\n")
            f.close()
            print(e, l, m)
            i += 1

# model = PMF(epsilon=e, _lambda=l, momentum=m, num_feat=50, maxepoch=100, num_batches=43)
# model.fit(data["train_vec"], data["val_vec"], train_users=data["train_users"], test_users=data["test_users"])


