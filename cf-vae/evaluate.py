__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_vae_cpmf_extend import cf_vae_extend, params
import argparse
import os
import scipy

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--mat_file',  type=str, default='cf_vae_1.mat',
                   help='where model is stored')

args = parser.parse_args()
ckpt = args.ckpt_folder
data_dir = args.data_dir
extend_file =args.mat_file

def load_cvae_data(data_dir):
  data = {}
  # variables = scipy.io.loadmat(data_dir + "mult_nor.mat")
  # data["content"] = variables['X']
  variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
  data["content"] = variables.toarray()
  data["train_users"] = load_rating(os.path.join(data_dir,"cf-train-15-users.dat"))
  data["train_items"] = load_rating(os.path.join(data_dir,"cf-train-15-items.dat"))
  data["test_users"] = load_rating(os.path.join(data_dir,"cf-test-15-users.dat"))
  data["test_items"] = load_rating(os.path.join(data_dir ,"cf-test-15-items.dat"))

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

params = params()
params.lambda_u = 1
params.lambda_v = 10
params.lambda_r = 0.1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1


data = load_cvae_data(data_dir)
num_factors = 50
model = cf_vae_extend(num_users=5584, num_items=13790, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[200, 100], z_dim = 50, decoding_dims=[100, 200, 8000], decoding_dims_str=[100,200, 1863],
    loss_type='cross_entropy', encoding_dims_str=[200,100])


d = os.path.join(ckpt, "vae.mat")
print(d)
model.load_model(os.path.join(ckpt, "vae.mat"))
pred = model.predict_all()
recalls_1, mapks_1 = model.predict(pred, data['train_users'], data['test_users'], 100)

model.load_model(os.path.join(ckpt, "vae_user.mat"))
pred = model.predict_all()
recalls_2, mapks_2 = model.predict(pred, data['train_users'], data['test_users'], 100)

model.load_model(os.path.join(ckpt, "dae.mat"))
# model.load_model("cf_vae.mat")
pred = model.predict_all()
recalls, mapks= model.predict(pred, data['train_users'], data['test_users'], 100)

plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(10, 100, 10), recalls_2, '-r', label="our model")
plt.plot(np.arange(10, 100, 10),recalls_1, '-b', label="CVAE")
plt.plot(np.arange(10,100, 10), recalls, '-g', label="CDL")

# plt.plot(np.arange(5, 40, 5), recalls_2, '-g', label="zdim=500")

plt.legend(loc='upper left')
data_dir = data_dir.split("/")[1]
ckpt = ckpt.split("/")[-1]
plt.savefig("result/recall_10_%s_%s.png"%(data_dir, ckpt))
plt.close()

# plt.figure()
# plt.ylabel("Precision@M")
# plt.xlabel("M")
# plt.plot(np.arange(1, 10, 1),precision, '-b', label="cvae")
# plt.plot(np.arange(1, 10, 1), precision_1, '-r', label="img-extend")
# # plt.plot(np.arange(5, 40, 5), recalls_2, '-g', label="zdim=500")

# plt.legend(loc='upper left')
# plt.savefig("result/precision_test.png")
# plt.close()
#
# plt.figure()
# plt.ylabel("MAP@M")
# plt.xlabel("M")
# plt.plot(np.arange(1, 10, 1),mapks_1, '-r', label="our proposed")
# plt.plot(np.arange(1, 10, 1), mapks, '-b', label="CVAE")
# plt.plot(np.arange(1, 10, 1), mapks_2, '-g', label="CDL")
# #
# plt.legend(loc='upper left')
# plt.savefig("result/map_10_%s_%s.png"%(data_dir, ckpt))
# plt.close()

# for j in [0, 1, 3]:
#     i = 0
#     recalls = []
#     for u in [0.1, 1, 10]:
#         for v in [1, 10, 100]:
#             for r in [0.1, 1, 10]:
#                 model.load_model(os.path.join(ckpt, "cf_vae_%d_%d.mat"%(j, i)))
#                 pred = model.predict_all()
#                 f = open(os.path.join(ckpt, "result_10_%d.txt"%j), 'a')
#                 f.write("-----------%f----------%f----------%f\n"%(u,v,r))
#                 model.predict_val(pred, data["train_users"], data["test_users"], f)
#                 f.write("\n")
#                 f.close()
#                 print(u, v, r)
#                 i += 1

