__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_dae import cf_vae_extend, params
import argparse
import os
import scipy
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--ckpt_folder',  type=str, default='pre_model/exp1/',
                   help='where model is stored')
parser.add_argument('--data_dir',  type=str, default='data/amazon',
                   help='where model is stored')
parser.add_argument('--mat_file',  type=str, default='cf_vae_1.mat',
                   help='where model is stored')
parser.add_argument('--type',  type=str, default=15,
                   help='where model is stored')

args = parser.parse_args()
ckpt = args.ckpt_folder
data_dir = args.data_dir
extend_file =args.mat_file

def load_cvae_data(data_dir):


  f = open(os.path.join(data_dir,"dataset.pkl"), 'rb')
  dataset = pickle.load(f)
  variables = load_npz(os.path.join(data_dir, "mult_nor.npz"))
  dataset["content"] = variables.toarray()

  train_item = [0] * dataset['item_no']
  train_tag = [0] * dataset['tag_no']
  tag_list = np.where(dataset['tag_item_onehot'] == 1)
  for i in range(len(tag_list[0])):
      item_id = tag_list[0][i]
      tag_id = tag_list[1][i]
      if train_item[item_id] == 0:
          train_item[item_id] = [tag_id]
      else:
          train_item[item_id].append(tag_id)

      if train_tag[tag_id] == 0:
          train_tag[tag_id] = [item_id]
      else:
          train_tag[tag_id].append(item_id)
  for i in range(len(train_item)):
      if train_item[i] == 0:
          train_item[i] = []
  for i in range(len(train_tag)):
      if train_tag[i] == 0:
          train_tag[i] = []


  dataset["train_users"] = train_item
  dataset["train_items"] = train_tag


  test_tag_id = []
  test_tag_y = []
  min_len = min(len(dataset['test'], dataset['tag_test']))

  for i in range(min_len):
      try:
          idx = test_tag_id.index(dataset['test'][i, 1])
          print(test_tag_y[idx], dataset['tag_test'][i])
          test_tag_y[idx] += dataset['tag_test'][i]
          test_tag_y[idx] = list(set(test_tag_y[idx]))
          print(test_tag_y[idx])
      except:
          test_tag_id.append(dataset['test'][i, 1])
          test_tag_y.append(dataset['tag_test'][i])
  dataset["test_item_id"] = test_tag_id
  dataset["test_item_tag"] = test_tag_y
  return dataset

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
    loss_type='cross_entropy')


# d = os.path.join(ckpt, "vae.mat")
# print(d)
model.load_model(os.path.join(ckpt, extend_file))
pred = model.predict_all()
pred_all = pred[data["test_item_id"]]
train_test = [data["train_users"][i] for i in data["test_item_id"]]
recall = model.predict_val(pred_all, train_test, data["test_item_tag"])

# model.load_model(os.path.join(ckpt, "vae_user.mat"))
# pred = model.predict_all()
# model.predict_val(pred, data['train_users'], data['test_users'])
#
# model.load_model(os.path.join(ckpt, "dae.mat"))
# # model.load_model("cf_vae.mat")
# pred = model.predict_all()
# model.predict_val(pred, data['train_users'], data['test_users'])
#
# plt.figure()
# plt.ylabel("Recall@M")
# plt.xlabel("M")
# plt.plot(np.arange(10, 100, 10), recalls_2, '-r', label="our model")
# plt.plot(np.arange(10, 100, 10),recalls_1, '-b', label="CVAE")
# plt.plot(np.arange(10,100, 10), recalls, '-g', label="CDL")
#
# # plt.plot(np.arange(5, 40, 5), recalls_2, '-g', label="zdim=500")
#
# plt.legend(loc='upper left')
# data_dir = data_dir.split("/")[1]
# ckpt = ckpt.split("/")[-1]
# plt.savefig("result/recall_10_%s_%s.png"%(data_dir, ckpt))
# plt.close()

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

