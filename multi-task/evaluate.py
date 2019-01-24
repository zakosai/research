__author__ = 'linh'

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from cf_vae_cpmf_extend import cf_vae_extend, params
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
  min_len = min(len(dataset['test']), len(dataset['tag_test']))

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

def load_cvae_data2(data_dir):
  variables = load_npz(os.path.join(data_dir,"mult_nor.npz"))

  f = open(os.path.join(data_dir,"dataset.pkl"), 'rb')
  dataset = pickle.load(f)
  dataset["content"] = variables.toarray()



  dataset["train_users"] = load_rating(dataset['user_onehot'])
  dataset["train_items"] = load_rating(dataset['item_onehot'])
  dataset["test_users"] = dataset['user_item_test']

  return dataset
def load_rating(path):
    arr = []
    for line in path:
        idx = np.where(line == 1)[0].tolist()
        arr.append(idx)
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
    loss_type='cross_entropy')


# d = os.path.join(ckpt, "vae.mat")
# print(d)
model.load_model(os.path.join(ckpt, extend_file))
pred = model.predict_all()
pred_all = pred[data["test_item_id"]]
train_test = [data["train_users"][i] for i in data["test_item_id"]]
recall = model.predict_val(pred_all, train_test, data["test_item_tag"])

# train_user = []
# for i in data['test_users'].keys():
#     train_user.append(data['train_users'][i])
# pred_all = model.predict_all()
# pred_all = pred_all[data['test_users'].keys()]
# recall = model.predict_val(pred_all, train_user, data["test_users"].values())