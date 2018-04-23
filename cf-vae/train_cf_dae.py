__author__ = 'linh'
import numpy as np
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from cf_dae import cf_vae_extend, params
from scipy.sparse import load_npz
import  argparse

np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model',  type=int, default=0,
                   help='type of model: 0-only text, 1-text+image, 2-text+image+structure, 3-text+structure')


args = parser.parse_args()
model = args.model
print(model)

def load_cvae_data():
  data = {}
  data_dir = "data/amazon2/"
  # variables = scipy.io.loadmat(data_dir + "mult_nor-small.mat")
  # data["content"] = variables['X']
  variables = load_npz("data/amazon2/mult_nor-small.npz")
  data["content"] = variables.toarray()
  # variables = load_npz("data/amazon2/structure_mult_nor-small.npz")
  # data["structure"] = variables.toarray()
  data["train_users"] = load_rating(data_dir + "cf-train-1-users-small.dat")
  data["train_items"] = load_rating(data_dir + "cf-train-1-items-small.dat")
  data["test_users"] = load_rating(data_dir + "cf-test-1-users-small.dat")
  data["test_items"] = load_rating(data_dir + "cf-test-1-items-small.dat")

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
params.lambda_u = 0.1
params.lambda_v = 10
params.lambda_r = 1
params.C_a = 1
params.C_b = 0.01
params.max_iter_m = 1
params.EM_iter = 15


# # for updating W and b in vae
# self.learning_rate = 0.001
# self.batch_size = 500
# self.num_iter = 3000
# self.EM_iter = 100


data = load_cvae_data()
np.random.seed(0)
tf.set_random_seed(0)



images = np.fromfile("data/amazon2/images.bin", dtype=np.uint8)
img = images.reshape((16000, 64, 64, 3))
img = img.astype(np.float32)/255
num_factors = 50
model = cf_vae_extend(num_users=8000, num_items=16000, num_factors=num_factors, params=params,
    input_dim=8000, encoding_dims=[1000, 200], z_dim = 50, decoding_dims=[200, 1000, 8000],
    decoding_dims_str=[100,200, 1863], loss_type='cross_entropy')
model.fit(data["train_users"], data["train_items"], data["content"],img, data["content"], params)
model.save_model("pre3/dae/cf_dae.mat")
# model.load_model("cf_vae.mat")
pred = model.predict_all()
recalls = model.predict(pred, data['train_users'], data['test_users'], 40)

plt.figure()
plt.ylabel("Recall@M")
plt.xlabel("M")
plt.plot(np.arange(5, 40, 5),recalls)
plt.show()
