import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import argparse
import os

class MultiTask(nn.Module):
    def __init__(self, enc_layers, dim_in, z_dim, dec_layers, dev, lambda_1=0.1, lambda_2=100):
        super(MultiTask, self).__init__()
        self.eps = 1e-10
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


        module_enc_list = []
        for i in range(len(enc_layers)):
            module_enc_list.extend([nn.Linear(dim_in, enc_layers[i]),
                                nn.LeakyReLU(0.5)])
            dim_in = enc_layers[i]
        self.enc = nn.Sequential(*module_enc_list)

        self.mu = nn.Linear(dim_in, z_dim)
        self.sigma = nn.Linear(dim_in, z_dim)

        module_dec_list = []
        dim_in = z_dim
        for i in range(len(dec_layers)):
            module_dec_list.extend([nn.Linear(dim_in, dec_layers[i]),
                                    nn.LeakyReLU(0.5)])
            dim_in = dec_layers[i]
        self.dec = nn.Sequential(*module_dec_list)

        self.log_softmax = nn.LogSoftmax()

    def kl_loss(self, mu, sigma):
        return 0.5*torch.mean(torch.sum(mu**2 + torch.exp(sigma) - sigma - 1, dim=1))

    def reconstruction_loss(self, y, y_pred):
        return -torch.mean(torch.sum(y_pred*y, dim=-1))

    def forward(self, x,y):
        dev = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        x_ = torch.cat((x, y), dim=1)
        x_ = self.enc(x_)
        mu = self.mu(x_)
        sigma = self.sigma(x_)
        e = torch.randn(mu.size(), device=dev)
        eps = torch.zeros(sigma.size(), device=dev) + self.eps
        self.z = mu + torch.sqrt(torch.max(torch.exp(sigma), eps)) * e
        y_ = self.dec(self.z)
        y_softmax = self.log_softmax(y_)

        l2_loss = 0
        for p in self.parameters():
            l2_loss += p.norm(2)
        self.loss = self.lambda_1 * self.kl_loss(mu, sigma) + self.lambda_2 * self.reconstruction_loss(y, y_softmax) \
                    + 0.1 * l2_loss
        return y_, self.loss


def calc_recall(pred, test, m=[100], type=None):
    result = {}
    for k in m:
        pred_ab = np.argsort(-pred)[:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
            if len(test[i]) != 0:
                hits = set(test[i]) & set(p)

                #recall
                recall_val = float(len(hits)) / len(test[i])
                recall.append(recall_val)

                #ncdg
                score = []
                for j in range(k):
                    if p[j] in hits:
                        score.append(1)
                    else:
                        score.append(0)
                actual = dcg_score(score, pred[i, p], k)
                best = dcg_score(score, score, k)
                if best == 0:
                    ndcg.append(0)
                else:
                    ndcg.append(float(actual) / best)

        print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))
        result['recall@%d'%k] = np.mean(recall)
        result['ndcg@%d'%k] = np.mean(ndcg)


    return np.mean(np.array(recall)), result

def dcg_score(y_true, y_score, k=50):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def calc_rmse(pred, test):
    idx = np.where(test != 0)
    pred = pred[idx]
    test = test[idx]
    return np.sqrt(np.mean((test-pred)**2))

def main():
    epoches = 3000
    batch_size= 500
    args = parser.parse_args()
    f = open(args.data, 'rb')
    dataset = pickle.load(f, encoding='latin1')
    forder = args.data.split("/")[:-1]
    forder = "/".join(forder)
    content = np.load(os.path.join(forder, "item.npz"))
    content = content['z']

    num_p = dataset['item_no']
    num_u = dataset['user_no']
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    encoding_dim = [600, 200]
    decoding_dim = [200, 600, num_p]

    z_dim = 50
    max_item = max(np.sum(dataset['user_onehot'], axis=1))
    x_dim = z_dim * max_item
    user_item = np.zeros((num_u,x_dim))
    for i in range(num_u):
        idx = np.where(dataset['user_onehot'][i] == 1)
        u_c = content[idx]
        u_c = u_c.flatten()
        user_item[i, :len(u_c)] = u_c

    dim_in = x_dim + num_p
    model = MultiTask(encoding_dim, dim_in, z_dim, decoding_dim, dev)
    opt = optim.Adam(model.parameters(), lr=1e-4)


    x_test = user_item[list(dataset['user_item_test'].keys())]
    y_test = dataset['user_onehot'][list(dataset['user_item_test'].keys())].astype(np.uint8)
    x_train, y_train, x_test, y_test = map(torch.tensor, (user_item.astype(np.float32), dataset['user_onehot'].astype(
        np.float32), x_test.astype(np.float32),y_test.astype(np.float32)))
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)

    model.to(dev)
    max_recall = 0
    for epoch in range(epoches):
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            y_pred, loss = model(xb, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        if epoch%10 == 0:
            print("loss last batch: %f", loss.item())
            x_test, y_test = x_test.to(dev), y_test.to(dev)
            item_pred, loss = model(x_test, y_test)
            item_pred = item_pred.cpu().detach().numpy()
            recall_item, _ = calc_recall(item_pred, list(dataset['user_item_test'].values()), [50], "item")
            model.train = True

            if recall_item > max_recall:
                max_recall = recall_item
                if max_recall < 0.1:
                    _, result = calc_recall(item_pred, list(dataset['user_item_test'].values()),
                                            [50, 100, 150, 200, 250, 300], "item")
                else:
                    _, result = calc_recall(item_pred, list(dataset['user_item_test'].values()),
                                            [10, 20, 30, 40, 50, 60], "item")



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data',  type=str, default="Tool",
                   help='dataset name')
parser.add_argument('--ckpt',  type=str, default="experiment/delicious",
                   help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')


if __name__ == '__main__':
    main()


