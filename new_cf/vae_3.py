import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset, recallK
import numpy as np
import argparse


class VAE(nn.Module):
    def __init__(self, input_size, layers, last_layer=None):
        super(VAE, self).__init__()

        # Encoder
        sequence = []
        prev = input_size
        for layer in layers[:-1]:
            sequence.append(nn.Linear(prev, layer))
            sequence.append(nn.ReLU())
            prev = layer
        self.encoder = nn.Sequential(*sequence).cuda()

        # z layer
        self.mu = nn.Sequential(nn.Linear(prev, layers[-1])).cuda()
        self.logvar = nn.Sequential(nn.Linear(prev, layers[-1])).cuda()

        # Decoder
        sequence = []
        layers = layers[::-1]
        prev = layers[0]
        for layer in layers[1:]:
            sequence.append(nn.Linear(prev, layer))
            sequence.append(nn.ReLU())
            prev = layer
        sequence.append(nn.Linear(prev, input_size))
        if not last_layer:
            sequence.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*sequence).cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoder = self.encoder(x)
        mu, logvar = self.mu(encoder), self.logvar(encoder)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        loss_kl = self.loss_kl(mu, logvar)
        return output, z, loss_kl

    def loss_kl(self, mu, logvar):
        return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1))


class MLP(nn.Module):
    def __init__(self, layers, last_layer=None):
        super(MLP, self).__init__()

        sequence = []
        for i in range(1, len(layers)-1):
            sequence.append(nn.Linear(layers[i-1], layers[i]))
            sequence.append(nn.ReLU())
        sequence.append(nn.Linear(layers[-2], layers[-1]))
        if not last_layer:
            sequence.append(nn.Sigmoid())
        else:
            sequence.append(nn.LogSoftmax())
        self.net = nn.Sequential(*sequence).cuda()

    def forward(self, x):
        return self.net(x)


def loss_kl(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1))


def loss_recon(x_recon, x):
    log_var = F.log_softmax(x_recon)
    neg_ll = -torch.mean(torch.sum(log_var * x, dim=-1))
    return neg_ll


# def train(data, model, op, loss, device):
#     user_info = torch.from_numpy(data[0]).float().to(device)
#     item_info = torch.from_numpy(data[1]).float().to(device)
#     label = torch.from_numpy(data[2]).float().to(device)
#
#     # # AutoEncoder - user
#     op['user'].zero_grad()
#     user_recon, z_user, loss_kl_user = model['user'](user_info)
#     loss_user = loss['user'](user_recon, user_info) + 0.01 * loss_kl_user
#     loss_user.backward()
#     op['user'].step()
#
#     # AutoEncoder - item
#     op['item'].zero_grad()
#     item_recon, z_item, loss_kl_item = model['item'](item_info)
#     loss_item = loss['item'](item_recon, item_info) + 0.01 * loss_kl_item
#     loss_item.backward()
#     op['item'].step()
#
#     # Predict
#     op['pred'].zero_grad()
#     user_recon, z_user, _ = model['user'](user_info)
#     item_recon, z_item, _ = model['item'](item_info)
#     # Simplest - Multiple
#     # pred = torch.sum(z_user * z_item, dim=-1)
#     # # NeuCF
#     pred = model['neuCF'](torch.cat([z_user, z_item], -1)).view(-1)
#     # pred = pred.clamp(0, 5)
#
#     # Loss
#     predict_loss = loss['pred'](pred, label)
#     predict_loss.backward()
#     op['pred'].step()
#     return loss_item.item(),  loss_user.item(), predict_loss.item()

def train_user(data, model, op, loss, device):
    user_info = torch.from_numpy(data).float().to(device)
    op['user'].zero_grad()
    user_recon, z_user, loss_kl_user = model['user'](user_info)
    loss_user = loss['user'](user_recon, user_info) + 0.01 * loss_kl_user
    loss_user.backward()
    op['user'].step()
    return loss_user.item()


def train_item(data, model, op, loss, device):
    item_info = torch.from_numpy(data).float().to(device)
    op['item'].zero_grad()
    item_recon, z_item, loss_kl_item = model['item'](item_info)
    loss_item = loss['item'](item_recon, item_info) + 0.01 * loss_kl_item
    loss_item.backward()
    op['item'].step()
    return loss_item.item()


def train_cf(data, model, op, device):
    user_transaction = torch.from_numpy(data[0]).float().to(device)
    # user_info = torch.from_numpy(data[1]).float().to(device)
    # item_info = torch.from_numpy(data[2]).float().to(device)
    #
    # op['pred'].zero_grad()
    # _, z_user, _ = model['user'](user_info)
    # _, z_item, _ = model['item'](item_info)
    #
    # content_matrix = torch.matmul(z_user, z_item.T)
    # content_matrix = F.normalize(content_matrix, dim=-1)
    # transaction = user_transaction * content_matrix
    trans_recon, _, loss_kl = model['neuCF'](user_transaction)
    loss = loss_recon(trans_recon, user_transaction) + 0.01 * loss_kl
    loss.backward()
    op['pred'].step()

    return loss.item()


def test(data, model, device, batch_size):
    with torch.no_grad():
        user_info = torch.from_numpy(data[0]).float().to(device)
        item_info = torch.from_numpy(data[1]).float().to(device)
        user_transaction = torch.from_numpy(data[2]).float().to(device)
        user_recon, z_user, _ = model['user'](user_info)
        item_recon, z_item, _ = model['item'](item_info)
        content_matrix = torch.matmul(z_user, z_item.T)
        content_matrix = F.normalize(content_matrix, dim=-1)
        transaction = content_matrix * user_transaction

        predict = []
        for i in range(0, len(user_info), batch_size):
            # pred = torch.matmul(z_user, z_item.T)
            _, pred, _ = model['neuCF'](transaction[i:i+batch_size])
            predict.append(pred.cpu().numpy())
        return np.concatenate(predict)


def main(args):
    iter = args.iter
    batch_size = 100

    dataset = Dataset(args.data_dir, args.data_type)

    model = {}
    model['user'] = VAE(dataset.user_size, [200, 100, 50])
    model['item'] = VAE(dataset.item_size, [100, 50])
    model['neuCF'] = VAE(dataset.no_item, [100, 50], nn.LogSoftmax())

    op = {}
    op['user'] = torch.optim.Adam(model['user'].parameters(), lr=0.01)
    op['item'] = torch.optim.Adam(model['item'].parameters(), lr=0.01)
    pred_parameters = list(model['user'].encoder.parameters()) + list(model['user'].mu.parameters()) + \
                        list(model['user'].logvar.parameters()) + list(model['item'].encoder.parameters()) + \
                      list(model['item'].mu.parameters()) + list(model['item'].logvar.parameters()) + \
                      list(model['neuCF'].parameters())
    op['pred'] = torch.optim.Adam(model['neuCF'].parameters(), lr=0.01)

    loss = {}
    loss['pred'] = nn.BCELoss(reduction='sum')
    loss['item'] = nn.BCELoss()
    loss['user'] = nn.BCELoss()

    best_result = 0
    for i in range(iter):
        loss_user, loss_item, loss_trans = 0, 0, 0
        tmp_user = np.random.permutation(range(dataset.no_user))
        for i in range(0, dataset.no_user, batch_size):
            loss_user += train_user(dataset.user_info[tmp_user[i:i+batch_size]],
                                    model, op, loss, 'cuda')

        tmp_item = np.random.permutation(range(dataset.no_item))
        for i in range(0, dataset.no_item, batch_size):
            loss_item += train_item(dataset.item_info[tmp_item[i:i+batch_size]],
                                    model, op, loss, 'cuda')

        tmp_user = np.random.permutation(range(dataset.no_user))
        for i in range(0, dataset.no_user, batch_size):
            loss_trans += train_cf([dataset.transaction[tmp_user[i:i+batch_size]], dataset.user_info[tmp_user[i:i+batch_size]], dataset.item_info], model, op, 'cuda')

        print("Loss user: %f, loss_item: %f, loss_pred: %f " % (loss_user, loss_item, loss_trans))

        # Test
        predict = test((dataset.user_info, dataset.item_info, dataset.transaction), model, 'cuda', batch_size)
        recall = recallK(dataset.train, dataset.test, predict, 10)
        print("Test:　Recall@10: %f "%(recall))
        if recall > best_result:
            best_result = recall

    print("Best result: ", best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='5')
    parser.add_argument('--data_dir', type=str, default='data/amazon')
    parser.add_argument('--iter', type=int, default=30)
    args = parser.parse_args()

    main(args)













