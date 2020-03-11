import torch
import torch.nn as nn
from dataset import Dataset, recallK
import numpy as np
import argparse


class DAE(nn.Module):
    def __init__(self, input_size, layers):
        super(DAE, self).__init__()

        # Encoder
        sequence = []
        prev = input_size
        for layer in layers:
            sequence.append(nn.Linear(prev, layer))
            sequence.append(nn.ReLU())
            prev = layer
        self.encoder = nn.Sequential(*sequence).cuda()

        # Decoder
        sequence = []
        layers = layers[::-1]
        prev = layers[0]
        for layer in layers[1:]:
            sequence.append(nn.Linear(prev, layer))
            sequence.append(nn.ReLU())
            prev = layer
        sequence.append(nn.Linear(prev, input_size))
        sequence.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*sequence).cuda()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoder = self.encoder(x)
        output = self.decoder(encoder)
        return output, encoder


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()

        sequence = []
        for i in range(1, len(layers)-1):
            sequence.append(nn.Linear(layers[i-1], layers[i]))
            sequence.append(nn.ReLU())
        sequence.append(nn.Linear(layers[-2], layers[-1]))
        sequence.append(nn.Sigmoid())
        self.net = nn.Sequential(*sequence).cuda()

    def forward(self, x):
        return self.net(x)


def loss_kl(mu, logvar):
    return 0.5 * torch.mean(torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=-1))


def loss_recon(x_recon, x):
    neg_ll = -torch.mean(torch.sum(x_recon * x, dim=-1))
    return neg_ll


def train(data, model, op, loss, device):
    user_info = torch.from_numpy(data[0]).float().to(device)
    item_info = torch.from_numpy(data[1]).float().to(device)
    label = torch.from_numpy(data[2]).float().to(device)

    # # AutoEncoder - user
    # op['user'].zero_grad()
    # user_recon, z_user = model['user'](user_info)
    # loss_user = loss_recon(user_recon, user_info)
    # loss_user.backward()
    # op['user'].step()
    #
    # # AutoEncoder - item
    # op['item'].zero_grad()
    # item_recon, z_item = model['item'](item_info)
    # loss_item = loss_recon(item_recon, item_info)
    # loss_item.backward()
    # op['item'].step()

    # Predict
    op['pred'].zero_grad()
    user_recon, z_user = model['user'](user_info)
    item_recon, z_item = model['item'](item_info)
    # Simplest - Multiple
    # pred = torch.sum(z_user * z_item, dim=-1)
    # # NeuCF
    pred = model['neuCF'](torch.cat([z_user, z_item], -1))

    # Loss
    predict_loss = loss['pred'](pred, label)
    predict_loss.backward()
    op['pred'].step()
    return predict_loss.item(),  predict_loss.item(), predict_loss.item()


def test(data, model, device, neg_size):
    with torch.no_grad():
        user_info = torch.from_numpy(data[0]).float().to(device)
        item_info = torch.from_numpy(data[1]).float().to(device)
        user_recon, z_user = model['user'](user_info)
        item_recon, z_item = model['item'](item_info)

        predict = []
        for i in range(len(user_info)):
            item_recon, z_item = model['item'](item_info[i])
            # pred = torch.matmul(z_user, z_item.T)
            concat = torch.cat([z_user[i].expand(item_info.shape[0], z_user.shape[-1]), z_item], -1)
            pred = model['neuCF'](concat)
            predict.append(pred.view(-1).numpy())
        return predict


def main(args):
    iter = args.iter
    batch_size = 100

    dataset = Dataset(args.data_dir, args.data_type)

    model = {}
    model['user'] = DAE(dataset.user_size, [200, 100])
    model['item'] = DAE(dataset.item_size, [200, 100])
    model['neuCF'] = MLP([200, 50, 1])

    op = {}
    op['user'] = torch.optim.Adam(model['user'].parameters(), lr=0.01)
    op['item'] = torch.optim.Adam(model['item'].parameters(), lr=0.01)
    pred_parameters = list(model['user'].encoder.parameters()) +\
                      list(model['item'].encoder.parameters()) +\
                        list(model['neuCF'].parameters())
    op['pred'] = torch.optim.Adam(pred_parameters, lr=0.01)

    loss = {}
    loss['pred'] = nn.BCELoss(reduction='sum')

    best_result = 0
    for i in range(iter):
        tmp_train = dataset.gen_epoch()
        loss_gen, loss_pred = 0, 0
        for idx in range(0, len(tmp_train), batch_size):
            data = dataset.gen_batch(tmp_train[idx:idx+batch_size])
            _loss_gen, _, _loss_pred = train(data, model, op, loss, 'cuda')
            loss_gen += _loss_gen
            loss_pred += _loss_pred
        print("Loss gen: %f loss_pred: %f "% (loss_gen/len(tmp_train), loss_pred/len(tmp_train)))

        # Test
        predict = test((dataset.user_info, dataset.item_info), model, 'cuda')
        recall, map, right = recallK(dataset.train, dataset.test, predict)
        print("Test:　Recall@10: %f, mAP@10:%f "%(recall, map))
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













