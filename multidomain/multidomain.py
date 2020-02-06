import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
import numpy as np


class MultiDomain(nn.Module):
    def __init__(self, input_size_list, layers, n_domain):
        super(MultiDomain, self).__init__()
        self.n_domain = n_domain
        self.domain_encode_net = self.create_enc_domain_net(input_size_list, layers[0])

        sequence_net = []
        for i in range(1, len(layers)-1):
            sequence_net.append(nn.Linear(layers[i-1], layers[i]))
            sequence_net.append(nn.ReLU())
        self.encoder = nn.Sequential(*sequence_net)
        self.z = nn.Sequential(nn.Linear(layers[-2], layers[-1]), nn.ReLU())

        sequence_net = []
        for i in range(len(layers)-2, 0, --1):
            sequence_net.append(nn.Linear(layers[i+1], layers[i]))
            sequence_net.append(nn.ReLU())
        self.decoder = nn.Sequential(*sequence_net)
        self.domain_decode_net = self.create_dec_domain_net(layers[0], input_size_list)

    def create_enc_domain_net(self, input_size, output_size):
        domain_net = {}
        for i in range(self.n_domain):
            domain_net[i] = nn.Sequential(nn.Linear(input_size[i], output_size), nn.ReLU())
        return domain_net

    def create_dec_domain_net(self, input_size, output_size):
        domain_net = {}
        for i in range(self.n_domain):
            domain_net[i] = nn.Sequential(nn.Linear(input_size, output_size[i]), nn.ReLU())
        return domain_net

    def forward(self, x, domain_in, domain_out):
        domain_enc_net = self.domain_encode_net[domain_in](x)
        z = self.z(self.encoder(domain_enc_net))
        output = self.domain_encode_net[domain_out](self.decoder(z))
        return z, output

    def reconstruction_loss(self, predict, label):
        log_softmax_var = F.log_softmax(predict)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * label, dim=-1))
        return neg_ll


def train(data, op, model):
    A_data = data[0].cuda()
    B_data = data[1].cuda()
    label = data[2]

    op.zero_grad()
    B_fake = model(A_data, label[0], label[1])
    A_fake = model(B_data, label[1], label[0])
    loss = model.reconstruction_loss(B_fake, B_data) + model.reconstruction_loss(A_fake, A_data)
    return loss


def main():
    iter = 100
    batch_size = 500
    dataset = Dataset(["Health", "Clothing", "Grocery"])
    model = MultiDomain(dataset.input_size_list, [200, 100, 50], 3)

    for i in range(iter):
        domain, ids = dataset.random_iter(batch_size)
        shuffle_idx = np.random.permutation(len(domain))
        loss = 0
        for idx in shuffle_idx:
            data = dataset.get_batch_train(domain[idx], ids[idx])
            print(data[2][0], data[2][1])
            parameters = list(model.domain_encode_net[data[2][0]].parameters()) + \
                list(model.domain_encode_net[data[2][1]].parameters())+ \
                list(model.encoder.parameters()) + list(model.z.parameters()) + list(model.decoder.parameters()) + \
                list(model. domain_decode_net[data[2][1]].parameters()) + \
                list(model.domain_decode_net[data[2][0]].parameters())
            op = torch.optim.Adam(parameters, lr=0.01)
            loss += train(data, op, model)
        print(loss)


if __name__ == '__main__':
    main()












