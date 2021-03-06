import torch
import torch.nn as nn
from dataset import Dataset, calc_recall
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_size_list, layers, n_domain):
        super(VAE, self).__init__()
        self.n_domain = n_domain
        self.domain_encode_net = self.create_enc_domain_net(input_size_list, layers[0])

        sequence_net = []
        for i in range(1, len(layers)-1):
            sequence_net.append(nn.Linear(layers[i-1], layers[i]))
            sequence_net.append(nn.ReLU())
        self.encoder = nn.Sequential(*sequence_net)
        self.mu = nn.Sequential(nn.Linear(layers[-2], layers[-1]), nn.ReLU())
        self.logvar = nn.Sequential(nn.Linear(layers[-2], layers[-1]), nn.ReLU())

        sequence_net = []
        for i in range(len(layers)-2, -1, -1):
            sequence_net.append(nn.Linear(layers[i+1], layers[i]))
            sequence_net.append(nn.ReLU())
        self.decoder = nn.Sequential(*sequence_net)
        self.domain_decode_net = self.create_dec_domain_net(layers[0], input_size_list)

    def create_enc_domain_net(self, input_size, output_size):
        domain_net = {}
        for i in range(self.n_domain):
            domain_net[i] = nn.Sequential(nn.Linear(input_size[i], output_size), nn.ReLU()).cuda()
        return domain_net

    def create_dec_domain_net(self, input_size, output_size):
        domain_net = {}
        for i in range(self.n_domain):
            domain_net[i] = nn.Sequential(nn.Linear(input_size, output_size[i]), nn.ReLU()).cuda()
        return domain_net

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, domain_in, domain_out):
        h = self.encoder(self.domain_encode_net[domain_in](x))
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        decoder = self.decoder(z)
        output = self.domain_decode_net[domain_out](decoder)
        return output, z, mu, logvar

    def reconstruction_loss(self, predict, label, loss):
        log_softmax_var = loss(predict)
        neg_ll = -torch.mean(torch.sum(log_softmax_var * label, dim=-1))
        return neg_ll


def train(data, op, model, device, loss_func):
    A_data = torch.from_numpy(data[0]).float().to(device)
    B_data = torch.from_numpy(data[1]).float().to(device)
    label = data[2]

    op.zero_grad()
    B_fake, z_B, mu_B, logvar_B = model(A_data, label[0], label[1])
    A_fake, z_A, mu_A, logvar_A = model(B_data, label[1], label[0])
    loss = model.reconstruction_loss(B_fake, B_data, loss_func) + \
           model.reconstruction_loss(A_fake, A_data, loss_func) + \
           0.5 * torch.mean(torch.sum(mu_A.pow(2) + logvar_A.exp() - logvar_A - 1, dim=-1)) + \
           0.5 * torch.mean(torch.sum(mu_B.pow(2) + logvar_B.exp() - logvar_B - 1, dim=-1))
    loss.backward()
    op.step()
    return loss.item()


def test(data, model, device):
    A_data = torch.from_numpy(data[0]).float().to(device)
    B_data = torch.from_numpy(data[1]).float().to(device)
    label = data[2]

    with torch.no_grad():
        B_fake, _, _, _ = model(A_data, label[0], label[1])
        A_fake, _, _, _ = model(B_data, label[1], label[0])
    return A_fake.cpu().detach().numpy(), B_fake.cpu().detach().numpy()


def main():
    iter = 100
    batch_size = 2000
    dataset = Dataset(["Health", "Clothing", "Grocery"])
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = VAE(dataset.input_size_list, [600, 200, 50], 3).to(device)
    loss_func = nn.LogSoftmax(dim=-1)
    result = [[0, 0], [0, 0]]

    for i in range(iter):
        domain, ids = dataset.random_iter(batch_size)
        shuffle_idx = np.random.permutation(len(domain))
        loss = 0
        for idx in shuffle_idx:
            data = dataset.get_batch_train(domain[idx], ids[idx])
            parameters = list(model.domain_encode_net[data[2][0]].parameters()) + \
                list(model.domain_encode_net[data[2][1]].parameters())+ \
                list(model.encoder.parameters()) + list(model.mu.parameters()) + list(model.logvar.parameters()) + \
                list(model.decoder.parameters()) + list(model.domain_decode_net[data[2][1]].parameters()) + \
                list(model.domain_decode_net[data[2][0]].parameters())
            op = torch.optim.Adam(parameters, lr=0.01)
            loss += train(data, op, model, device, loss_func)
        print(loss)

        data = dataset.get_batch_test(0, list(range(batch_size)))
        A_data, B_data = data[3], data[4]
        A_fake, B_fake = test(data, model, device)
        recall_A = calc_recall(A_fake, A_data, [50], "A")
        recall_B = calc_recall(B_fake, B_data, [50], "B")
        if recall_A > result[0][0]:
            result[0] = [recall_A, recall_B]
        if recall_B > result[1][1]:
            result[1] = [recall_A, recall_B]
        print("recall A: %f, recall B: %f"%(recall_A, recall_B))
    print(result)


if __name__ == '__main__':
    main()












