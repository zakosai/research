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


class GAN(nn.Module):
    def __init__(self, list_inputs, layers, n_domain):
        super(GAN, self).__init__()
        self.n_domain = n_domain
        self.net = {}
        for i in range(len(list_inputs)):
            self.net[i] = nn.Sequential(*self.create_net(list_inputs[i], layers)).cuda()

    def create_net(self, input, layers):
        sequence = []
        prev = input
        for layer in layers:
            sequence.append(nn.Linear(prev, layer))
            sequence.append(nn.ReLU())
            prev = layer
        return sequence

    def forward(self, x, domain_name):
        pred = self.net[domain_name](x)
        return pred


def train(data, op, model, device, loss_func):
    A_data = torch.from_numpy(data[0]).float().to(device)
    B_data = torch.from_numpy(data[1]).float().to(device)
    label = data[2]

    # Generator
    op['gen'].zero_grad()
    B_fake, z_B, mu_B, logvar_B = model['VAE'](A_data, label[0], label[1])
    A_fake, z_A, mu_A, logvar_A = model['VAE'](B_data, label[1], label[0])
    loss = model['VAE'].reconstruction_loss(B_fake, B_data, loss_func['recon']) + \
           model['VAE'].reconstruction_loss(A_fake, A_data, loss_func['recon']) + \
           0.5 * torch.mean(torch.sum(mu_A.pow(2) + logvar_A.exp() - logvar_A - 1, dim=-1)) + \
           0.5 * torch.mean(torch.sum(mu_B.pow(2) + logvar_B.exp() - logvar_B - 1, dim=-1))

    out_GAN_A_fake = model['GAN'](A_fake, label[0]).view(-1)
    out_GAN_B_fake = model['GAN'](B_fake, label[1]).view(-1)
    gen_loss = loss_func['gan'](out_GAN_A_fake, torch.ones_like(out_GAN_A_fake)) + \
            loss_func['gan'](out_GAN_B_fake, torch.ones_like(out_GAN_B_fake))
    loss += gen_loss

    loss.backward()
    op['gen'].step()

    # Discriminator
    op['dis_A'].zero_grad()
    A_fake, z_A, mu_A, logvar_A = model['VAE'](B_data, label[1], label[0])
    out_GAN_A_fake = model['GAN'](A_fake, label[0]).view(-1)
    out_GAN_A_real = model['GAN'](A_data, label[0]).view(-1)

    dis_loss = loss_func['gan'](out_GAN_A_fake, torch.zeros_like(out_GAN_A_fake)) + \
               loss_func['gan'](out_GAN_A_real, torch.ones_like(out_GAN_A_real))
    dis_loss.backward()
    op['dis_A'].step()

    op['dis_B'].zero_grad()
    B_fake, z_B, mu_B, logvar_B = model['VAE'](A_data, label[0], label[1])
    out_GAN_B_fake = model['GAN'](B_fake, label[1]).view(-1)
    out_GAN_B_real = model['GAN'](B_data, label[1]).view(-1)

    dis_loss = loss_func['gan'](out_GAN_B_fake, torch.zeros_like(out_GAN_B_fake)) + \
               loss_func['gan'](out_GAN_B_real, torch.ones_like(out_GAN_B_real))
    dis_loss.backward()
    op['dis_B'].step()

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

    model = {}
    model['VAE'] = VAE(dataset.input_size_list, [600, 200, 50], 3).to(device)
    model['GAN'] = GAN(dataset.input_size_list, [50, 1], 3).to(device)

    loss_func = {}
    loss_func['recon'] = nn.LogSoftmax(dim=-1)
    loss_func['gan'] = nn.BCELoss()
    result = [[0, 0], [0, 0]]

    op = {}

    for i in range(iter):
        domain, ids = dataset.random_iter(batch_size)
        shuffle_idx = np.random.permutation(len(domain))
        loss = 0
        for idx in shuffle_idx:
            data = dataset.get_batch_train(domain[idx], ids[idx])
            parameters_vae = list(model['VAE'].domain_encode_net[data[2][0]].parameters()) + \
                list(model['VAE'].domain_encode_net[data[2][1]].parameters())+ \
                list(model['VAE'].encoder.parameters()) + list(model['VAE'].mu.parameters()) + list(model['VAE'].logvar.parameters()) + \
                list(model['VAE'].decoder.parameters()) + list(model['VAE'].domain_decode_net[data[2][1]].parameters()) + \
                list(model['VAE'].domain_decode_net[data[2][0]].parameters())
            op['gen'] = torch.optim.Adam(parameters_vae, lr=0.01)
            op['dis_A'] = torch.optim.Adam(model['GAN'].net[data[2][0]].parameters(), lr=0.01)
            op['dis_B'] = torch.optim.Adam(model['GAN'].net[data[2][1]].parameters(), lr=0.01)

            loss += train(data, op, model, device, loss_func)
        print(loss)

        data = dataset.get_batch_test(1, list(range(batch_size)))
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












