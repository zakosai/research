import numpy as np


class Dataset:
    def __init__(self, domain_list):
        self.in_data_train, self.in_data_test = [], []
        self.out_data_train, self.out_data_test = [], []
        self.label = []
        self.input_size_list = []
        source_dir = '_'.join(domain_list)

        for i in range(len(domain_list)-1):
            for j in range(i+1, len(domain_list)):
                in_data_train, in_data_test, item_no = self.read_data("data/%s/%s_%s_user_product.txt"%
                                                   (source_dir, domain_list[i], domain_list[j]))
                if i == 0 and j == 1:
                    self.input_size_list.append(item_no)
                out_data_train, out_data_test, item_no = self.read_data("data/%s/%s_%s_user_product.txt"%
                                                    (source_dir, domain_list[j], domain_list[i]))
                if i == 0:
                    self.input_size_list.append(item_no)
                self.label.append([i, j])
                self.in_data_train.append(in_data_train)
                self.out_data_train.append(out_data_train)
                self.in_data_test.append(in_data_test)
                self.out_data_test.append(out_data_test)

    def read_data(self, dataset):
        data = list(open(dataset))
        data = [d.strip().split(' ') for d in data]
        data = [[int(i) for i in d[1:]] for d in data]
        item_no = max([max(d) for d in data]) + 1
        one_hot = np.zeros((len(data), item_no))
        for j, d in enumerate(data):
            for i in d:
                one_hot[j, i] = 1

        one_hot_train = one_hot[:int(len(one_hot)*0.8), :]
        one_hot_test = one_hot[int(len(one_hot)*0.8):, :]
        return one_hot_train, one_hot_test, item_no

    def random_iter(self, batch_size):
        domain = []
        ids = []
        for i in range(len(self.label)):
            shuffle_idx = np.random.permutation(range(len(self.in_data_train[i])))
            for j in range(len(shuffle_idx)//batch_size+1):
                domain.append(i)
                ids.append(shuffle_idx[i*batch_size:(i+1)*batch_size])
        return domain, ids

    def get_batch_train(self, domain_id, user_ids):
        return (self.in_data_train[domain_id][user_ids],
                self.out_data_train[domain_id][user_ids],
                self.label[domain_id])

    def get_batch_test(self, domain_id, user_ids):
        return (self.in_data_test[domain_id][user_ids],
                self.out_data_test[domain_id][user_ids],
                self.label[domain_id])


