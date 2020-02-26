import numpy as np
import argparse


def create_dataset(A="Health", B="Clothing"):
    dense_A = read_data("data/%s_%s/%s_user_product.txt"%(A,B,A))
    num_A = 0
    for i in dense_A:
        if num_A < max(i):
            num_A = max(i)
    num_A += 1
    user_A = one_hot_vector(dense_A, num_A)

    dense_B = read_data("data/%s_%s/%s_user_product.txt"%(A, B, B))
    num_B = 0
    for i in dense_B:
        if num_B < max(i):
            num_B = max(i)
    num_B += 1
    user_B = one_hot_vector(dense_B, num_B)

    return user_A, user_B, dense_A, dense_B, num_A, num_B


def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f


def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j< num_product:
                one_hot_A[i,j] = 1
    return one_hot_A


def main(args):
    iter = 300
    batch_size = 500
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/"%(A,B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)

    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    # user_A = user_A[perm]
    # user_B = user_B[perm]

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]
    item_A_popular = user_A_train.mean(axis=0)
    item_B_popular = user_B_train.mean(axis=0)

    print("A: Train: %f, Test: %f" %((user_A_train * item_A_popular).mean(axis=1).mean(),
                                     (user_A_test * item_A_popular).mean(axis=1).mean()))
    print("B: Train: %f, Test: %f" % ((user_B_train * item_B_popular).mean(axis=1).mean(),
                                      (user_B_test * item_B_popular).mean(axis=1).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--A', type=str, default="Health", help='domain A')
    parser.add_argument('--B', type=str, default='Grocery', help='domain B')
    parser.add_argument('--k', type=int, default=100, help='top-K')
    args = parser.parse_args()

    main(args)

