from vae_unet import Translation, calc_recall
import tensorflow as tf
import numpy as np
import argparse
import os


def main():
    iter = 3000
    batch_size = 500
    args = parser.parse_args()
    f = open(args.data, 'rb')
    dataset = pickle.load(f)
    forder = args.data.split("/")[:-1]
    forder = "/".join(forder)
    content = np.load(os.path.join(forder, "item_tag.npz"))
    content = content['z']
    # content = load_npz(os.path.join(forder, "mult_nor.npz"))
    # content = content.toarray()

    num_p = dataset['item_no']
    num_u = dataset['user_no']
    encoding_dim = [600, 200]
    decoding_dim = [200, 600, num_p]

    z_dim = 50
    max_item = max(np.sum(dataset['user_onehot'], axis=1))
    x_dim = z_dim * max_item
    user_item = np.zeros((num_u, x_dim))
    for i in range(num_u):
        idx = np.where(dataset['user_onehot'][i] == 1)
        u_c = content[idx]
        u_c = u_c.flatten()
        user_item[i, :len(u_c)] = u_c

    model = Translation(batch_size, x_dim, num_p, encoding_dim, decoding_dim, z_dim)
    model.build_model()

    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    checkpoint_dir = args.ckpt
    saver.restore(sess, os.path.join(checkpoint_dir, args.ckpt))
    x = user_item[dataset['user_item_test'].keys()]
    y = dataset['user_onehot'][dataset['user_item_test'].keys()]
    item_pred = sess.run(model.x_recon,
                         feed_dict={model.x: x, model.y: y})
    _, result = calc_recall(item_pred, dataset['user_item_test'].values(),
                            [10, 20, 30, 40, 50], "item")






parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data',  type=str, default="Tool",
                   help='dataset name')
parser.add_argument('--ckpt',  type=str, default="experiment/delicious",
                   help='1p or 8p')
parser.add_argument('--num_p', type=int, default=7780, help='number of product')


if __name__ == '__main__':
    main()