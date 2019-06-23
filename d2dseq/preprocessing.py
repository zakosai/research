import numpy as np
import pandas as pd
import gzip
import pickle
import os


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def write_file(filename, data):
    f = open(filename, "w")
    f.write("\n".join(data))
    f.close()


def convert_dataset(filename, user, item, dataset):
    f = open(filename, "w")
    f.write("user,item,score,time\n")

    dataset.sort_values(by=["unixReviewTime"])
    rating = [0]*len(user)

    for _, row in dataset.iterrows():
        uid = user.index(row.reviewerID)
        iid = item.index(row.asin)
        f.write("%d,%d,%d,%d\n" % (uid, iid, int(row.overall), row.unixReviewTime))
        if rating[uid] == 0:
            rating[uid] = [iid]
        else:
            rating[uid].append(iid)
    f.close()

    return rating


def statistic(data, f, name="A"):
    len_user = [len(i) for i in data]
    print("Dataset %s - min: %d - mean: %d - max: %d"%(name, min(len_user), np.mean(len_user), max(len_user)))
    f.write("\trating set %s - min: %d - mean: %d - max: %d\n" % (name, min(len_user), np.mean(len_user), max(len_user)))


def main(A, B):
    # read file
    dA = getDF("../cf-vae/data/%s/reviews.json.gz"%A)
    dB = getDF("../cf-vae/data/%s/reviews.json.gz"%B)

    # get mutual user and prune dataset
    user = list(set(dA.reviewerID.unique()) & set(dB.reviewerID.unique()))
    dA = dA[dA.reviewerID.isin(user)]
    dB = dB[dB.reviewerID.isin(user)]

    # get unique item and save
    save_dir = "data/%s_%s"%(A, B)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    item_A = dA.asin.unique().tolist()
    item_B = dB.asin.unique().tolist()
    write_file("%s/user.txt"%save_dir, user)
    write_file("%s/itemA.txt"%save_dir, item_A)
    write_file("%s/itemB.txt"%save_dir, item_B)

    # save dataset summary
    f = open("data/summary.txt", "a")
    f.write("--------------------------------------------------\n")
    f.write("Dataset: %s_%s\n"%(A, B))
    f.write("\tuser_no: %d - %s_no: %d - %s_no: %d\n"(len(user), A, len(item_A), B, len(item_B)))

    # convert user, item
    rating_A = convert_dataset("%s/ratingA.txt"%save_dir, user, item_A, dA)
    rating_B = convert_dataset("%s/ratingA.txt"%save_dir, user, item_A, dA)

    # statistic two new datasets
    statistic(rating_A, f, A)
    statistic(rating_B, f, B)
    f.close()

    # divide train, val, test id
    shuffle_id = np.random.permutation(len(user))
    test_id = shuffle_id[:int(len(user)*0.2)]
    val_id = shuffle_id[int(len(user)*0.2): int(len(user)*0.3)]

    # create dataset and save
    dataset = {'rating_A':rating_A,
               'rating_B': rating_B,
               'test_id':test_id,
               'val_id':val_id}
    f = open("%s/dataset.obj", "wb")
    pickle.dump(dataset, f)
    f.close()

if __name__ == '__main__':
    dataset = ["Health", "Grocery", "Kitchen", "Garden"]
    for i in range(len(dataset)):
        for j in range(i+1, len(dataset)):
            main(dataset[i], dataset[j])




