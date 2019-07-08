#!/usr/bin/env bash
dataset="Pet Clothing Phone"
for data in $dataset; do
    python user_seq2seq.py --data=$data --cat=True --time=True --w_size=4
done

dataset="TV CD Kitchen Kindle Health App Electronics Books"
wsize="5 10"
for data in $dataset; do
    for w in $wsize; do
        python seq2seq.py --data=$data --bilstm=False --n_layers=1 --w_size=$w
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w
    done
    python multi-VAE.py --data=$data
done
    python cvae_user.py --data_dir=data/Phone --ckpt_folder=experiment/Phone/cvae


