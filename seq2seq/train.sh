#!/usr/bin/env bash
dataset="Pet Phone"
for data in $dataset; do
    python user_seq2seq.py --data=$data --cat=True --time=True --w_size=7 --batch_size=500
done

dataset="TV CD Kitchen Kindle Health Electronics"
wsize="5 10"
for data in $dataset; do
    for w in $wsize; do
        python seq2seq.py --data=$data --bilstm=False --n_layers=1 --w_size=$w
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --batch_size=500
    done
    python multi-VAE.py --data=$data
    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae --type=user
    python cvae_user.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
done


