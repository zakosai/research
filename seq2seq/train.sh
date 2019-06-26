#!/usr/bin/env bash
dataset="Garden Automotive Beauty Grocery Outdoor Tool Toy Office"
wsize="5 10"
for data in $dataset; do
    mkdir experiment/$data
    mkdir experiment/$data/cvae
    python preprocessing.py --data=$data
#    python seq2seq.py --data=$data/ratings --bilstm=False --n_layers=1
#    python seq2seq.py --data=$data/ratings --cat=True --time=True
    for w in $wsize; do
        python seq2seq.py --data=$data --bilstm=False --n_layers=1 --w_size=$w
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w
    done
    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae --type=user
    python cvae_user.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
#    python user_seq2seq.py --data=$data --cat=True --time=True --w_size=15 --batch_size=200
done

