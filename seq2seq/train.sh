#!/usr/bin/env bash

#python cf_vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
#source ../.env3/bin/activate
type="Office CD Grocery Kitchen Outdoor"
for t in $type; do
    mkdir experiment/$t
    python hybrid_seq2seq.py --data=$t --w_size=4
    python seq2seq.py --data=$t --w_size=4
done