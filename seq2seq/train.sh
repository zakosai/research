#!/usr/bin/env bash

python cf_vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
source ../.env3/bin/activate
python hybrid_seq2seq.py --data=ml-1m --num_p=3706
python seq2seq.py --data=ml-1m --num_p=3706

