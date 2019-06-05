#!/usr/bin/env bash

#python cf_vae.py --ckpt_folder=experiment/ml-1m/cvae_100 --data_dir=data/ml-1m --zdim=100
#source ../.env3/bin/activate

#python seq2seq.py --data=ml-1m --w_size=15 --bilstm=False --n_layers=1
#python seq2seq.py --data=ml-1m --w_size=15
##python seq2seq.py --data=ml-1m --w_size=15 --cat=True
#python seq2seq.py --data=ml-1m --w_size=15 --cat=True --time=True
##python seq2seq.py --data=ml-1m --w_size=15 --time=True
#
#python user_seq2seq.py --data=ml-1m --w_size=15 --cat=True
#python user_seq2seq.py --data=ml-1m --w_size=15 --cat=True --time=True
#python user_seq2seq.py --data=ml-1m --w_size=15 --time=True
#python user_seq2seq.py --data=ml-1m --w_size=15
#
#python hybrid_seq2seq.py --data=ml-1m --w_size=15
#python hybrid_seq2seq.py --data=ml-1m --w_size=15 --cat=True --time=True
#
#type="Garden Outdoor"
#for t in $type; do
#    python seq2seq.py --data=$t --w_size=4 --bilstm=False --n_layers=1
#    python seq2seq.py --data=$t --w_size=4
#    #python seq2seq.py --data=ml-1m --w_size=15 --cat=True
#    python seq2seq.py --data=$t --w_size=4 --cat=True --time=True
#    #python seq2seq.py --data=ml-1m --w_size=15 --time=True
#
#    python user_seq2seq.py --data=$t --w_size=4 --cat=True
#    python user_seq2seq.py --data=$t --w_size=4 --cat=True --time=True
#    python user_seq2seq.py --data=$t --w_size=4 --time=True
#    python user_seq2seq.py --data=$t --w_size=4
#
#    python hybrid_seq2seq.py --data=$t --w_size=4
#    python hybrid_seq2seq.py --data=$t --w_size=4 --cat=True --time=True
#done

#python preprocessing.py

dataset="Garden Office Kitchen CD"
for data in $dataset; do
    mkdir experiment/$data/ratings
    python vae.py --data_dir=data/$data/ratings --ckpt_folder=experiment/$data/ratings
    python vae.py --data_dir=data/$data/ratings --ckpt_folder=experiment/$data/ratings --type=user
    python cvae_user.py --data_dir=data/$data/ratings --ckpt_folder=experiment/$data/ratings
#    python seq2seq.py --data=$data/ratings --bilstm=False --n_layers=1
#    python seq2seq.py --data=$data/ratings --cat=True --time=True
#    python user_seq2seq.py --data=$data/ratings --cat=True --time=True
done


