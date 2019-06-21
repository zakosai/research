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

dataset="Baby Beauty CD Grocery Kitchen Outdoor Tool Toy Office"
wsize="5 10"
for data in $dataset; do
#    mkdir experiment/$data
#    mkdir experiment/$data/cvae
#    python seq2seq.py --data=$data/ratings --bilstm=False --n_layers=1
#    python seq2seq.py --data=$data/ratings --cat=True --time=True
#    for w in $wsize; do
#        python seq2seq.py --data=$data --bilstm=False --n_layers=1 --w_size=$w
#        python seq2seq.py --data=$data --w_size=$w
#        python seq2seq.py --data=$data --bilstm=False --n_layers=1 --w_size=$w --cat=True --time=True
#    done
    python multi-VAE.py --data=$data --iter=400
#    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
#    python vae.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae --type=user
#    python cvae_user.py --data_dir=data/$data --ckpt_folder=experiment/$data/cvae
#    python user_seq2seq.py --data=$data --cat=True --time=True --w_size=15 --batch_size=200
done

python user_seq2seq.py --data=book --time=True --w_size=20

python user_seq2seq.py --data=Tool --time=True --cat=True --w_size=5
python user_seq2seq.py --data=Tool --time=True --cat=True --w_size=10
python user_seq2seq.py --data=Toy --time=True --cat=True --w_size=5
python user_seq2seq.py --data=Toy --time=True --cat=True --w_size=10
