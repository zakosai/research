#!/usr/bin/env bash
python user_seq2seq.py --data=Beauty --cat=True --time=True --w_size=5 --n_layers=2 --model_type=bilstm
python user_seq2seq.py --data=Beauty --cat=True --time=True --w_size=5 --n_layers=1 --model_type=bilstm
python user_seq2seq.py --data=Beauty --cat=True --time=True --w_size=10 --n_layers=2 --model_type=bilstm
python user_seq2seq.py --data=Beauty --cat=True --time=True --w_size=10 --n_layers=1 --model_type=bilstm
dataset="Grocery Outdoor Tool Toy Office Pet Music Instrument Clothing Video Phone"
wsize="5"
for data in $dataset; do
    for w in $wsize; do
        python seq2seq.py --data=$data --n_layers=1 --w_size=$w
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --n_layers=2 --model_type=bilstm
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --n_layers=1 --model_type=bilstm
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --model_type=bigru --n_layers=2
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --model_type=bigru --n_layers=1
        python user_seq2seq.py --data=$data --cat=True --time=True --w_size=$w --model_type=gru --n_layers=1

    done
done

dataset="Baby TV CD Kitchen Kindle Health Electronics"
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


