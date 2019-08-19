#!/usr/bin/env bash
#dataset="Tool Toy Office Phone Video"
##dataset="Office"
#wsize="5"
#for data in $dataset; do
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=lstm --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=lstm --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=gru --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=gru --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=bilstm --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=bilstm --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=bigru --iter=500
#    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=bigru --iter=500
##    python seq2seq.py --data=$data --cat=True --time=True --w_size=5 --n_layers=2 --model_type=bilstm
##    python seq2seq.py --data=$data --cat=True --time=True --w_size=5 --n_layers=2 --model_type=bigru
#done
#
##    python user_seq2seq.py --data=Garden --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=bigru --iter=500

python vae.py --data_dir=data/Tool --ckpt_folder=experiment/Tool
mv experiment/Tool/text.npy data/Tool/
python user_seq2seq.py --data=Tool --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=bigru --iter=500