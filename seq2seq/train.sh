#!/usr/bin/env bash
#dataset="Garden Automotive Beauty Grocery Outdoor Tool Toy Office Phone Video"
dataset="Beauty"
wsize="5"
for data in $dataset; do
    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=lstm
    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=lstm
    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=2 --model_type=gru
    python user_seq2seq.py --data=$data --cat=True  --time=True --w_size=5 --n_layers=1 --model_type=gru
    python seq2seq.py --data=$data --cat=True --time=True --w_size=5 --n_layers=2 --model_type=bilstm
    python seq2seq.py --data=$data --cat=True --time=True --w_size=5 --n_layers=2 --model_type=bigru
done

