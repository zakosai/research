#!/usr/bin/env bash
dataset="Garden Automotive Instrument Grocery"
wsize="5"
for data in $dataset; do
    python user_seq2seq.py --data=$data --cat=True  --w_size=5 --n_layers=2 --model_type=bilstm
    python user_seq2seq.py --data=$data  --time=True --w_size=5 --n_layers=2 --model_type=bilstm
done


