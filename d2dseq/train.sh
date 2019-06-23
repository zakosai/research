#!/usr/bin/env bash
dataset="Health_Grocery Health_Kitchen Health_Garden Grocery_Garden Grocery_Kitchen Kitchen_Garden"
for data in $dataset; do
    echo $data
    mkdir experiment/$data
    python d2d_seq.py --data=$data --batch_size=100 --iter=50
done