#!/usr/bin/env bash


folder="Kindle CD Health TV Beauty Tool Toy Outdoor Video Clothing Baby Office Electronics Books"
for f in $folder;
do
    mkdir data/$f
    python dataset.py --data=../cf-vae/data/$f/reviews.json.gz --output_folder=data/$f/
done

folder="Kitchen AmzVideo Instrument Music Automotive Pet App Phone Kindle CD Health TV Beauty Tool Toy Outdoor Video
Clothing Baby Office Electronics Books"
for f in $folder;
do
    python tf_idf.py --data=data/$f/dataset.pkl --deep=True --output=data/$f/
done