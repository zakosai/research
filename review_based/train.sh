#!/usr/bin/env bash


folder="Kitchen AmzVideo Instrument Music Automotive Pet App Phone Kindle CD Health TV Beauty Tool Toy Outdoor Video
Clothing Baby Office Electronics Books"
for f in $folder;
do
    mkdir data/$f
    python dataset.py --data=$f
done
#multi=(1.220 0.997 0.923 0.970 0.861 1.328 1.494 1.413 0.775 1.005 1.238 1.144 1.387 1.096 0.973 0.980 1.257 1.187 1.304 0.779 1.350)
#folder="Kitchen AmzVideo Instrument Music Automotive Pet App Phone Kindle CD Health TV Beauty Tool Toy Outdoor
#Video Clothing Baby Office Electronics"
#i=0
#for f in $folder;
#do
#
#    python tf_idf.py --data=data/$f/dataset.pkl --deep=True --output=data/$f/ --multi=${multi[($i)]}
#    i=$((i+1))
#done

