folders='Health Toy TV CD Tool Beauty Kitchen Office Grocery Baby Clothing Kindle Phone Video Pet Music Instrument
Automotive Garden Electronics Books'
rate="1 8"
u_no=("5584" "7981")
i_no=("13790" "19184")
u_dim=("786" "896")

for f in $folders;
do
    mkdir wae/${dir[$i]}
    user_no="$(sed -n '1p' data2/$f/info.txt)"
    item_no="$(sed -n '2p' data2/$f/info.txt)"
    for r in $rate;
    do
        mkdir wae/$f/$r
        ckpt=wae/$f/$r
        python wae.py --ckpt_folder=$ckpt --data_dir=data2/$f/ --zdim=100 \
        --data_type=$r --type=text
        python train_cf_wae.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/$f/ \
        --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no --gridsearch=1 --zdim=100
#        python wae.py --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ --zdim=50 \
#        --data_type=$r --type=user --user_dim=${u_dim[$i]}
#        python train_cwae_user.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ \
#        --iter=50 --data_type=$r --user_no=${u_no[$i]} --item_no=${i_no[$i]} --gridsearch=1 --zdim=50 \
#        --user_dim=${u_dim[$i]}
    done
done



