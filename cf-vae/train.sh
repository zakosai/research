dir=("Tool" "Outdoor" "Health" "Kitchen")
rate="1 8"
u_no=("2118" "4062" "5584" "7981")
i_no=("7780" "11560" "13790" "19184")
u_dim=("83" "994" "786" "896")
mkdir wae

for i in `seq 0 3`;
do
    mkdir wae/${dir[$i]}
    for r in $rate;
    do
        mkdir wae/${dir[$i]}/$r
        ckpt=wae/${dir[$i]}/$r
        python wae.py --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ --zdim=50 \
        --data_type=$r --type=text
        python train_cf_wae.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ \
        --iter=50 --data_type=$r --user_no=${u_no[$i]} --item_no=${i_no[$i]} --gridsearch=1 --zdim=50
        python wae.py --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ --zdim=50 \
        --data_type=$r --type=user
        python train_cwae_user.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/${dir[$i]}/ \
        --iter=50 --data_type=$r --user_no=${u_no[$i]} --item_no=${i_no[$i]} --gridsearch=1 --zdim=50 \
        --user_dim=${u_dim[$i]}
    done
done



