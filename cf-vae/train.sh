stdbuf -oL python CCCFNET.py --A=Health --B=Clothing --k=50 |tee -a translation/Health_Clothing/CCCFNET.txt
stdbuf -oL python CCCFNET.py --A=Video --B=TV --k=50 |tee -a translation/Video_TV/CCCFNET.txt
stdbuf -oL python CCCFNET.py --A=Drama --B=Comedy --k=50 |tee -a translation/Drama_Comedy/CCCFNET.txt
stdbuf -oL python CCCFNET.py --A=Romance --B=Thriller --k=50 |tee -a translation/Romance_Thriller/CCCFNET.txt

stdbuf -oL python adversarial_personalized_ranking/AMF.py --path=data/Health_Clothing/ --dataset=Health_Clothing \
--adv_epoch=1000 --epochs=2000 --eps=0.5 --reg_adv=1 --ckpt=1 --verbose=20 |tee -a translation/Health_Clothing/AMF.txt

stdbuf -oL python adversarial_personalized_ranking/AMF.py --path=data/Video_TV/ --dataset=Video_TV \
--adv_epoch=1000 --epochs=2000 --eps=0.5 --reg_adv=1 --ckpt=1 --verbose=20 |tee -a translation/Video_TV/AMF.txt

stdbuf -oL python adversarial_personalized_ranking/AMF.py --path=data/Drama_Comedy/ --dataset=Drama_Comedy \
--adv_epoch=1000 --epochs=2000 --eps=0.5 --reg_adv=1 --ckpt=1 --verbose=20 |tee -a translation/Drama_Comedy/AMF.txt

stdbuf -oL python adversarial_personalized_ranking/AMF.py --path=data/Romance_Thriller/ --dataset=Romance_Thriller \
--adv_epoch=1000 --epochs=2000 --eps=0.5 --reg_adv=1 --ckpt=1 --verbose=20 |tee -a translation/Romance_Thriller/AMF.txt


python train_dae.py --ckpt_folder=wae/Baby/8/ --data_dir=data2/Baby/ --zdim=100 \
--data_type=8 --type=text
python train_cf_dae.py --model=0 --ckpt_folder=wae/Baby/8/ --data_dir=data2/Baby/ \
--iter=50 --data_type=8 --user_no=2792 --item_no=4897 --gridsearch=1 --zdim=100
python wae.py --ckpt_folder=wae/Baby/8/ --data_dir=data2/Baby/ --zdim=100 \
--data_type=8 --type=text
python train_cf_wae.py --model=0 --ckpt_folder=wae/Baby/8/ --data_dir=data2/Baby/ \
--iter=50 --data_type=8 --user_no=2792 --item_no=4897 --gridsearch=1 --zdim=100


folders='Clothing Kindle Phone Video Pet Music Instrument Automotive Garden Electronics Books'
rate="1 8"
for f in $folders;
do
    mkdir wae/$f
    user_no="$(sed -n '1p' data2/$f/info.txt)"
    item_no="$(sed -n '2p' data2/$f/info.txt)"
    for r in $rate;
    do
        mkdir wae/$f/$r
        ckpt=wae/$f/$r
        python train_vae.py --ckpt_folder=$ckpt --data_dir=data2/$f/ --zdim=100 \
        --data_type=$r --type=text
        python train_cvae_extend.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/$f/ \
        --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no --gridsearch=1 --zdim=100

        python train_dae.py --ckpt_folder=$ckpt --data_dir=data2/$f/ --zdim=100 \
        --data_type=$r --type=text
        python train_cf_dae.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/$f/ \
        --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no --gridsearch=1 --zdim=100

        python wae.py --ckpt_folder=$ckpt --data_dir=data2/$f/ --zdim=100 \
        --data_type=$r --type=text
        python train_cf_wae.py --model=0 --ckpt_folder=$ckpt --data_dir=data2/$f/ \
        --iter=50 --data_type=$r --user_no=$user_no --item_no=$item_no --gridsearch=1 --zdim=100
    done
done



