
python train_cvae_extend.py --model=0 --ckpt_folder=translation/Health_Clothing/ --data_dir=data/Health_Clothing/ \
--iter=50 --data_type=70 --user_no=6557 --item_no=34295 --gridsearch=1 --zdim=100

python train_cf_dae.py --model=0 --ckpt_folder=translation/Health_Clothing/ --data_dir=data/Health_Clothing/ \
--iter=50 --data_type=70 --user_no=6557 --item_no=34295 --gridsearch=1 --zdim=100

#python train_cvae_extend.py --model=0 --ckpt_folder=translation/Grocery_Health/ --data_dir=data/Grocery_Health/ \
#--iter=50 --data_type=70 --user_no=6848 --item_no=23448 --gridsearch=1 --zdim=100
#
#python train_vae.py --ckpt_folder=translation/Video_TV/ --data_dir=data/Video_TV/ --zdim=100 \
#--data_type=70 --type=text
#python train_cvae_extend.py --model=0 --ckpt_folder=translation/Video_TV/ --data_dir=data/Video_TV/ \
#--iter=50 --data_type=70 --user_no=5459 --item_no=38650 --gridsearch=1 --zdim=100
#
#python train_cf_dae.py --model=0 --ckpt_folder=translation/Video_TV/ --data_dir=data/Video_TV/ \
#--iter=50 --data_type=70 --user_no=5459 --item_no=38650 --gridsearch=1 --zdim=100