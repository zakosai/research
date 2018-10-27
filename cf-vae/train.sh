mkdir translation/Drama_Comedy
python train_vae.py --ckpt_folder=translation/Drama_Comedy/ --data_dir=data/Drama_Comedy/ --zdim=50 \
--data_type=70 --type=text
python train_cvae_extend.py --model=0 --ckpt_folder=translation/Drama_Comedy/ --data_dir=data/Drama_Comedy/ \
--iter=50 --data_type=70 --user_no=6023 --item_no=2577 --gridsearch=1 --zdim=50
python train_dae.py --ckpt_folder=translation/Drama_Comedy/ --data_dir=data/Drama_Comedy/ --zdim=50 \
--data_type=70 --type=text
python train_cf_dae.py --model=0 --ckpt_folder=translation/Drama_Comedy/ --data_dir=data/Drama_Comedy/ \
--iter=50 --data_type=70 --user_no=6023 --item_no=2577 --gridsearch=1 --zdim=50

mkdir translation/Romance_Thriller

python train_vae.py --ckpt_folder=translation/Romance_Thriller/ --data_dir=data/Romance_Thriller/ --zdim=50 \
--data_type=70 --type=text
python train_cvae_extend.py --model=0 --ckpt_folder=translation/Romance_Thriller/ --data_dir=data/Romance_Thriller/ \
--iter=50 --data_type=70 --user_no=5875 --item_no=930 --gridsearch=1 --zdim=50
python train_dae.py --ckpt_folder=translation/Romance_Thriller/ --data_dir=data/Romance_Thriller/ --zdim=50 \
--data_type=70 --type=text
python train_cf_dae.py --model=0 --ckpt_folder=translation/Romance_Thriller/ --data_dir=data/Romance_Thriller/ \
--iter=50 --data_type=70 --user_no=5875 --item_no=930 --gridsearch=1 --zdim=50


#mkdir translation2/Health_Grocery
#
#python train_vae.py --ckpt_folder=translation2/Health_Grocery/ --data_dir=data/Health_Grocery/ --zdim=50 \
#--data_type=70 --type=text
#python train_cvae_extend.py --model=0 --ckpt_folder=translation2/Health_Grocery/ --data_dir=data/Health_Grocery/ \
#--iter=50 --data_type=70 --user_no=6848 --item_no=23448 --gridsearch=1 --zdim=50
#python train_dae.py --ckpt_folder=translation2/Health_Grocery/ --data_dir=data/Health_Grocery/ --zdim=50 \
#--data_type=70 --type=text
#python train_cf_dae.py --model=0 --ckpt_folder=translation2/Health_Grocery/ --data_dir=data/Health_Grocery/ \
#--iter=50 --data_type=70 --user_no=6848 --item_no=23448 --gridsearch=1 --zdim=50

mkdir translation2/Health_Beauty

python train_vae.py --ckpt_folder=translation2/Health_Beauty/ --data_dir=data/Health_Beauty/ --zdim=50 \
--data_type=70 --type=text
python train_cvae_extend.py --model=0 --ckpt_folder=translation2/Health_Beauty/ --data_dir=data/Health_Beauty/ \
--iter=50 --data_type=70 --user_no=7026 --item_no=26248 --gridsearch=1 --zdim=50
python train_dae.py --ckpt_folder=translation2/Health_Beauty/ --data_dir=data/Health_Beauty/ --zdim=50 \
--data_type=70 --type=text
python train_cf_dae.py --model=0 --ckpt_folder=translation2/Health_Beauty/ --data_dir=data/Health_Beauty/ \
--iter=50 --data_type=70 --user_no=7026 --item_no=26248 --gridsearch=1 --zdim=50

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