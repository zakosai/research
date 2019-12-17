python -u translation2.py --A=Video --B=TV |tee -a translation/Video_TV/d2d.log
python -u multi_VAE_single.py --A=Video --B=TV |tee -a translation/Video_TV/multiVAE.log

python -u translation2.py --A=Drama --B=Comedy |tee -a translation/Drama_Comedy/d2d.log
python -u multi_VAE_single.py --A=Drama --B=Comedy |tee -a translation/Drama_Comedy/multiVAE.log


python -u translation2.py --A=Romance --B=Thriller |tee -a translation/Romance_Thriller/d2d.log
python -u multi_VAE_single.py --A=Romance --B=Thriller |tee -a translation/Romance_Thriller/multiVAE.log

python cdae_new.py --ckpt_folder=translation/Health_Clothing --data_dir=data/Health_Clothing \
--data_type=Health --item_no=18226 --user_no=6557 --gridsearch=1

python cdae_new.py --ckpt_folder=translation/Health_Clothing --data_dir=data/Health_Clothing \
--data_type=Clothing --item_no=16069 --user_no=6557 --gridsearch=1

python train_dae.py --ckpt_folder=translation/Video_TV --data_dir=data/Video_TV
python cdae_new.py --ckpt_folder=translation/Video_TV --data_dir=data/Video_TV \
--data_type=Video --item_no=10072 --user_no=5459 --gridsearch=1
python cdae_new.py --ckpt_folder=translation/Video_TV --data_dir=data/Video_TV \
--data_type=TV --item_no=28578 --user_no=5459 --gridsearch=1

python train_dae.py --ckpt_folder=translation/Drama_Comedy --data_dir=data/Drama_Comedy
python cdae_new.py --ckpt_folder=translation/Drama_Comedy --data_dir=data/Drama_Comedy \
--data_type=Drama --item_no=1490 --user_no=6023 --gridsearch=1
python cdae_new.py --ckpt_folder=translation/Drama_Comedy --data_dir=data/Drama_Comedy \
--data_type=Comedy --item_no=1081 --user_no=6023 --gridsearch=1

python train_dae.py --ckpt_folder=translation/Romance_Thriller --data_dir=data/Romance_Thriller
python cdae_new.py --ckpt_folder=translation/Romance_Thriller --data_dir=data/Romance_Thriller \
--data_type=Romance --item_no=455 --user_no=5891 --gridsearch=1
python cdae_new.py --ckpt_folder=translation/Romance_Thriller --data_dir=data/Romance_Thriller \
--data_type=Thriller --item_no=475 --user_no=5891 --gridsearch=1
