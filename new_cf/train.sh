python multiVAE.py --data_dir=../cf-vae/data2/TV/ --data_type=8p --iter=100 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result/multi_tv_8.txt
python triple_vae.py --data_dir=../cf-vae/data2/TV/ --data_type=8p  --iter=100 --learning_rate = 0.001 >> /media/linh/DATA/research/new_cf/result/tv_8.txt
python multiVAE.py --data_dir=../cf-vae/data2/Toy/ --data_type=1p --iter=100 >> /media/linh/DATA/research/new_cf/result/multi_toy_1.txt
python triple_vae.py --data_dir=../cf-vae/data2/Toy/ --data_type=1p  --iter=100 >> /media/linh/DATA/research/new_cf/result/toy_1.txt
