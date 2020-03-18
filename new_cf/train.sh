
python multiVAE.py --data_dir=../cf-vae/data2/Health/ --data_type=8p --iter=100  >> /media/linh/DATA/research/new_cf/result/multi_health_8.txt
python triple_vae.py --data_dir=../cf-vae/data2/Health/ --data_type=8p  --iter=100 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result/health_8.txt
python multiVAE.py --data_dir=../cf-vae/data2/Health/ --data_type=1p --iter=100  >> /media/linh/DATA/research/new_cf/result/multi_health_1.txt
python triple_vae.py --data_dir=../cf-vae/data2/Health/ --data_type=1p  --iter=100  >> /media/linh/DATA/research/new_cf/result/health_1.txt
python multiVAE.py --data_dir=../cf-vae/data2/Outdoor/ --data_type=8p --iter=100 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result/multi_outdoor_8.txt
python triple_vae.py --data_dir=../cf-vae/data2/Outdoor/ --data_type=8p  --iter=100 --learning_rate=0.001 >> /media/linh/DATA/research/new_cf/result/outdoor_8.txt
