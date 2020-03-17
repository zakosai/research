python multiVAE.py --data_dir=../cf-vae/data/Tool/ --data_type=8p --iter=200 >> result/tool_8.txt
python multiVAE.py --data_dir=../cf-vae/data/Outdoor/ --data_type=8p --iter=200 >> result/outdoor_8.txt
python multiVAE.py --data_dir=../cf-vae/data/Outdoor/ --data_type=1p --iter=200 >> result/outdoor_1.txt
python multiVAE.py --data_dir=../cf-vae/data/Health/ --data_type=1 --iter=200 >> result/health_1.txt
python multiVAE.py --data_dir=../cf-vae/data/Health/ --data_type=80p --iter=200 >> result/health_80.txt
python multiVAE.py --data_dir=../cf-vae/data/kitchen/ --data_type=1 --iter=200 >> result/kitchen_1.txt
python multiVAE.py --data_dir=../cf-vae/data/kitchen/ --data_type=8 --iter=200 >> result/kitchen_8.txt

