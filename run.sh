# PIN 3119481346890577
# download dataset into data folder
mkdir data data/para
cd data
wget https://www.cse.scu.edu/~yfang/coen272/train.txt
wget https://www.cse.scu.edu/~yfang/coen272/test5.txt
wget https://www.cse.scu.edu/~yfang/coen272/test10.txt
wget https://www.cse.scu.edu/~yfang/coen272/test20.txt
# return to root
cd ../
# process data
python -m process_data.make_dict

# generate score5_cf1.txt score10_cf1.txt score20_cf1.txt
# Best score 0.831185355442456
python cf_1.py

# My own alogrithm
# neural network
# generate nn_test5.txt nn_test10.txt nn_test20.txt
# Best score 0.77314890822525
python -m process_data.make_train_valid_test
python train_nn.py --epoch=1
python predict_nn.py --epoch=0
# matrix factorization
# generate mf_test5.txt mf_test10.txt mf_test20.txt
# Best score 0.864636348711213
python -m process_data.make_mf_data
python train_mf.py --epoch=5
python predict_mf.py --epoch=0

