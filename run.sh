# download dataset into data folder
mkdir data
cd data
wget https://www.cse.scu.edu/~yfang/coen272/train.txt
wget https://www.cse.scu.edu/~yfang/coen272/test5.txt
wget https://www.cse.scu.edu/~yfang/coen272/test10.txt
wget https://www.cse.scu.edu/~yfang/coen272/test20.txt
# return to root
cd ../
# process data
python -m process_data.make_dict
# First algorithm is run in CF_1.ipynb

# My alogrithm
python -m process_data.make_train_valid_test


# PIN 3119481346890577

