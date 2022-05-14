import pandas as pd
import pickle
import os
from tqdm import tqdm
import random
import numpy as np
random.seed(7)

dpath = 'data'
max_l = 20
pad = (0, 3)
ratio = 10

udict = pickle.load(open(os.path.join(dpath, 'udict.pkl'), 'rb'))
mdict = pickle.load(open(os.path.join(dpath, 'mdict.pkl'), 'rb'))
test5 = pickle.load(open(os.path.join(dpath, 'test5.pkl'), 'rb'))
test10 = pickle.load(open(os.path.join(dpath, 'test10.pkl'), 'rb'))
test20 = pickle.load(open(os.path.join(dpath, 'test20.pkl'), 'rb'))

trainp = os.path.join(dpath, 'train_mf.npy')
validp = os.path.join(dpath, 'valid_mf.npy')
test5p = os.path.join(dpath, 'test5_mf.npy')
test10p = os.path.join(dpath, 'test10_mf.npy')
test20p = os.path.join(dpath, 'test20_mf.npy')

all_train = []

def generate_set(train_list, cdict):

    for uid, uinfo in tqdm(cdict.items(), total=len(cdict)):
        for mid, rate in uinfo['rated'].items():
            train_list.append([uid, mid, rate - 1])

def generate_test(cdict, savep):

    test_list = []

    for uid, uinfo in tqdm(cdict.items(), total=len(cdict)):
        for pmid in uinfo['to_predict']:
            test_list.append([uid, pmid])
    
    testnp = np.array(test_list, dtype=np.short)
    print(testnp.shape)
    np.save(savep, testnp)

generate_set(all_train, udict)
generate_set(all_train, test5)
generate_set(all_train, test10)
generate_set(all_train, test20)

trainnp = np.array(all_train, dtype=np.short)
print(trainnp.shape)
np.save(trainp, trainnp)

generate_test(test5, test5p)
generate_test(test10, test10p)
generate_test(test20, test20p)


