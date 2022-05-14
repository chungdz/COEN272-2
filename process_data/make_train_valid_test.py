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

trainp = os.path.join(dpath, 'train.npy')
validp = os.path.join(dpath, 'valid.npy')
test5p = os.path.join(dpath, 'test5.npy')
test10p = os.path.join(dpath, 'test10.npy')
test20p = os.path.join(dpath, 'test20.npy')

all_train = []
all_valid = []

def generate_set(train_list, valid_list, cdict):

    for uid, uinfo in tqdm(cdict.items(), total=len(cdict)):
        rlist = [(mid, rate - 1) for mid, rate in uinfo['rated'].items()]
        rlen = len(rlist)
        if rlen <= max_l + 1:
            # generate rlen - 1 dataset
            for i in range(rlen):
                clist = rlist.copy()
                cur_mid, cur_rate =  clist.pop(i)
                remain = max_l - len(clist)
                for _ in range(remain):
                    clist.append(pad)
                
                new_row = []
                new_row.append(uid)
                new_row.append(cur_rate)
                new_row.append(cur_mid)
                for tmid, trate in clist:
                    new_row.append(tmid)
                for tmid, trate in clist:
                    new_row.append(trate)
                assert(len(new_row) == 43)

                if (i + 1) % ratio == 0:
                    valid_list.append(new_row)
                else:
                    train_list.append(new_row)
        else:
            exceed = rlen - max_l
            for i in range(100 * exceed):
                clist = random.sample(rlist, max_l + 1)
                cur_mid, cur_rate =  clist.pop()

                new_row = []
                new_row.append(uid)
                new_row.append(cur_rate)
                new_row.append(cur_mid)
                for tmid, trate in clist:
                    new_row.append(tmid)
                for tmid, trate in clist:
                    new_row.append(trate)
                assert(len(new_row) == 43)
                
                if (i + 1) % ratio == 0:
                    valid_list.append(new_row)
                else:
                    train_list.append(new_row)

def generate_test(cdict, savep):

    test_list = []

    for uid, uinfo in tqdm(cdict.items(), total=len(cdict)):
        rlist = [(mid, rate - 1) for mid, rate in uinfo['rated'].items()]
        rlen = len(rlist)
        remain = max_l - rlen
        for _ in range(remain):
            rlist.append(pad)
        for pmid in uinfo['to_predict']:
            new_row = []
            new_row.append(uid)
            new_row.append(pmid)
            for tmid, trate in rlist:
                new_row.append(tmid)
            for tmid, trate in rlist:
                new_row.append(trate)
            assert(len(new_row) == 42)
            test_list.append(new_row)
    
    testnp = np.array(test_list, dtype=np.short)
    print(testnp.shape)
    np.save(savep, testnp)

generate_set(all_train, all_valid, udict)
generate_set(all_train, all_valid, test5)
generate_set(all_train, all_valid, test10)
generate_set(all_train, all_valid, test20)

trainnp = np.array(all_train, dtype=np.short)
validnp = np.array(all_valid, dtype=np.short)
print(trainnp.shape, validnp.shape)
np.save(trainp, trainnp)
np.save(validp, validnp)

generate_test(test5, test5p)
generate_test(test10, test10p)
generate_test(test20, test20p)


