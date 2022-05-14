from modules.cf_basic import CFBasic
import pickle
import os

dpath = 'data'
udict = pickle.load(open(os.path.join(dpath, 'udict.pkl'), 'rb'))
mdict = pickle.load(open(os.path.join(dpath, 'mdict.pkl'), 'rb'))
test5 = pickle.load(open(os.path.join(dpath, 'test5.pkl'), 'rb'))
test10 = pickle.load(open(os.path.join(dpath, 'test10.pkl'), 'rb'))
test20 = pickle.load(open(os.path.join(dpath, 'test20.pkl'), 'rb'))

cfb = CFBasic(udict, mdict)

res5 = cfb.cf(test5)
res5.to_csv(os.path.join(dpath, 'score5_cf1.txt'), index=None, header=None, sep=' ')
res10 = cfb.cf(test10)
res10.to_csv(os.path.join(dpath, 'score10_cf1.txt'), index=None, header=None, sep=' ')
res20 = cfb.cf(test20)
res20.to_csv(os.path.join(dpath, 'score20_cf1.txt'), index=None, header=None, sep=' ')
