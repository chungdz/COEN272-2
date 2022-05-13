import pandas as pd
import pickle
import os
from tqdm import tqdm

dpath = 'data'

def load_test(fname, output):
    df = pd.read_csv(os.path.join(dpath, fname), sep=' ', header=None, names=['uid', 'mid', 'rating'])
    
    test = {}
    for uid, mid, rating in df.values:
        if uid not in test:
            test[uid] = {
                'rated': {},
                'to_predict': []
            }
        if rating != 0:
            test[uid]['rated'][mid] = rating
        else:
            test[uid]['to_predict'].append(mid)
    
    for uid, uinfo in test.items():
        uinfo['avg'] = sum(uinfo['rated'].values()) / len(uinfo['rated'])
        
    pickle.dump(test, open(os.path.join(dpath, output), 'wb'))
    return test

trainset = pd.read_csv(os.path.join(dpath, 'train.txt'), sep=' ', header=None, names=['uid', 'mid', 'rating'])

udict = {}
mdict = {}
for uid, mid, rating in trainset.values:
    if uid not in udict:
        udict[uid] = {
            'rated': {}
        }
    udict[uid]['rated'][mid] = rating

    if mid not in mdict:
        mdict[mid] = {
            'rated': {}
        }
    mdict[mid]['rated'][uid] = rating
    assert(rating >= 1 and rating <= 5)

for uid, uinfo in udict.items():
    uinfo['avg'] = sum(uinfo['rated'].values()) / len(uinfo['rated'])
pickle.dump(udict, open(os.path.join(dpath, 'udict.pkl'), 'wb'))

for mid, minfo in mdict.items():
    minfo['avg'] = sum(minfo['rated'].values()) / len(minfo['rated'])
pickle.dump(mdict, open(os.path.join(dpath, 'mdict.pkl'), 'wb'))

test5 = load_test('test5.txt', 'test5.pkl')
test10 = load_test('test10.txt', 'test10.pkl')
test20 = load_test('test20.txt', 'test20.pkl')

print(len(udict), len(mdict), len(test5), len(test10), len(test20))
