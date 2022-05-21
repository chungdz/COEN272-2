import numpy as np
from tqdm import tqdm
import math
import pandas as pd

class CFItem:

    def __init__(self, database: dict, mvdb: dict):
        self.s = database
        self.m = mvdb
        for mid, minfo in self.m.items():
            minfo['Adjusted_rate'] = {}
            for uid, rating in minfo['rated'].items():
                minfo['Adjusted_rate'][uid] = rating - self.s[uid]['avg']
        self.epsilon = 1e-4
    
    def cf(self, testd: dict):

        final_res = []

        for uid, uinfo in tqdm(testd.items()):
            # uinfo['Adjusted_rate'] = {}
            # for mid, rating in uinfo['rated'].items():
            #     uinfo['Adjusted_rate'][mid] = rating - uinfo['avg']
            
            for predmid in uinfo['to_predict']:
                # calculate weights
                wl = []
                for his_mid in uinfo['rated']:
                    curw = self.cal_weights(predmid, his_mid)
                    if curw != 0:
                        wl.append([curw, his_mid])
                # sort weights
                # use absolute value, in descending order
                score = self.cal_score(predmid, wl, uinfo)
                score = round(score)
                if score < 1:
                    score = 1
                if score > 5:
                    score = 5
                final_res.append([uid, predmid, score])
        
        final_df = pd.DataFrame(sorted(final_res, key=lambda x: (x[0], x[1])), columns=['uid', 'mid', 'ratings'])
        return final_df
    
    def cal_score(self, predmid, wl, uinfo, print_=False):

        topdbw = []
        topdbr = []
        for mwieght, smid in wl:
            topdbw.append(mwieght)
            topdbr.append(uinfo['rated'][smid])
        # if no similar one
        if len(topdbw) == 0:
            if predmid in self.m:
                score = self.m[predmid]['avg']
            else:
                score = 3
        else:
            score = self.inner_product(topdbw, topdbr) / sum([abs(n) for n in topdbw])
        
        if print_:
            print(predmid)
            print(topdbw, topdbr, score)

        return score

    
    def cal_weights(self, mid1, mid2):
        if mid1 not in self.m or mid2 not in self.m:
            return 0

        intersect = []
        for uid in self.m[mid2]['Adjusted_rate']:
            if uid in self.m[mid1]['Adjusted_rate']:
                intersect.append(uid)
        
        if len(intersect) == 0:
            return 0
        
        r1 = [self.m[mid1]['Adjusted_rate'][uid] for uid in intersect]
        r2 = [self.m[mid2]['Adjusted_rate'][uid] for uid in intersect]
        # first inner product to check relations
        relation = self.inner_product(r1, r2)
        # add one smooth
        if relation >= 0:
            r1.append(self.epsilon)
            r2.append(self.epsilon)
        else:
            r1.append(-self.epsilon)
            r2.append(-self.epsilon)
        # final weight
        cs = self.cosine(r1, r2)
        return cs
    
    # judge whether two array of rates are positive related or negative related
    def inner_product(self, a1, a2):
        res = 0
        for n1, n2 in zip(a1, a2):
            res += n1 * n2
        return res
    
    def cosine(self, a1, a2):
        inp = self.inner_product(a1, a2)
        a1_size = math.sqrt(sum(n ** 2 for n in a1))
        a2_size = math.sqrt(sum(n ** 2 for n in a2))
        return inp / (a1_size * a2_size)