import numpy as np
from tqdm import tqdm
import math
import pandas as pd

class CFBasicCos:

    def __init__(self, database: dict, mvdb: dict):
        self.s = database
        self.m = mvdb
    
    def cf(self, testd: dict, k=10):

        final_res = []

        for uid, uinfo in tqdm(testd.items()):
            # calculate weights
            wl = []
            for dbuid, dbuinfo in self.s.items():
                curw = self.cal_weights(dbuinfo, uinfo)
                if curw != 0:
                    wl.append([curw, dbuid])
            # sort weights
            # use absolute value, in descending order
            swl = sorted(wl, key=lambda x: -abs(x[0]))
            # predict
            for predmid in uinfo['to_predict']:
                score = self.cal_score(predmid, swl, uinfo, k)
                score = round(score)
                if score < 1:
                    score = 1
                if score > 5:
                    score = 5
                final_res.append([uid, predmid, score])
        
        final_df = pd.DataFrame(sorted(final_res, key=lambda x: (x[0], x[1])), columns=['uid', 'mid', 'ratings'])
        return final_df
    
    def cal_score(self, predmid, swl, uinfo, k, print_=False):

        topdbw = []
        topdbr = []
        for uwieght, dbuid in swl:
            if not predmid in self.s[dbuid]['rated']:
                continue
            topdbw.append(uwieght)
            topdbr.append(self.s[dbuid]['rated'][predmid])
            if len(topdbw) >= k:
                break
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
            print(topdbw, topdbr, score, uinfo['avg'])

        return score

    
    def cal_weights(self, dbuinfo, testuinfo):
        intersect = []
        for mid in testuinfo['rated']:
            if mid in dbuinfo['rated']:
                intersect.append(mid)
        
        if len(intersect) == 0:
            return 0
        
        dbrate = [dbuinfo['rated'][mid] for mid in intersect]
        testrate = [testuinfo['rated'][mid] for mid in intersect]
        # final weight
        cs = self.cosine(dbrate, testrate)
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