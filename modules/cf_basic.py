import numpy as np
from tqdm import tqdm
import math
import pandas as pd

class CFBasic:

    def __init__(self, database: dict, mvdb: dict):
        self.s = database
        self.m = mvdb
        for uid, uinfo in self.s.items():
            uinfo['avg_rated'] = {}
            for mid, rating in uinfo['rated'].items():
                uinfo['avg_rated'][mid] = rating - uinfo['avg']
        self.epsilon = 1e-4
    
    def cf(self, testd: dict, k=10):

        final_res = []

        for uid, uinfo in tqdm(testd.items()):
            uinfo['avg_rated'] = {}
            for mid, rating in uinfo['rated'].items():
                uinfo['avg_rated'][mid] = rating - uinfo['avg']
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
            if not predmid in self.s[dbuid]['avg_rated']:
                continue
            topdbw.append(uwieght)
            topdbr.append(self.s[dbuid]['avg_rated'][predmid])
            if len(topdbw) >= k:
                break
        # if no similar one
        if len(topdbw) == 0:
            if predmid in self.m:
                score = self.m[predmid]['avg']
            else:
                score = 3
        else:
            score = uinfo['avg'] + self.inner_product(topdbw, topdbr) / sum([abs(n) for n in topdbw])
        
        if print_:
            print(predmid)
            print(topdbw, topdbr, score, uinfo['avg'])

        return score

    
    def cal_weights(self, dbuinfo, testuinfo):
        intersect = []
        for mid in testuinfo['avg_rated']:
            if mid in dbuinfo['avg_rated']:
                intersect.append(mid)
        
        if len(intersect) == 0:
            return 0
        
        dbrate = [dbuinfo['avg_rated'][mid] for mid in intersect]
        testrate = [testuinfo['avg_rated'][mid] for mid in intersect]
        # first inner product to check relations
        relation = self.inner_product(dbrate, testrate)
        # add one smooth
        if relation >= 0:
            dbrate.append(self.epsilon)
            testrate.append(self.epsilon)
        else:
            dbrate.append(-self.epsilon)
            testrate.append(-self.epsilon)
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