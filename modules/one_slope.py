import numpy as np
from tqdm import tqdm
import math
import pandas as pd

class OneSlope:

    def __init__(self, database: dict, mvdb: dict):
        self.s = database
        self.m = mvdb
    
    def cf(self, testd: dict):

        final_res = []

        for uid, uinfo in tqdm(testd.items()):
            
            for predmid in uinfo['to_predict']:
                if predmid in self.m:
                    score = self.cal_score(predmid, uinfo)
                else:
                    score = uinfo['avg']
                score = round(score)
                if score < 1:
                    score = 1
                if score > 5:
                    score = 5
                final_res.append([uid, predmid, score])
        
        final_df = pd.DataFrame(sorted(final_res, key=lambda x: (x[0], x[1])), columns=['uid', 'mid', 'ratings'])
        return final_df
    
    def cal_score(self, predmid, uinfo, print_=False):

        his_mid = uinfo['rated']
        pred_info = self.m[predmid]
        devs = []
        dev_len = []
        urating = []
        for hm in his_mid:
            if hm in pred_info['dev']:
                devs.append(pred_info['dev'][hm])
                dev_len.append(pred_info['dev_len'][hm])
                urating.append(uinfo['rated'][hm])
        
        if len(devs) == 0:
            score = self.m[predmid]['avg']
            return score
        
        total = 0
        for dev_ji, card, ur in zip(devs, dev_len, urating):
            total += (dev_ji + ur) * card

        return total / sum(dev_len)