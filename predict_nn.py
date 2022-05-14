import numpy as np
import argparse
from datasets.config import ModelConfig
from modules.basic_nn import BasicRS
from utils.train_util import set_seed
from torch.utils.data import DataLoader
from datasets.dl import FNNData
import torch.nn.functional as F
import torch
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

def run(cfg, testset, savep):

    set_seed(7)
    # Build Dataloader
    data_loader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model.
    model = BasicRS(cfg.model_info)
    pretrained_model = torch.load(os.path.join(cfg.save_path, "model.ep{}".format(cfg.epoch)), map_location='cpu')
    print(model.load_state_dict(pretrained_model, strict=False))
    model.to(0)
    
    model.eval()  
        
    uids = []
    mids = []
    preds = []
    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), desc="predict"):
            uid_data = data[:, 0]
            mid_data = data[:, 1]
            data = data.type(torch.LongTensor).to(0)
            res = model(data[:, 1:])
            maxidx = res.argmax(dim=-1) + 1
            uids += uid_data.cpu().numpy().tolist()
            mids += mid_data.cpu().numpy().tolist()
            preds += maxidx.cpu().numpy().tolist()

    rlist = []
    for uid, mid, pred in zip(uids, mids, preds):
        rlist.append(uid, mid, pred)
        
    final_df = pd.DataFrame(sorted(rlist, key=lambda x: (x[0], x[1])), columns=['uid', 'mid', 'ratings'])
    final_df.to_csv(os.path.join(cfg.dpath, savep), index=None, header=None, sep=' ')

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data/", type=str,
                        help="root path of all data")
parser.add_argument("--epoch", default=14, type=int, help="training epoch")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size")
parser.add_argument("--save_path", default='data/para/', type=str, help="path to save training model parameters")
args = parser.parse_args()
print('load data')
test5p = os.path.join(args.dpath, 'test5.npy')
test10p = os.path.join(args.dpath, 'test10.npy')
test20p = os.path.join(args.dpath, 'test20.npy')

args.model_info = ModelConfig()

test5set = FNNData(np.load(test5p))
test10set = FNNData(np.load(test10p))
test20set = FNNData(np.load(test20p))

run(args, test5set, 'nn_test5.txt')
run(args, test10set, 'nn_test10.txt')
run(args, test20set, 'nn_test20.txt')


