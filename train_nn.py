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
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

def run(cfg, train_dataset, valid_dataset):
    """
    train and evaluate
    :param args: config
    :param rank: process id
    :param device: device
    :param train_dataset: dataset instance of a process
    :return:
    """
    
    set_seed(7)
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model.
    model = BasicRS(cfg.model_info)
    # for continuous model training
    if cfg.start_epoch != -1:
        pretrained_model = torch.load(os.path.join(cfg.save_path, "model.ep{}".format(cfg.start_epoch)), map_location='cpu')
        print(model.load_state_dict(pretrained_model, strict=False))
    model.to(0)
    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = cfg.epoch * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr * pow(cfg.lr_shrink, cfg.start_epoch + 1))
    steplr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=cfg.lr_shrink)
    # Training and validation
    for epoch in range(cfg.epoch):
        print("lr in this epoch:", steplr.get_last_lr())
        if epoch <= cfg.start_epoch:
            continue
        train(cfg, epoch, model, train_data_loader, optimizer, steps_one_epoch)
        validate(cfg, model, valid_data_loader)
        steplr.step()

def train(cfg, epoch, model, loader, optimizer, steps_one_epoch):
    model.train()
    model.zero_grad()
    enum_dataloader = tqdm(enumerate(loader), total=len(loader), desc="EP-{} train".format(epoch))
    mean_loss = 0
    loss_list = []
    for index, data in enum_dataloader:
        # 1. Forward
        data = data.type(torch.LongTensor).to(0)
        pred = model(data[:, 2:])
        loss = F.cross_entropy(pred, data[:, 1])

        # 3.Backward.
        loss.backward()
        # try gradient clipper to avoid gradient explosion
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        model.zero_grad()
        # index add
        mean_loss += loss.item()
        loss_list.append(loss.item())
        if str(loss.item()) == 'nan':
            print(loss_list[-100:])
            print(index)
            exit()
            
        if index % cfg.show_batch == 0 and index > 0:
            cur_mean_loss = mean_loss / cfg.show_batch
            enum_dataloader.set_description("EP-{} train, batch {} loss is {}".format(epoch, index, cur_mean_loss))
            mean_loss = 0
    
    torch.save(model.state_dict(), os.path.join(cfg.save_path, "model.ep{}".format(epoch)))

def validate(cfg, model, valid_data_loader):
    model.eval()  
        
    labels = []
    preds = []
    with torch.no_grad():
        for data in tqdm(valid_data_loader, total=len(valid_data_loader), desc="valid"):
            label_data = data[:, 1]
            data = data.type(torch.LongTensor).to(0)
            res = model(data[:, 2:])
            maxidx = res.argmax(dim=-1)
            labels += label_data.cpu().numpy().tolist()
            preds += maxidx.cpu().numpy().tolist()

    print("AUC:", accuracy_score(labels, preds))

parser = argparse.ArgumentParser()
parser.add_argument("--dpath", default="data/", type=str,
                        help="root path of all data")
parser.add_argument("--epoch", default=20, type=int, help="training epoch")
parser.add_argument("--batch_size", default=128, type=int, help="batch_size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--save_path", default='data/para/', type=str, help="path to save training model parameters")
parser.add_argument("--show_batch", default=20, type=int, help="averaged batch loss will change each show_batch times")
parser.add_argument("--lr_shrink", default=0.9, type=float, help="learning rate will multiply this shrinking number after one epoch")
parser.add_argument("--start_epoch", default=-1, type=int, help='''whether to start training from scratch 
                            or load parameter saved before and continue training. For example, if start_epoch=0, then model will load parameter 
                            save_path/model.ep0 and start the second epoch of training''')
args = parser.parse_args()
print('load data')
trainp = os.path.join(args.dpath, 'train.npy')
devp = os.path.join(args.dpath, 'valid.npy')

args.model_info = ModelConfig()

trainset = FNNData(np.load(trainp))
validset = FNNData(np.load(devp))

run(args, trainset, validset)


