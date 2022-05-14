from torch.utils.data.dataset import Dataset
import torch
 
class FNNData(Dataset):
    def __init__(self, dataset):
        self.dataset = torch.ShortTensor(dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
 
    def __len__(self):
        return self.dataset.shape[0]
