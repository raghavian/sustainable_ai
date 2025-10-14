### Import torch and utils
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

class AerialNIST(Dataset):
    def __init__(self,path='../data/',data=['AerialNIST_00.pt','AerialNIST_01.pt','AerialNIST_02.pt','AerialNIST_03.pt'], \
                 normalize=True):
        super().__init__()

        # Load data from the pytorch file
        xs, ys = zip(*(torch.load(path+p, map_location="cpu") for p in data))
        self.data = torch.cat(xs, dim=0)
        self.target = torch.cat(ys, dim=0)

        ### Normalize intensities to be between 0-1
        if normalize:
            self.data = self.data/ self.data.max() ##########

    def __len__(self):
        ### Method to return number of data points
        return len(self.target)

    def __getitem__(self,index):
        ### Method to fetch indexed element
        return self.data[index], self.target[index].type(torch.FloatTensor)


class FAIRYTALES(Dataset):
    def __init__(self, path='../data/fairytales.txt'): 
        self.lines = Path(path).read_text(encoding="utf-8").splitlines()

    def __len__(self): 
        return len(self.lines)
        
    def __getitem__(self, i): 
        return self.lines[i]
