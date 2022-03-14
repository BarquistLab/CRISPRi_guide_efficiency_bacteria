############################################
# imports
############################################

import torch
from torch.utils.data import Dataset
import numpy as np


############################################
# Dataset for training with y
############################################

class CrisprDatasetTrain(Dataset):

    def __init__(self, X, Y, features):

        self.X_sequence_40nt = X["sequence_30nt"]
        self.X_features = X.loc[:,features]

        self.Y = Y

    def __len__(self):
        return self.X_features.shape[0]

    def __getitem__(self, idx):
        x_sequence_40nt = torch.FloatTensor(self.X_sequence_40nt.iloc[idx].tolist())
        x_features = torch.FloatTensor(self.X_features.iloc[idx].tolist())

        y = torch.tensor(np.float32(self.Y.iloc[idx]))

        return  x_sequence_40nt,x_features, y

    
############################################
# Dataset for evaluation wo y
############################################
  
class CrisprDatasetEval(Dataset):

    def __init__(self, X, features):

        self.X_sequence_40nt = X["sequence_30nt"]
        self.X_features = X.loc[:,features]

    def __len__(self):
        return self.X_sequence_40nt.shape[0]

    def __getitem__(self, idx):

        x_sequence_40nt = torch.FloatTensor(self.X_sequence_40nt.iloc[idx].tolist())
        x_features = torch.FloatTensor(self.X_features.iloc[idx].tolist())

        return  x_sequence_40nt, x_features 