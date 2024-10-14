from torch.utils.data import Dataset
import numpy as np
import torch
        

class NinaDataset(Dataset):
    def __init__(self, dataframe, dataset = 'nina1', model = 'EMGHandNet', transform = None):
        
        if dataset == 'nina1':
            self.channel = 10
        elif dataset == 'nina2' or dataset == 'nina4':
            self.channel = 12
        else:
            raise TypeError(f"Unsupported dataset name '{dataset}'. Please choose either 'nina1', 'nina2', or 'nina4'.")
        
        self.dataframe = dataframe
        self.transform = transform
        self.model = model

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        target_row = self.dataframe.iloc[idx]
        data = target_row['normalized'][0:500]

        "Zero-Padding"
        if len(data)<500:
            data = np.concatenate((data,np.zeros(((500-len(data)),self.channel))),axis=0)
        
        labels = torch.tensor(target_row['stimulus'],dtype=torch.long)

        subjects = torch.tensor(target_row['subject'],dtype=torch.long)

        if self.transform:
            inputs = self.transform(torch.tensor(data, dtype = torch.float))
        else:
            inputs = torch.tensor(data.reshape((25,20,self.channel)), dtype=torch.float)
        
        return inputs, labels, subjects