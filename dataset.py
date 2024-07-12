from torch.utils.data import Dataset
import numpy as np
import torch

class Nina1Dataset(Dataset):
    def __init__(self, dataframe, mode = 'labels', model = 'STCNet', transform = None):
        self.dataframe = dataframe
        self.mode = mode
        self.transform = transform
        self.model = model
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        target_row = self.dataframe.iloc[idx]
        data = target_row['normalized'][0:500]
        # means = target_row['normalized_mean']
        #means = target_row['total_means']
        "Zero-Padding"
        if len(data)<500:
            data = np.concatenate((data,np.zeros(((500-len(data)),10))),axis=0)

        "Division data by time-segment"
        # #noised = data + 0.05*np.random.normal(-means,stds,data.shape)
        # noised = data + 1*np.random.uniform(-means,means,data.shape)
        # noised_data = torch.tensor(np.transpose(noised.reshape((20,25,10)),(0,2,1)),dtype=torch.float)
        # noised_data = noised_data.flatten(1)
        
        labels = torch.tensor(target_row['stimulus'],dtype=torch.long)

        subjects = torch.tensor(target_row['subject'],dtype=torch.long)

        if self.transform:
            inputs = self.transform(torch.tensor(data, dtype = torch.float))
        else:
            if self.model == 'EMGHandNet':
                inputs = torch.tensor(np.transpose(data.reshape((25,20,10)),(0,2,1)),dtype=torch.float)
            elif self.model == 'EvCNN':
                inputs = torch.tensor(data.reshape((25,20,10)), dtype=torch.float)
            elif self.model == 'STCNet':
                inputs = torch.tensor(data.reshape((25,20,10)), dtype=torch.float)
        
        if self.mode == 'labels':
            return inputs, labels
        elif self.mode == 'subjects':
            return inputs, subjects
        elif self.mode == 'multi':
            return inputs, labels, subjects
        

class Nina2Dataset(Dataset):
    def __init__(self, dataframe, sampled = False, mode = 'labels', model = 'EMGHandNet', transform = None):
        self.dataframe = dataframe
        self.sampled = sampled
        self.mode = mode
        self.transform = transform
        self.model = model
        
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        
        trial_len = 10000
        Ts = 400
        if self.sampled:
            trial_len = 500
            Ts = 20

        target_row = self.dataframe.iloc[idx]
        data = target_row['sampled_normalized'][0: trial_len]
        # means = target_row['normalized_mean']
        #means = target_row['total_means']
        "Zero-Padding"
        if len(data)< trial_len:
            data = np.concatenate((data,np.zeros(((trial_len-len(data)),12))),axis=0)
        
        labels = torch.tensor(target_row['stimulus'],dtype=torch.long)

        subjects = torch.tensor(target_row['subject'],dtype=torch.long)

        if self.transform:
            inputs = self.transform(torch.tensor(data, dtype = torch.float))
        else:
            if self.model == 'EMGHandNet':
                inputs = torch.tensor(np.transpose(data.reshape((25,Ts,12)),(0,2,1)),dtype=torch.float)
            elif self.model == 'EvCNN':
                 inputs = torch.tensor(data.reshape((25,Ts,12)), dtype=torch.float)
            elif self.model == 'STCNet':
                inputs = torch.tensor(data.reshape((25,Ts,12)), dtype=torch.float)
        
        if self.mode == 'labels':
            return inputs, labels
        elif self.mode == 'subjects':
            return inputs, subjects
        elif self.mode == 'multi':
            return inputs, labels, subjects
