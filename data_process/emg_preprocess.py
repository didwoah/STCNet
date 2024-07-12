from scipy.io import loadmat
import pandas as pd
import numpy as np
import os
from skimage.measure import block_reduce
import argparse


def parse_option():
    parser = argparse.ArgumentParser(description='Argument for preprocessing')
    parser.add_argument('--path', type=str, required=True, help='Input folder path')
    parser.add_argument('--dataset', type=str, required=True, choices=['nina1', 'nina2', 'nina4'], help='Dataset type')
    return parser.parse_args()

def preprocess(opt):
    dir_path = opt.path
    file_pathlist = []
    dir_list = []
    file_type = []

    if opt.dataset == 'nina1':
        for (root, directories, files) in os.walk(dir_path):
            for d in directories:
                d_path = os.path.join(root,d)
                dir_list.append(d)
            for file in files:
                file_path = os.path.join(root,file)
                file_pathlist.append(file_path)
                if ("E3" in file):
                    file_type.append("E3")
                elif ("E2" in file):
                    file_type.append("E2")
                else:
                    file_type.append("E1")
                    
        train = pd.DataFrame(columns=['stimulus','repetition','subject','normalized'])
        test = pd.DataFrame(columns=['stimulus','repetition','subject','normalized'])

        now = 100;
        start=0;
        idx=-1;
        for j, file_path in enumerate(file_pathlist):
            res = loadmat(file_path)
            res['emg'] = (res['emg']-np.min(res["emg"],axis=0))/(np.max(res["emg"],axis=0)-np.min(res["emg"],axis=0))
            for i in range(len(res['emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start= i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5,7))):
                    idx+=1
                    train.loc[idx,'normalized']=res['emg'][start:i]
                    if (file_type[j]=="E3"):
                        now = now+29
                    elif (file_type[j]=="E2"):
                        now = now+12
                    train.loc[idx,'stimulus'] = now-1
                    train.loc[idx,'repetition'] = f'{res["repetition"][i][0]}'
                    train.loc[idx,'subject'] = int(j/3)

        now = 100;
        start = 0;
        end = 0;
        idx = -1;
        for j,file_path in enumerate(file_pathlist):
            res = loadmat(file_path)
            res['emg'] = (res['emg']-np.min(res["emg"],axis=0))/(np.max(res["emg"],axis=0)-np.min(res["emg"],axis=0))
            for i in range(len(res['emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start= i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6,8,9,10) )):
                    idx+=1
                    test.loc[idx,'normalized'] = res['emg'][start:i]
                    if (file_type[j]=="E3"):
                        now = now+29
                    elif (file_type[j]=="E2"):
                        now = now+12
                    test.loc[idx,'stimulus']=now-1
                    test.loc[idx,'repetition']= f'{res["repetition"][i][0]}'
                    test.loc[idx,'subject'] = int(j/3)

    elif opt.dataset == 'nina2':
        for (root, directories, files) in os.walk(dir_path):
            for d in directories:
                d_path = os.path.join(root,d)
                dir_list.append(d)

            for file in files:
                file_path = os.path.join(root,file)
                file_pathlist.append(file_path)
                if ("E3" in file):
                    file_type.append("E3")
                elif ("E2" in file):
                    file_type.append("E2")
                else:
                    file_type.append("E1")


        train = pd.DataFrame(columns=['stimulus','subject','normalized','sampled_normalized'])
        test = pd.DataFrame(columns=['stimulus','subject','normalized','sampled_normalized'])

        now = 100;
        start = 0;
        idx = -1;
        for j,file_path in enumerate(file_pathlist):
            # gc.collect()
            res = loadmat(file_path)
            res['preprocessed_emg']= (res['preprocessed_emg']-np.min(res['preprocessed_emg'],axis=0))/(np.max(res["preprocessed_emg"],axis=0)-np.min(res["preprocessed_emg"],axis=0))

            for i in range(len(res['preprocessed_emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start = i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5))) or (i==len(res['preprocessed_emg'])-2 and now !=0 and (res['repetition'][i] not in (2,5))):
                    idx+=1
                    train.loc[idx,'stimulus']=now-1
                    train.loc[idx,'subject'] = int(j/3)
                    train.loc[idx,'normalized'] = res['preprocessed_emg'][start:i]
                    train.loc[idx,'sampled_normalized'] = block_reduce(res['preprocessed_emg'][start:i],(20, 1),np.max)

        now = 100;
        start = 0;
        idx = -1;
        for j,file_path in enumerate(file_pathlist):
            # gc.collect()
            res = loadmat(file_path)
            res['preprocessed_emg']= (res['preprocessed_emg']-np.min(res['preprocessed_emg'],axis=0))/(np.max(res["preprocessed_emg"],axis=0)-np.min(res["preprocessed_emg"],axis=0))

            for i in range(len(res['preprocessed_emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start = i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6))) or (i==len(res['preprocessed_emg'])-2 and now !=0 and (res['repetition'][i] not in (1,3,4,6))):
                    idx+=1
                    test.loc[idx,'stimulus']=now-1
                    test.loc[idx,'subject'] = int(j/3)
                    test.loc[idx,'normalized'] = res['preprocessed_emg'][start:i]
                    test.loc[idx,'sampled_normalized'] = block_reduce(res['preprocessed_emg'][start:i],(20, 1),np.max)

    elif opt.dataset == 'nina4':
        for (root, directories, files) in os.walk(dir_path):
            for d in directories:
                d_path = os.path.join(root,d)
                dir_list.append(d)
                
            for file in files:
                file_path = os.path.join(root,file)
                file_pathlist.append(file_path)
                if ("E3" in file):
                    file_type.append("E3")
                elif ("E2" in file):
                    file_type.append("E2")
                else:
                    file_type.append("E1")

        train = pd.DataFrame(columns=['stimulus','subject','normalized','sampled_normalized'])
        test = pd.DataFrame(columns=['stimulus','subject','normalized','sampled_normalized'])

        now = 100;
        start = 0;
        idx = -1;
        for j,file_path in enumerate(file_pathlist):
            # gc.collect()
            res = loadmat(file_path)
            res['preprocessed_emg']= (res['preprocessed_emg']-np.min(res['preprocessed_emg'],axis=0))/(np.max(res["preprocessed_emg"],axis=0)-np.min(res["preprocessed_emg"],axis=0))

            for i in range(len(res['preprocessed_emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start = i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (2,5))) or (i==len(res['preprocessed_emg'])-2 and now !=0 and (res['repetition'][i] not in (2,5))):
                    idx+=1
                    train.loc[idx,'stimulus']=now-1
                    train.loc[idx,'subject'] = int(j/3)
                    train.loc[idx,'normalized'] = res['preprocessed_emg'][start:i]
                    train.loc[idx,'sampled_normalized'] = block_reduce(res['preprocessed_emg'][start:i],(20, 1),np.max)

        now = 100;
        start = 0;
        idx = -1;
        for j,file_path in enumerate(file_pathlist):
            # gc.collect()
            res = loadmat(file_path)
            res['preprocessed_emg']= (res['preprocessed_emg']-np.min(res['preprocessed_emg'],axis=0))/(np.max(res["preprocessed_emg"],axis=0)-np.min(res["preprocessed_emg"],axis=0))

            for i in range(len(res['preprocessed_emg'])-1):
                if(res['stimulus'][i][0] != now):
                    now = res['stimulus'][i][0]
                    start = i
                if(res['stimulus'][i+1][0] != now and now !=0 and (res['repetition'][i] not in (1,3,4,6))) or (i==len(res['preprocessed_emg'])-2 and now !=0 and (res['repetition'][i] not in (1,3,4,6))):
                    idx+=1
                    test.loc[idx,'stimulus']=now-1
                    test.loc[idx,'subject'] = int(j/3)
                    test.loc[idx,'normalized'] = res['preprocessed_emg'][start:i]
                    test.loc[idx,'sampled_normalized'] = block_reduce(res['preprocessed_emg'][start:i],(20, 1),np.max)
    else:
        raise ValueError("Unsupported dataset type.")
    
    # Check if the output directory exists, and if not, create it
    if not os.path.exists('../pkl/'):
        os.makedirs('../pkl/', exist_ok=True)

    train.to_pickle('../pkl/train_{}.pkl'.format(opt.dataset))
    test.to_pickle('../pkl/test_{}.pkl'.format(opt.dataset))

def main():
    opt = parse_option()
    preprocess(opt)

if __name__ == '__main__':
    main()