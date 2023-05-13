import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class Traindataset(Dataset):
    def __init__(self, user_train, time_matrix,dis_matirx,itemnum,maxlen):

        self.user_train = user_train
        self.time_matrix = time_matrix
        self.itemnum = itemnum
        self.dis_matrix = dis_matirx
        self.max_len = maxlen

    def __getitem__(self, user_idx):

        user_idx +=1 # the default min value of user_idx is 0, however, the idx of user_train begins at 1
        seq = np.zeros([self.max_len], dtype=np.int32)
        time_seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = self.user_train[user_idx][-1][0]

        idx = self.max_len - 1
        ts = set(map(lambda x: x[0], self.user_train[user_idx]))
        for i in reversed(self.user_train[user_idx][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        user_time_matrix = self.time_matrix[user_idx]
        user_dis_matrix = self.dis_matrix[user_idx]

        user_idx = torch.tensor(user_idx, dtype=torch.long)
        seq = torch.tensor(seq, dtype=torch.long)
        time_seq = torch.tensor(time_seq, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.long)
        neg = torch.tensor(neg, dtype=torch.long)
        user_time_matrix = torch.tensor(user_time_matrix, dtype=torch.long)
        user_dis_matrix = torch.tensor(user_dis_matrix, dtype=torch.long)

        return user_idx,seq,time_seq,pos,neg,user_time_matrix,user_dis_matrix

    def __len__(self):
        return len(self.user_train)
    
class Validdataset(Dataset):
    def __init__(self, all_seqs, all_time_matrix, all_distance_matrix,all_labels):

        self.seqs = all_seqs
        self.time_matrix = all_time_matrix
        self.dis_matrix = all_distance_matrix
        self.labels = all_labels

    def __getitem__(self, user_idx):

        seq = self.seqs[user_idx]
        time_matrix = self.time_matrix[user_idx]
        dis_matrix = self.dis_matrix[user_idx]
        labels = self.labels[user_idx]

        user_idx = torch.tensor(user_idx, dtype=torch.long)
        seq = torch.tensor(seq, dtype=torch.long)
        time_matrix = torch.tensor(time_matrix, dtype=torch.long)
        dis_matrix = torch.tensor(dis_matrix, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return user_idx,seq,time_matrix,dis_matrix,labels

    def __len__(self):
        return len(self.seqs)
    
class Testdataset(Dataset):
    def __init__(self, all_seqs, all_time_matrix, all_distance_matrix,all_labels):

        self.seqs = all_seqs
        self.time_matrix = all_time_matrix
        self.dis_matrix = all_distance_matrix
        self.labels = all_labels

    def __getitem__(self, user_idx):

        seq = self.seqs[user_idx]
        time_matrix = self.time_matrix[user_idx]
        dis_matrix = self.dis_matrix[user_idx]
        labels = self.labels[user_idx]

        user_idx = torch.tensor(user_idx, dtype=torch.long)
        seq = torch.tensor(seq, dtype=torch.long)
        time_matrix = torch.tensor(time_matrix, dtype=torch.long)
        dis_matrix = torch.tensor(dis_matrix, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return user_idx,seq,time_matrix,dis_matrix,labels

    def __len__(self):
        return len(self.seqs)
    