import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class CsvDate(Dataset):
    def __int__(self):
        csv_path = './dataSet/mydata.csv'
        self.classes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        data = pd.read_csv(csv_path)
        feature_col = ['性别', '年龄段级别']
        feature = data.loc[:, feature_col]
        feature = feature.values
        labels = data.loc[:, ['疾病风险']].values
        self.X = torch.from_numpy(feature)
        self.Y = torch.from_numpy(labels)
        self.len = feature.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return int(self.len)


if __name__ == '__main__':
    # cvs_dataset = CsvDate()
    # train_loader = DataLoader(dataset=cvs_dataset, batch_size=4, shuffle=True)

    data = pd.read_csv('./dataSet/mydata.csv')
    feature_col = ['性别', '年龄段级别']
    feature = data.loc[:, feature_col].values
    labels = data.loc[:, ['疾病风险']].values
    X = torch.from_numpy(feature)
    Y = torch.from_numpy(labels)

    cvs_data = TensorDataset(X, Y)
    data_loader = DataLoader(dataset=cvs_data, batch_size=4, shuffle=True)
    for i, set_1 in enumerate(data_loader):
        print(i)
        x, y = set_1
