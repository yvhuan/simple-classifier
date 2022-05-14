import datetime
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model.myModel import myModel
import torch.nn.functional as F
from myUtils.calAcc import count_equal_num
from tqdm import tqdm


def save_acc_result(acc=0.0, save_path=''):
    with open(save_path, 'w') as f:
        print('acc: ', acc, file=f)
        # 加上时间戳
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, file=f)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    acc_save_path = './runs/test/acc_result.txt'
    # data loading
    data = pd.read_csv('./data/dataSet/mydata.csv')
    feature_col = ['行政区划代码', '性别', '年龄段级别']
    feature = data.loc[:, feature_col].values
    labels = data.loc[:, ['疾病风险']].values
    # convert labels to n*3
    y_list = []
    for i, label in enumerate(labels):
        if label == [0]:
            y_list.append([1.0, 0.0, 0.0])
            continue
        if label == [1]:
            y_list.append([0.0, 1.0, 0.0])
            continue
        if label == [2]:
            y_list.append([0.0, 0.0, 1.0])
    Y = np.array(y_list)
    X = torch.from_numpy(feature.astype(np.float32))
    Y = torch.from_numpy(Y)
    dataset = TensorDataset(X, Y)
    test_loader = DataLoader(dataset=dataset, batch_size=512, shuffle=False)

    # out by net and validate
    acc = 0.0
    net = myModel().to(device)
    net.load_state_dict(torch.load('./runs/train/model.pt'))
    with tqdm(total=Y.shape[0]) as pbar:
        with torch.no_grad():
            all_num = 0
            right_num = 0
            for i, data in enumerate(test_loader):
                x, y = data
                output = net(x.to(device))
                output = F.softmax(output, dim=0)
                rows = output.shape[0]
                pre_all, pre_right = count_equal_num(y, output, rows=rows)
                all_num += pre_all
                right_num += pre_right
                pbar.update(rows)
            acc = right_num / all_num
    # save acc result
    print('acc:%.3f' % acc)
    print('acc result is saved to ', acc_save_path)
    save_acc_result(acc, acc_save_path)
