import torch
from torch import nn, optim
import torch.nn.functional as F
from model.myModel import myModel
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from myUtils.plot_loss import plot_list
from myUtils.calAcc import count_equal_num
import sys

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

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

# parameter setting
test_size = 0.2
epochs = 10
batch_size = 512
lr = 0.005
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
myClassifier = myModel().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(myClassifier.parameters(), lr=lr)

# 用于绘制误差变化
train_loss = []
train_epochs_loss = []
acc_epo = []

with tqdm(total=Y.shape[0] * epochs, desc='now_epo_Acc=0.00', file=sys.stdout) as pbar:
    for epoch in range(epochs):
        # dataLoader
        for i, data in enumerate(train_loader):
            x, y = data
            output = myClassifier(x.to(device))
            loss = criterion(output, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss.append(loss.item())  # train loss
            if i == int(x_train.shape[0]/batch_size):
                train_epochs_loss.append(loss.item())
            update = output.shape[0]
            pbar.update(update)

        # print epoch loss
        content = f'epoch:{epoch+1}/{epochs}  loss:{train_epochs_loss[epoch]}'
        tqdm.write(str(content))

        # 每一个epoch 验证一下
        with torch.no_grad():
            acc = 0.0
            all_num = 0
            right_num = 0
            for i, data in enumerate(test_loader):
                x, y = data
                output = myClassifier(x.to(device))
                output = F.softmax(output, dim=0)
                rows = output.shape[0]
                pre_all, pre_right = count_equal_num(y, output, rows=rows)
                all_num += pre_all
                right_num += pre_right
                pbar.update(rows)
            acc = right_num / all_num
            acc_epo.append(acc)
            pbar.set_description('now_epo_Acc=%.3f |' % acc)



print("train completed!")
# plot loss
save_path1 = './runs/train/train_loss.svg'
save_path2 = './runs/train/train_epoch_loss.svg'
save_path3 = './runs/train/acc_epoch.svg'
plot_list(train_loss, 'o-', "train_loss", "LR=0.005 EPOCH=10 OPT=ADAM", "LOSS", "iter", save_path1)
plot_list(train_epochs_loss, 'rD-', "train_epoch_loss", "LR=0.005 EPOCH=10 OPT=ADAM", "LOSS", "epoch", save_path2)
plot_list(acc_epo, 'y^-', "acc", "LR=0.005 EPOCH=10 OPT=ADAM", "ACC", "epoch", save_path3)




# 保存权重
torch.save(myClassifier.state_dict(), './runs/train/model.pt')
# 加载权重
# model = myModel()
# model.load_state_dict(torch.load('./runs/train/model.pt'))
