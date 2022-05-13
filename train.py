import torch
from torch import nn, optim
from model.myModel import myModel
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from plot_loss import plot_loss

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
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_loader = DataLoader(dataset=train_data, batch_size=512, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=512, shuffle=False)

myClassifier = myModel().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(myClassifier.parameters(), lr=0.005)
epochs = 10

# 用于绘制误差变化
train_loss = []
test_loss = []
train_epochs_loss = []
test_epochs_loss = []

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
        if i % 200 == 0:
            print('Epoch:%d/%d, iter: %d, Loss: %.7f' % (epoch, epochs, i, loss.item()))

print("train completed!")
# plot loss
plot_loss(train_loss)

# 保存权重
torch.save(myClassifier.state_dict(), './runs/train/model.pt')
# 加载权重
# model = myModel()
# model.load_state_dict(torch.load('./runs/train/model.pt'))
