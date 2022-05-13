import torch
from model.myModel import myModel
import torch.nn.functional as F

net = myModel()
net.load_state_dict(torch.load('./runs/train/model.pt'))

with torch.no_grad():
    para_1 = input("行政区号：")
    para_2 = input("性别(0-女，1-男): ")
    para_3 = input("年龄段级别：")
    para_list = [int(para_1), int(para_2), int(para_3)]
    # convert to tensor
    features = torch.tensor(para_list, dtype=torch.float32)
    predict = net(features)
    predict = F.softmax(predict, dim=0)
    print(predict)
    print("您的健康评估结果为:\n", "低风险：%.5f, 中风险:%.5f, 高风险：%.5f"
          % (predict[0], predict[1], predict[2]))
