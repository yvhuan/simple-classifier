import torch
from torch import nn, optim
import torch.nn.functional as F


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    a = torch.tensor([1.0, 0.0])  #
    res = F.softmax(a, dim=0)
    print(a)
    print(res)

    classifier = myModel()
    print(classifier)
