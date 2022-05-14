import torch
import numpy as np


def count_equal_num(label=torch.tensor([[]]), out=torch.tensor([[]]), rows=0):
    label = label.cpu().numpy().astype(np.int16)
    out = out.cpu().numpy()
    max_list = np.argmax(out, axis=1)  # row内比较，return column
    new_out = np.zeros(out.shape)
    for i in range(rows):
        new_out[i][max_list[i]] = 1
    new_out = new_out.astype(np.int16)
    res = np.logical_xor(label, new_out)
    count_all = label.shape[0]
    count_right = count_all - np.count_nonzero(res) / 2
    return count_all, int(count_right)


# test
if __name__ == '__main__':
    label = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
    out = torch.tensor([[0.9, 0.1, 0], [0.4, 0.5, 0.1], [0.6, 0.2, 0.2], [0, 0, 1]])
    batch_size = 3
    count_all, count_right = count_equal_num(label, out, batch_size)
    print(count_all, count_right)  # 4 2
