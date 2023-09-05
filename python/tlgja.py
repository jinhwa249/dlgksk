# class Torch:
# #     def __init__(self, name):
# #         self.name = name
        
# #     def print(self):
# #         print('this is torch class'+self.name)
        
# # def main():
# #     print('this is my first python programming')
# #     torch1 = Torch('하나')
# #     torch1.print()
# #     print(torch1.name)
    
# from torch import Torch

# class PyTorch(Torch):
#     def __init__(self, name):
#         super().__init__(name)
#         self.pyname = 'py이하나'
        
#     def sud_print(self):
#         print('파이네임:' + self.pyname+ '\t이름은 :' + self.name)
        
#     def __eq__(self, value):
#         return self.pyname == value.pyname
    
# def main():
#     t = Torch('이하나')
#     t2 = PyTorch('이이하나')
#     t3 = pyToch('이이하나')
#     t2.print()
#     t2.print()
#     print(t2 == t3)
    
# if __name__ == '__name__':
#     main()

# import torch
# b = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.int8)
# a = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.int8)+3
# print(a.cpu().numpy())
# print(a[1][0])
# print(a[1, 0])
# print(a[0][1:3])
# print(b)
# print(a.shape)
# print(a+b)
# c = a+b
# print(c.view(8,1))
# print(c.view(1,8))
# print(c.view(2,-1,2))

#  1, 2, 3, 4, 5, 6
#  name, math, kor, sci
#  choi
#  kim
#  lee
#  pakr

# import torch
# import pandas as pd

# date = pd.read_csv('pytorch/test.csv')
# # print(date,keys())
# torch_date = torch.from_numpy(date['name'].values)
# print(torch_date)

import pandas as pd
import torch
from torch.utils.date import Dataset
from torch.utils.date import DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.label = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        sample = torch.tensor(self.label.iloc[idx,0:3]).int()
        iabel = torch.tensor(self.label.iloc[idx,3]).int()
        return sample, label

tensor_dataset = MyDataset('pytorch/test.csv'11)
dataset = DateLoader(tensor_dataset, batch_size=4, shuffle=True)

for i, data in enumerate(dataset, 0):
    print(i, data[0])





