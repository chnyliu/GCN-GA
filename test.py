import numpy as np
from torch import Tensor, tensor
import torch.nn.functional as F
import torch
import torch.nn as nn

A = [0.881, 0.888, 0.888]
print(np.mean(A), np.std(A))
#
# A = torch.randn(3, 3)
# c = torch.tensor([1, 2, 0])
# B = F.log_softmax(A, 1)
# loss1 = F.nll_loss
# loss = loss1(B, c)
# print(loss)
# loss2 = torch.nn.CrossEntropyLoss()
# loss = loss2(A, c)
# print(loss)
# x_input = torch.randn(3, 3)
# print('x_input:\n', x_input)
# y_target = torch.tensor([1, 2, 0])
# 计算输⼊softmax，此时可以看到每⼀⾏加到⼀起结果都是1
# softmax_func = nn.Softmax(dim=1)
# soft_output = softmax_func(x_input)
# print('soft_output:\n', soft_output)
# # 在softmax的基础上取log
# logsoftmax_output = torch.log(soft_output)
# print('log_output:\n', logsoftmax_output)
# # nn.NLLLoss()的计算过程：
# loss = logsoftmax_output[0, 1] + logsoftmax_output[1, 2] + logsoftmax_output[2, 0]
# a = ((-loss) / logsoftmax_output.size()[0])
# print(a)
# # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
# nllloss_func = nn.NLLLoss()
# nlloss_output = nllloss_func(logsoftmax_output, y_target)
# print('nlloss_output:\n', nlloss_output)
# # 直接使⽤pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是⼀样
# crossentropyloss = nn.CrossEntropyLoss()
# crossentropyloss_output = crossentropyloss(x_input, y_target)
# print('crossentropyloss_output:\n', crossentropyloss_output)
